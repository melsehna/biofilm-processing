import os
import sys
import io
import time
import glob
import re
import threading

import numpy as np
import tifffile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QTextEdit,
)
from PySide6.QtCore import QObject, QThread, Signal


class LogStream(io.StringIO):
    """Captures stdout and emits lines as Qt signals."""
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def write(self, text):
        if text.strip():
            self._signal.emit(text.rstrip())

    def flush(self):
        pass


class ProcessingWorker(QObject):
    progress = Signal(str, int, int, str, str)   # plate_name, plate_idx, total_plates, well_id, stage
    well_progress = Signal(int, int)              # well_idx, total_wells
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, state_dict, stop_event):
        super().__init__()
        self._state = state_dict
        self._stop = stop_event

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = LogStream(self.log)

        try:
            self._run_pipeline()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            self.finished.emit()

    def _run_pipeline(self):
        import csv as csv_mod
        from multiWellAnalysis.processing.analysis_main import timelapse_processing

        plates = self._state['plates']
        total_plates = len(plates)
        s = self._state

        for plate_idx, plate_path in enumerate(plates):
            if self._stop.is_set():
                self.log.emit('Cancelled by user.')
                return

            plate_name = os.path.basename(plate_path)
            self.log.emit(f'\n{"="*60}')
            self.log.emit(f'Plate {plate_idx+1}/{total_plates}: {plate_name}')
            self.log.emit(f'{"="*60}')

            wells = self._discover_wells(plate_path)
            if not wells:
                self.log.emit(f'  No wells found in {plate_name}, skipping.')
                continue

            outdir = os.path.join(plate_path, 'processedImages')
            os.makedirs(outdir, exist_ok=True)

            # Index: one row per well, built up across stages
            index = {}  # well_id → dict of paths
            well_items = list(wells.items())
            total_wells = len(well_items)

            # ── Stage 1: Processing ──
            self.log.emit(f'\n  --- Stage 1: Processing ({total_wells} wells) ---')
            for well_idx, (well_id, well_files) in enumerate(well_items):
                if self._stop.is_set():
                    self.log.emit('Cancelled by user.')
                    return
                self.progress.emit(plate_name, plate_idx, total_plates, well_id, 'Processing')
                self.well_progress.emit(well_idx, total_wells)
                t0 = time.time()

                try:
                    m = re.match(r'^[A-P]\d+(_\d+)$', well_id)
                    mag = m.group(1) if m else ''

                    stack = self._load_stack(well_files)

                    block_diam, fixed_thresh, dust_correction = self._get_params(s, mag)

                    self.log.emit(f'  {well_id}: preprocessing + registration...')
                    masks, biomass, od_mean = timelapse_processing(
                        images=stack,
                        block_diameter=block_diam,
                        ntimepoints=stack.shape[2],
                        shift_thresh=s['shiftThresh'],
                        fixed_thresh=fixed_thresh,
                        dust_correction=dust_correction,
                        outdir=plate_path,
                        filename=well_id,
                        image_records=None,
                        fftStride=s['fftStride'],
                        downsample=s['downsample'],
                        skip_overlay=True,
                    )
                    del stack

                    # Save biomass CSV
                    biomass_path = os.path.join(outdir, f'{well_id}_biomass.csv')
                    import pandas as pd
                    pd.DataFrame({'frame': range(len(biomass)), 'biomass': biomass}).to_csv(
                        biomass_path, index=False
                    )

                    index[well_id] = {
                        'plate': plate_name,
                        'plate_path': plate_path,
                        'well': well_id,
                        'mag': mag,
                        'registered_raw': os.path.join(outdir, f'{well_id}_registered_raw.tif'),
                        'processed': os.path.join(outdir, f'{well_id}_processed.tif'),
                        'masks': os.path.join(outdir, f'{well_id}_masks.npz'),
                        'biomass': biomass_path,
                    }

                    elapsed = time.time() - t0
                    self.log.emit(f'  {well_id} done ({elapsed:.1f}s)')
                except Exception as e:
                    self.log.emit(f'  {well_id} ERROR: {e}')

            self.well_progress.emit(total_wells, total_wells)

            # ── Stage 2: Whole-image features ──
            if s.get('wholeImageFeats'):
                self.log.emit(f'\n  --- Stage 2: Whole-image features ---')
                for well_idx, well_id in enumerate(index):
                    if self._stop.is_set():
                        return
                    self.progress.emit(plate_name, plate_idx, total_plates, well_id, 'Whole-image')
                    self.well_progress.emit(well_idx, len(index))
                    row = index[well_id]
                    try:
                        self.log.emit(f'  {well_id}: whole-image features...')
                        from multiWellAnalysis.wholeImage.runWholeImage import processWellWholeImage
                        processWellWholeImage(
                            plate_name, well_id,
                            row['registered_raw'], row['processed'], outdir
                        )
                        row['whole_image_feats'] = os.path.join(outdir, f'{well_id}_wholeImageFeatures.csv')
                    except Exception as e:
                        self.log.emit(f'  {well_id} whole-image ERROR: {e}')
                self.well_progress.emit(len(index), len(index))

            # ── Stage 3: Colony tracking ──
            if s.get('colonyTracking') or s.get('colonyFeats'):
                self.log.emit(f'\n  --- Stage 3: Colony tracking ---')
                for well_idx, well_id in enumerate(index):
                    if self._stop.is_set():
                        return
                    self.progress.emit(plate_name, plate_idx, total_plates, well_id, 'Tracking')
                    self.well_progress.emit(well_idx, len(index))
                    row = index[well_id]
                    try:
                        self.log.emit(f'  {well_id}: colony tracking...')
                        labels_path = self._run_tracking(plate_name, row)
                        if labels_path:
                            row['tracked_labels'] = labels_path
                    except Exception as e:
                        self.log.emit(f'  {well_id} tracking ERROR: {e}')
                self.well_progress.emit(len(index), len(index))

            # ── Stage 4: Colony features ──
            if s.get('colonyFeats'):
                self.log.emit(f'\n  --- Stage 4: Colony features ---')
                for well_idx, well_id in enumerate(index):
                    if self._stop.is_set():
                        return
                    self.progress.emit(plate_name, plate_idx, total_plates, well_id, 'Colony feats')
                    self.well_progress.emit(well_idx, len(index))
                    row = index[well_id]
                    if 'tracked_labels' not in row:
                        self.log.emit(f'  {well_id}: skipping (no tracked labels)')
                        continue
                    try:
                        self.log.emit(f'  {well_id}: colony features...')
                        colony_path, agg_path = self._run_colony_feats(plate_name, row)
                        if colony_path:
                            row['colony_feats'] = colony_path
                            row['well_colony_feats'] = agg_path
                    except Exception as e:
                        self.log.emit(f'  {well_id} colony feats ERROR: {e}')
                self.well_progress.emit(len(index), len(index))

            # ── Save index.csv ──
            index_path = os.path.join(outdir, 'index.csv')
            if index:
                all_keys = list(list(index.values())[0].keys())
                # Collect all possible keys across all wells
                for row in index.values():
                    for k in row:
                        if k not in all_keys:
                            all_keys.append(k)
                with open(index_path, 'w', newline='') as f:
                    writer = csv_mod.DictWriter(f, fieldnames=all_keys)
                    writer.writeheader()
                    for row in index.values():
                        writer.writerow(row)
                self.log.emit(f'\n  Index saved: {index_path}')

            self.progress.emit(plate_name, plate_idx + 1, total_plates, '', 'Done')

    def _load_stack(self, well_files):
        """Load image stack as float32, ensure (H, W, T)."""
        if isinstance(well_files, str):
            raw = tifffile.imread(well_files)
            if raw.ndim == 2:
                stack = raw[np.newaxis].astype(np.float32)
            else:
                stack = raw.astype(np.float32)
            del raw
        else:
            first = tifffile.imread(well_files[0])
            h, w = first.shape[:2]
            stack = np.empty((len(well_files), h, w), dtype=np.float32)
            stack[0] = first.astype(np.float32)
            del first
            for fi in range(1, len(well_files)):
                stack[fi] = tifffile.imread(well_files[fi]).astype(np.float32)

        if stack.ndim == 3 and stack.shape[0] < stack.shape[2]:
            stack = np.transpose(stack, (1, 2, 0))
        return stack

    def _get_params(self, s, mag):
        """Get processing params with per-mag overrides applied."""
        block_diam = s['blockDiam']
        fixed_thresh = s['fixedThresh']
        dust_correction = s['dustCorrection']
        mag_params = s.get('magParams', {})
        if mag and mag in mag_params:
            overrides = mag_params[mag]
            block_diam = overrides.get('blockDiam', block_diam)
            fixed_thresh = overrides.get('fixedThresh', fixed_thresh)
            dust_correction = overrides.get('dustCorrection', dust_correction)
        return block_diam, fixed_thresh, dust_correction

    def _run_tracking(self, plate_name, row):
        """Run colony tracking using paths from the index row. Returns labels path or None."""
        raw_path = row['registered_raw']
        mask_path = row['masks']
        well_id = row['well']

        if not os.path.exists(raw_path) or not os.path.exists(mask_path):
            self.log.emit(f'    Skipping tracking: missing raw/mask files')
            return None

        raw_stack = tifffile.imread(raw_path)
        mask_data = np.load(mask_path)
        mask_key = 'masks' if 'masks' in mask_data else list(mask_data.keys())[0]
        mask_stack = mask_data[mask_key]

        n_frames = mask_stack.shape[-1] if mask_stack.ndim == 3 and mask_stack.shape[-1] < mask_stack.shape[0] else mask_stack.shape[0]
        mask_sums = np.array([mask_stack[..., t].sum() for t in range(n_frames)])
        peak_frame = int(np.argmax(mask_sums))
        seed_frame = max(0, peak_frame // 3)

        from multiWellAnalysis.colony.runTrackingMpTraining import trackColoniesAllFrames
        trackColoniesAllFrames(
            raw_stack, mask_stack, seed_frame, peak_frame,
            plate_name, well_id
        )

        # Find the saved tracked labels file
        outdir = os.path.dirname(raw_path)
        labels_paths = glob.glob(os.path.join(outdir, f'{well_id}_trackedLabels_*.npz'))
        return labels_paths[0] if labels_paths else None

    def _run_colony_feats(self, plate_name, row):
        """Run colony feature extraction using paths from the index row. Returns (colony_path, agg_path) or (None, None)."""
        from multiWellAnalysis.colony.runColonyFeatsTrackedMP import extractTrackedColonyFeatures
        from multiWellAnalysis.colony.wellAggMicrons import aggregateWellFeatures

        well_id = row['well']
        labels_path = row['tracked_labels']
        raw_path = row['registered_raw']
        outdir = os.path.dirname(raw_path)

        data = np.load(labels_path)
        raw_stack = tifffile.imread(raw_path)
        labels = data['labels']
        frames = data['frames']
        was_tracked = data.get('wasTracked', np.ones(len(frames), dtype=bool))

        colony_df = extractTrackedColonyFeatures(
            raw_stack, labels, frames,
            plate_name, well_id, was_tracked,
            labels_path, raw_path
        )
        all_frames = list(range(raw_stack.shape[0]))
        agg_df = aggregateWellFeatures(colony_df, all_frames, plate_name, well_id)

        colony_path = os.path.join(outdir, f'{well_id}_colonyFeatures.csv')
        agg_path = os.path.join(outdir, f'{well_id}_wellColonyFeatures.csv')
        colony_df.to_csv(colony_path, index=False)
        agg_df.to_csv(agg_path, index=False)
        return colony_path, agg_path

    def _discover_wells(self, plate_path):
        """Find wells and their image files, filtered by selected magnifications.

        Groups Bright Field TIF files by (well, mag_suffix), then returns
        only wells matching the magnifications selected in the setup tab.
        Returns dict: {(well_id, mag_suffix): [files]}
        """
        from collections import defaultdict

        tif_files = sorted(glob.glob(os.path.join(plate_path, '*.tif')))
        self.log.emit(f'  Scanning {plate_path}: {len(tif_files)} TIFs at root')
        if not tif_files:
            # Look one level down using os.listdir (no stat calls, SMB-safe)
            try:
                for child in os.listdir(plate_path):
                    child_path = os.path.join(plate_path, child)
                    child_tifs = sorted(glob.glob(os.path.join(child_path, '*.tif')))
                    if child_tifs:
                        tif_files = child_tifs
                        self.log.emit(f'  Found {len(tif_files)} TIFs in {child}')
                        break
            except (PermissionError, OSError):
                pass

        bf_files = [f for f in tif_files if 'Bright Field' in f or 'Bright_Field' in f]
        candidates = bf_files if bf_files else tif_files
        self.log.emit(f'  {len(tif_files)} TIFs, {len(bf_files)} BF, {len(candidates)} candidates')

        # Group by (well, mag_suffix)
        groups = defaultdict(list)
        for f in candidates:
            name = os.path.basename(f)
            m = re.match(r'^([A-P]\d+)(_\d+)_', name)
            if m:
                well, mag = m.group(1), m.group(2)
                groups[(well, mag)].append(f)
            else:
                # No mag suffix
                m2 = re.match(r'^([A-P]\d{1,2})[_.]', name)
                if m2:
                    groups[(m2.group(1), '')].append(f)
        self.log.emit(f'  {len(groups)} well+mag groups, mag_setting={mag_setting}')

        # Filter by selected magnifications
        mag_setting = self._state.get('magnification', 'all')
        if mag_setting == 'all':
            selected_mags = None  # process all
        elif isinstance(mag_setting, str):
            selected_mags = {mag_setting}
        else:
            selected_mags = set(mag_setting)

        wells = {}
        for (well, mag), files in sorted(groups.items()):
            if selected_mags is not None and mag not in selected_mags:
                continue
            key = f'{well}{mag}' if mag else well
            wells[key] = sorted(files)

        return wells


class RunTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread = None
        self._worker = None
        self._stop_event = threading.Event()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # progress
        self.plate_label = QLabel('Plate: \u2014')
        layout.addWidget(self.plate_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.well_label = QLabel('Well: \u2014')
        layout.addWidget(self.well_label)

        self.well_progress_bar = QProgressBar()
        self.well_progress_bar.setValue(0)
        layout.addWidget(self.well_progress_bar)

        self.stage_label = QLabel('Stage: \u2014')
        layout.addWidget(self.stage_label)

        # log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, stretch=1)

    def _start(self):
        plates = self.state.get('plates', [])
        if not plates:
            self.log_text.append('ERROR: No plates selected. Go to Setup tab.')
            return

        self.log_text.clear()
        self._stop_event.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        state_dict = self.state.to_dict()

        self._thread = QThread()
        self._worker = ProcessingWorker(state_dict, self._stop_event)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.well_progress.connect(self._on_well_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _stop(self):
        self._stop_event.set()
        self.log_text.append('Stopping after current well...')
        self.stop_btn.setEnabled(False)

    def _on_progress(self, plate, plate_idx, total, well, stage):
        self.plate_label.setText(f'Plate: {plate}  ({plate_idx + 1} / {total})')
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(plate_idx)
        self.well_label.setText(f'Well: {well}' if well else 'Well: \u2014')
        self.stage_label.setText(f'Stage: {stage}')

    def _on_well_progress(self, well_idx, total_wells):
        self.well_progress_bar.setMaximum(total_wells)
        self.well_progress_bar.setValue(well_idx)

    def _on_log(self, msg):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_error(self, msg):
        self.log_text.append(f'ERROR: {msg}')

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append('\nDone.')
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.well_progress_bar.setValue(self.well_progress_bar.maximum())
        self.stage_label.setText('Stage: Complete')

        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
