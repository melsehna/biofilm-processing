"""
GUI version of run.py

Copy of the Run tab pipeline adapted for the GUI:
- Uses runTrackingGUI, runColonyFeatsGUI, runWholeImageGUI
  (no hardcoded server paths, no io_utils logging)
- Biomass array passed through to tracking for seed frame detection
- Whole-image features save directly to processedImages/ (no plate subdirs)
"""

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
    progress = Signal(str, int, int, str, str)   # plate, plateIdx, totalPlates, well, stage
    well_progress = Signal(int, int)              # wellIdx, totalWells
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
        from multiWellAnalysis.processing.analysis_main import timelapse_processing

        plates = self._state['plates']
        total_plates = len(plates)
        output_root = self._state.get('outputDir', '')

        if not output_root:
            self.log.emit('ERROR: No output directory set. Go to Setup tab.')
            return
        if not os.path.isdir(output_root):
            os.makedirs(output_root, exist_ok=True)

        mag_filter = self._state.get('magnification', 'all')

        for plate_idx, plate_path in enumerate(plates):
            if self._stop.is_set():
                self.log.emit('Cancelled by user.')
                return

            plate_name = os.path.basename(plate_path)
            self.log.emit(f'\n{"="*60}')
            self.log.emit(f'Plate {plate_idx+1}/{total_plates}: {plate_name}')
            self.log.emit(f'{"="*60}')

            mag_groups = self._discover_wells(plate_path, mag_filter)
            if not mag_groups:
                self.log.emit(f'  No wells found in {plate_name}, skipping.')
                continue

            # mirror plate structure in output directory
            outdir = os.path.join(output_root, plate_name, 'processedImages')
            os.makedirs(outdir, exist_ok=True)
            outdir_parent = os.path.join(output_root, plate_name)

            for mag_label, wells_dict in mag_groups:
                if mag_label:
                    self.log.emit(f'  Magnification: {mag_label}')

                well_items = list(wells_dict.items())
                total_wells = len(well_items)
                for well_idx, (well_id, well_files) in enumerate(well_items):
                    if self._stop.is_set():
                        self.log.emit('Cancelled by user.')
                        return

                    self.progress.emit(
                        plate_name, plate_idx, total_plates,
                        well_id, 'Processing'
                    )
                    self.well_progress.emit(well_idx, total_wells)
                    t0 = time.time()

                    try:
                        self._process_well(
                            plate_path, plate_name, outdir, outdir_parent,
                            well_id, well_files
                        )
                        elapsed = time.time() - t0
                        self.log.emit(f'  {well_id} done ({elapsed:.1f}s)')
                    except Exception as e:
                        self.log.emit(f'  {well_id} ERROR: {e}')

                self.well_progress.emit(total_wells, total_wells)

            self.progress.emit(plate_name, plate_idx + 1, total_plates, '', 'Done')

    def _load_well_stack(self, well_files):
        """Load well images into (H, W, T) stack."""
        if isinstance(well_files, str):
            stack = tifffile.imread(well_files).astype(np.float64)
            if stack.ndim == 2:
                # single frame -> (H, W, 1)
                stack = stack[:, :, np.newaxis]
                return stack
        else:
            # multiple single-frame TIFs -> stack along axis 0 gives (T, H, W)
            frames = [tifffile.imread(f).astype(np.float64) for f in well_files]
            stack = np.stack(frames)  # (T, H, W)

        # For 3D stacks: determine axis order.
        # Multi-frame TIFFs are typically (T, H, W) where T << H and T << W.
        # analysis_main expects (H, W, T).
        if stack.ndim == 3:
            # If first dim is much smaller than both others, it's (T, H, W)
            if stack.shape[0] < stack.shape[1] and stack.shape[0] < stack.shape[2]:
                stack = np.transpose(stack, (1, 2, 0))
            # If last dim is much smaller than both others, it's already (H, W, T)
            # Otherwise leave as-is (ambiguous case, assume H, W, T)

        return stack

    def _process_well(self, plate_path, plate_name, outdir, outdir_parent, well_id, well_files):
        from multiWellAnalysis.processing.analysis_main import timelapse_processing

        stack = self._load_well_stack(well_files)
        s = self._state

        # Step 1: image processing (always runs — saves all outputs by default)
        # timelapse_processing creates processedImages/ under outdir_parent
        self.log.emit(f'  {well_id}: preprocessing + registration...')
        masks, biomass, od_mean = timelapse_processing(
            images=stack,
            block_diameter=s['blockDiam'],
            ntimepoints=stack.shape[2],
            shift_thresh=s['shiftThresh'],
            fixed_thresh=s['fixedThresh'],
            dust_correction=s['dustCorrection'],
            outdir=outdir_parent,
            filename=well_id,
            image_records=None,
            fftStride=s['fftStride'],
            downsample=s['downsample'],
            skip_overlay=not s.get('saveOverlays', True),
            workers=s.get('workers', 4),
        )

        # Remove outputs the user doesn't want to keep
        self._cleanup_outputs(outdir, well_id, s)

        # Step 2: colony tracking
        if s.get('colonyTracking') or s.get('colonyFeats'):
            self.log.emit(f'  {well_id}: colony tracking...')
            self._run_tracking(plate_name, outdir, well_id, masks, biomass)

        # Step 3: colony features
        if s.get('colonyFeats'):
            self.log.emit(f'  {well_id}: colony feature extraction...')
            self._run_colony_feats(plate_name, outdir, well_id)

        # Step 4: whole-image features
        if s.get('wholeImageFeats'):
            self.log.emit(f'  {well_id}: whole-image features...')
            self._run_whole_image_feats(plate_name, outdir, well_id)

    def _cleanup_outputs(self, outdir, well_id, s):
        """Remove output files the user chose not to save."""
        removals = []
        if not s.get('saveRegistered', True):
            removals.append(os.path.join(outdir, f'{well_id}_registered_raw.tif'))
        if not s.get('saveProcessed', True):
            removals.append(os.path.join(outdir, f'{well_id}_processed.tif'))
        if not s.get('saveMasks', True):
            # keep masks if colony tracking/feats need them, even if user unchecked
            if not (s.get('colonyTracking') or s.get('colonyFeats')):
                removals.append(os.path.join(outdir, f'{well_id}_masks.npz'))
        if not s.get('saveOverlays', True):
            removals.append(os.path.join(outdir, f'{well_id}_overlay.mp4'))
            removals.append(os.path.join(outdir, f'{well_id}_overlay.gif'))

        for path in removals:
            if os.path.exists(path):
                os.remove(path)

    def _run_tracking(self, plate_name, outdir, well_id, masks, biomass):
        from multiWellAnalysis.colony.runTrackingGUI import trackAndSave

        raw_path = os.path.join(outdir, f'{well_id}_registered_raw.tif')
        mask_path = os.path.join(outdir, f'{well_id}_masks.npz')

        if not os.path.exists(raw_path) or not os.path.exists(mask_path):
            self.log.emit(f'    Skipping tracking: missing raw/mask files')
            return

        raw_stack = tifffile.imread(raw_path)
        mask_data = np.load(mask_path)
        mask_key = 'masks' if 'masks' in mask_data else list(mask_data.keys())[0]
        mask_stack = mask_data[mask_key]

        npz_path = trackAndSave(
            raw_stack, mask_stack, outdir,
            plate_name, well_id,
            biomass=biomass,
        )

        if npz_path:
            self.log.emit(f'    Saved tracked labels: {os.path.basename(npz_path)}')
        else:
            self.log.emit(f'    Tracking produced no output')

    def _run_colony_feats(self, plate_name, outdir, well_id):
        from multiWellAnalysis.colony.runColonyFeatsGUI import extractAndSave

        labels_paths = glob.glob(os.path.join(outdir, f'{well_id}_trackedLabels_*.npz'))
        raw_path = os.path.join(outdir, f'{well_id}_registered_raw.tif')
        if not labels_paths or not os.path.exists(raw_path):
            self.log.emit(f'    Skipping colony feats: missing files')
            return

        data = np.load(labels_paths[0])
        raw_stack = tifffile.imread(raw_path)
        labels = data['labels']
        frames = data['frames']
        was_tracked = bool(data['wasTracked']) if 'wasTracked' in data else True

        colony_df, well_df = extractAndSave(
            raw_stack, labels, frames,
            plate_name, well_id, was_tracked,
            labels_paths[0], raw_path,
            outdir=outdir,
        )

        if colony_df is not None:
            self.log.emit(f'    Colony features: {len(colony_df)} rows')
        else:
            self.log.emit(f'    No colonies found for feature extraction')

    def _run_whole_image_feats(self, plate_name, outdir, well_id):
        from multiWellAnalysis.wholeImage.runWholeImageGUI import extractWholeImageFeatures

        proc_path = os.path.join(outdir, f'{well_id}_processed.tif')
        if not os.path.exists(proc_path):
            self.log.emit(f'    Skipping whole-image: missing processed stack')
            return

        status = extractWholeImageFeatures(
            proc_path, plate_name, well_id, outdir
        )
        self.log.emit(f'    Whole-image features: {status}')

    def _discover_wells(self, plate_path, mag_filter='all'):
        """Find wells and their image files, respecting magnification selection.

        Returns list of (mag_label, wells_dict) tuples.
        wells_dict maps well_id -> file_path or [file_paths].
        """
        tif_files = sorted(glob.glob(os.path.join(plate_path, '*.tif')))

        # Try magnification-aware discovery first
        from multiWellAnalysis.processing.batch_runner import discover_mag_groups
        from multiWellAnalysis.processing.analysis_main import frame_index_from_filename

        bf_files = [f for f in tif_files
                    if 'Bright Field' in f or 'Bright_Field' in f]

        mag_groups = discover_mag_groups(plate_path, bf_files) if bf_files else {}

        if mag_groups:
            # Filter to selected magnification
            if mag_filter != 'all':
                mag_groups = {k: v for k, v in mag_groups.items()
                              if k == mag_filter or k == mag_filter.lstrip('_')}

            result = []
            for mag_label, wells_dict in sorted(mag_groups.items()):
                # Sort files within each well by frame index
                sorted_wells = {}
                for well, files in wells_dict.items():
                    well_label = f'{well}_{mag_label}'
                    sorted_wells[well_label] = sorted(files, key=frame_index_from_filename)
                result.append((mag_label, sorted_wells))
            return result

        # Fallback: simple well discovery (no magnification suffixes)
        wells = {}
        for f in tif_files:
            name = os.path.basename(f)
            if re.match(r'^[A-H]\d{1,2}\.tif$', name):
                well_id = os.path.splitext(name)[0]
                wells[well_id] = f
                continue
            m = re.match(r'^([A-H]\d{1,2})_', name)
            if m:
                well_id = m.group(1)
                wells.setdefault(well_id, [])
                if isinstance(wells[well_id], list):
                    wells[well_id].append(f)

        return [('', wells)] if wells else []


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
        self.well_progress_bar.setFormat('Well %v / %m')
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

        root = self.state.get('rootDir', '')
        if root and not os.path.isdir(root):
            self.log_text.append(f'ERROR: Root directory does not exist: {root}')
            return

        output_dir = self.state.get('outputDir', '')
        if not output_dir:
            self.log_text.append('ERROR: No output directory set. Go to Setup tab.')
            return

        # check at least one output or feature is enabled
        has_output = any(self.state.get(k, False) for k in [
            'saveRegistered', 'saveProcessed', 'saveMasks', 'saveOverlays',
            'wholeImageFeats', 'colonyTracking', 'colonyFeats',
        ])
        if not has_output:
            self.log_text.append('ERROR: No outputs or features enabled. Go to Parameters tab.')
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
        self.stage_label.setText('Stage: Complete')

        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
