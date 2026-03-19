import os
import sys
import io
import time
import glob
import re
import csv as csv_mod
import threading
import traceback

import numpy as np
import pandas as pd
import tifffile

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QTextEdit,
)
from PySide6.QtCore import QObject, QThread, Signal


# ── Top-level worker functions (picklable for multiprocessing) ──

def _process_one_well(plate_path, outdir, well_id, well_files, params):
    """Run timelapse processing on a single well. Returns index row dict."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    from multiWellAnalysis.processing.analysis_main import timelapse_processing

    try:
        t0 = time.perf_counter()

        # Load stack
        if isinstance(well_files, str):
            raw = tifffile.imread(well_files)
            stack = raw[np.newaxis].astype(np.float32) if raw.ndim == 2 else raw.astype(np.float32)
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

        masks, biomass, od_mean = timelapse_processing(
            images=stack,
            block_diameter=params['blockDiam'],
            ntimepoints=stack.shape[2],
            shift_thresh=params['shiftThresh'],
            fixed_thresh=params['fixedThresh'],
            dust_correction=params['dustCorrection'],
            outdir=plate_path,
            filename=well_id,
            image_records=None,
            fftStride=params.get('fftStride', 6),
            downsample=params.get('downsample', 4),
            skip_overlay=True,
            workers=1,  # parallelism is at the well level, not frame level
        )
        del stack

        # Save biomass CSV
        biomass_path = os.path.join(outdir, f'{well_id}_biomass.csv')
        pd.DataFrame({'frame': range(len(biomass)), 'biomass': biomass}).to_csv(
            biomass_path, index=False
        )

        elapsed = time.perf_counter() - t0
        return {
            'well': well_id,
            'status': 'done',
            'elapsed': elapsed,
            'registered_raw': os.path.join(outdir, f'{well_id}_registered_raw.tif'),
            'processed': os.path.join(outdir, f'{well_id}_processed.tif'),
            'masks': os.path.join(outdir, f'{well_id}_masks.npz'),
            'biomass': biomass_path,
        }
    except Exception as e:
        return {'well': well_id, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _track_one_well(plate_name, row):
    """Run colony tracking on a single well using trackAndSave."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    well_id = row['well']
    raw_path = row['registered_raw']
    mask_path = row['masks']
    biomass_path = row.get('biomass', '')

    try:
        if not os.path.exists(raw_path) or not os.path.exists(mask_path):
            return {'well': well_id, 'status': 'skipped', 'reason': 'missing files'}

        t0 = time.perf_counter()

        # Load raw stack: tifffile returns (T, H, W), convert to (H, W, T)
        raw_stack = tifffile.imread(raw_path)
        if raw_stack.ndim == 3 and raw_stack.shape[0] < raw_stack.shape[1]:
            raw_stack = np.transpose(raw_stack, (1, 2, 0))

        mask_data = np.load(mask_path)
        mask_key = 'masks' if 'masks' in mask_data else list(mask_data.keys())[0]
        mask_stack = mask_data[mask_key]

        # Load biomass for seed frame detection
        biomass = None
        if biomass_path and os.path.exists(biomass_path):
            bdf = pd.read_csv(biomass_path)
            if 'biomass' in bdf.columns:
                biomass = bdf['biomass'].values

        outdir = os.path.dirname(raw_path)

        from multiWellAnalysis.colony.runTrackingGUI import trackAndSave
        npz_path = trackAndSave(
            raw_stack, mask_stack, outdir,
            plate_name, well_id,
            biomass=biomass,
        )

        elapsed = time.perf_counter() - t0

        if npz_path:
            return {
                'well': well_id,
                'status': 'done',
                'elapsed': elapsed,
                'tracked_labels': npz_path,
            }
        else:
            return {'well': well_id, 'status': 'skipped', 'reason': 'no tracking output'}

    except Exception as e:
        return {'well': well_id, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _whole_image_one_well(plate_name, row):
    """Run whole-image feature extraction on a single well."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    well_id = row['well']
    try:
        from multiWellAnalysis.wholeImage.runWholeImageGUI import extractWholeImageFeatures
        outdir = os.path.dirname(row['processed'])
        t0 = time.perf_counter()
        status = extractWholeImageFeatures(
            row['processed'], plate_name, well_id, outdir
        )
        elapsed = time.perf_counter() - t0
        # Find the actual output file (has version string in name)
        feats_files = glob.glob(os.path.join(outdir, f'{well_id}_wholeImage_*.csv'))
        feats_path = feats_files[0] if feats_files else ''
        return {
            'well': well_id,
            'status': 'done' if feats_path else status,
            'elapsed': elapsed,
            'whole_image_feats': feats_path,
        }
    except Exception as e:
        return {'well': well_id, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _colony_feats_one_well(plate_name, row):
    """Run colony feature extraction on a single well."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    well_id = row['well']
    try:
        from multiWellAnalysis.colony.runColonyFeatsGUI import extractAndSave

        labels_path = row['tracked_labels']
        raw_path = row['registered_raw']
        outdir = os.path.dirname(raw_path)

        data = np.load(labels_path)
        # Load raw: tifffile returns (T, H, W), convert to (H, W, T)
        raw_stack = tifffile.imread(raw_path)
        if raw_stack.ndim == 3 and raw_stack.shape[0] < raw_stack.shape[1]:
            raw_stack = np.transpose(raw_stack, (1, 2, 0))

        labels = data['labels']
        frames = data['frames']
        was_tracked = bool(data['wasTracked']) if 'wasTracked' in data else True

        t0 = time.perf_counter()
        colony_df, well_df = extractAndSave(
            raw_stack, labels, frames,
            plate_name, well_id, was_tracked,
            labels_path, raw_path,
            outdir=outdir,
        )
        elapsed = time.perf_counter() - t0

        # Find actual output files (have version string in name)
        colony_files = glob.glob(os.path.join(outdir, f'{well_id}_colonyFeatures_*.csv'))
        agg_files = glob.glob(os.path.join(outdir, f'{well_id}_wellColonyFeatures_*.csv'))

        return {
            'well': well_id,
            'status': 'done',
            'elapsed': elapsed,
            'colony_feats': colony_files[0] if colony_files else '',
            'well_colony_feats': agg_files[0] if agg_files else '',
        }
    except Exception as e:
        return {'well': well_id, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


# ── Well discovery ──

def discover_wells(plate_path, mag_setting='all'):
    """Find wells and their BF image files, filtered by selected magnifications.

    Returns (resolved_plate_path, wells_dict).
    resolved_plate_path is the directory that actually contains the TIF files
    (may be plate_path itself or a child directory).
    """
    resolved = plate_path
    tif_files = sorted(glob.glob(os.path.join(plate_path, '*.tif')))
    if not tif_files:
        try:
            for child in os.listdir(plate_path):
                child_path = os.path.join(plate_path, child)
                child_tifs = sorted(glob.glob(os.path.join(child_path, '*.tif')))
                if child_tifs:
                    tif_files = child_tifs
                    resolved = child_path
                    break
        except (PermissionError, OSError):
            pass

    bf_files = [f for f in tif_files if 'Bright Field' in f or 'Bright_Field' in f]
    candidates = bf_files if bf_files else tif_files

    groups = defaultdict(list)
    for f in candidates:
        name = os.path.basename(f)
        m = re.match(r'^([A-P]\d+)(_\d+)_', name)
        if m:
            groups[(m.group(1), m.group(2))].append(f)
        else:
            m2 = re.match(r'^([A-P]\d{1,2})[_.]', name)
            if m2:
                groups[(m2.group(1), '')].append(f)

    if mag_setting == 'all':
        selected_mags = None
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

    return resolved, wells


# ── Qt Worker ──

class ProcessingWorker(QObject):
    progress = Signal(str, int, int, str, str)
    well_progress = Signal(int, int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, state_dict, stop_event):
        super().__init__()
        self._state = state_dict
        self._stop = stop_event

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            self.error.emit(f'{e}\n{traceback.format_exc()}')
        finally:
            self.finished.emit()

    def _run_pipeline(self):
        s = self._state
        plates = s['plates']
        total_plates = len(plates)
        n_workers = s.get('workers', 4)

        for plate_idx, plate_path in enumerate(plates):
            if self._stop.is_set():
                self.log.emit('Cancelled by user.')
                return

            plate_name = os.path.basename(plate_path)
            self.log.emit(f'\n{"="*60}')
            self.log.emit(f'Plate {plate_idx+1}/{total_plates}: {plate_name}')
            self.log.emit(f'{"="*60}')

            mag_setting = s.get('magnification', 'all')
            resolved_plate, wells = discover_wells(plate_path, mag_setting)
            if resolved_plate != plate_path:
                self.log.emit(f'  Resolved plate dir: {os.path.basename(resolved_plate)}')
            self.log.emit(f'  Found {len(wells)} wells (mag={mag_setting})')
            if not wells:
                self.log.emit(f'  No wells found, skipping.')
                continue

            outdir = os.path.join(resolved_plate, 'processedImages')
            os.makedirs(outdir, exist_ok=True)

            index = {}
            well_items = list(wells.items())
            total_wells = len(well_items)

            # ── Stage 1: Processing (parallel) ──
            self.log.emit(f'\n  --- Stage 1: Processing ({total_wells} wells, {n_workers} workers) ---')
            self._run_stage_parallel(
                plate_name, plate_idx, total_plates, 'Processing',
                well_items, index, outdir, n_workers,
                self._submit_processing, resolved_plate, s
            )

            # ── Stage 2: Whole-image features (parallel) ──
            if s.get('wholeImageFeats') and index:
                self.log.emit(f'\n  --- Stage 2: Whole-image features ({len(index)} wells) ---')
                self._run_stage_parallel(
                    plate_name, plate_idx, total_plates, 'Whole-image',
                    list(index.items()), index, outdir, n_workers,
                    self._submit_whole_image, plate_name
                )

            # ── Stage 3: Colony tracking (parallel) ──
            if (s.get('colonyTracking') or s.get('colonyFeats')) and index:
                self.log.emit(f'\n  --- Stage 3: Colony tracking ({len(index)} wells) ---')
                self._run_stage_parallel(
                    plate_name, plate_idx, total_plates, 'Tracking',
                    list(index.items()), index, outdir, n_workers,
                    self._submit_tracking, plate_name
                )

            # ── Stage 4: Colony features (parallel) ──
            if s.get('colonyFeats') and index:
                trackable = [(k, v) for k, v in index.items() if 'tracked_labels' in v]
                if trackable:
                    self.log.emit(f'\n  --- Stage 4: Colony features ({len(trackable)} wells) ---')
                    self._run_stage_parallel(
                        plate_name, plate_idx, total_plates, 'Colony feats',
                        trackable, index, outdir, n_workers,
                        self._submit_colony_feats, plate_name
                    )

            # ── Save index.csv ──
            self._save_index(index, outdir, plate_name, resolved_plate)
            self.progress.emit(plate_name, plate_idx + 1, total_plates, '', 'Done')

    def _run_stage_parallel(self, plate_name, plate_idx, total_plates, stage_name,
                            items, index, outdir, n_workers, submit_fn, *submit_args):
        """Run a stage in parallel using ProcessPoolExecutor."""
        total = len(items)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for well_id, data in items:
                if self._stop.is_set():
                    self.log.emit('Cancelled by user.')
                    return
                fut = submit_fn(pool, well_id, data, outdir, *submit_args)
                if fut is not None:
                    futures[fut] = well_id

            done_count = 0
            for fut in as_completed(futures):
                well_id = futures[fut]
                done_count += 1
                self.progress.emit(plate_name, plate_idx, total_plates, well_id, stage_name)
                self.well_progress.emit(done_count, total)

                try:
                    result = fut.result()
                except Exception as e:
                    self.log.emit(f'  {well_id} {stage_name} EXCEPTION: {e}')
                    continue

                if result['status'] == 'done':
                    elapsed = result.get('elapsed', 0)
                    self.log.emit(f'  {well_id} done ({elapsed:.1f}s)')
                    # Merge result into index
                    if well_id not in index:
                        index[well_id] = {}
                    for k, v in result.items():
                        if k not in ('well', 'status', 'elapsed'):
                            index[well_id][k] = v
                elif result['status'] == 'error':
                    self.log.emit(f'  {well_id} ERROR: {result.get("error", "unknown")}')
                else:
                    self.log.emit(f'  {well_id} {result["status"]}: {result.get("reason", "")}')

        self.well_progress.emit(total, total)

    # ── Submit helpers (create futures) ──

    def _submit_processing(self, pool, well_id, well_files, outdir, plate_path, state):
        m = re.match(r'^[A-P]\d+(_\d+)$', well_id)
        mag = m.group(1) if m else ''

        params = {
            'blockDiam': state['blockDiam'],
            'fixedThresh': state['fixedThresh'],
            'dustCorrection': state['dustCorrection'],
            'shiftThresh': state['shiftThresh'],
            'fftStride': state.get('fftStride', 6),
            'downsample': state.get('downsample', 4),
        }
        mag_params = state.get('magParams', {})
        if mag and mag in mag_params:
            params.update(mag_params[mag])

        return pool.submit(_process_one_well, plate_path, outdir, well_id, well_files, params)

    def _submit_whole_image(self, pool, well_id, row, outdir, plate_name):
        if 'registered_raw' not in row:
            return None
        return pool.submit(_whole_image_one_well, plate_name, {**row, 'well': well_id})

    def _submit_tracking(self, pool, well_id, row, outdir, plate_name):
        if 'registered_raw' not in row:
            return None
        return pool.submit(_track_one_well, plate_name, {**row, 'well': well_id})

    def _submit_colony_feats(self, pool, well_id, row, outdir, plate_name):
        if 'tracked_labels' not in row:
            return None
        return pool.submit(_colony_feats_one_well, plate_name, {**row, 'well': well_id})

    def _save_index(self, index, outdir, plate_name, plate_path):
        if not index:
            return
        index_path = os.path.join(outdir, 'index.csv')
        all_keys = ['plate', 'plate_path', 'well', 'mag']
        extra_keys = set()
        for row in index.values():
            extra_keys.update(row.keys())
        extra_keys -= set(all_keys)
        all_keys.extend(sorted(extra_keys))

        with open(index_path, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for well_id, row in sorted(index.items()):
                m = re.match(r'^[A-P]\d+(_\d+)$', well_id)
                mag = m.group(1) if m else ''
                full_row = {'plate': plate_name, 'plate_path': plate_path, 'well': well_id, 'mag': mag}
                full_row.update(row)
                writer.writerow(full_row)

        self.log.emit(f'\n  Index saved: {index_path}')


# ── Qt Tab ──

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
        self.log_text.append('Stopping...')
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
