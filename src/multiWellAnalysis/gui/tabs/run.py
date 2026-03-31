import json
import os
import sys
import io
import time
import glob
import re
import csv as csv_mod
import threading
import traceback


def _fmt_time(seconds):
    """Format a duration in seconds as a human-readable string."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    elif seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f'{h}h{m:02d}m'

import numpy as np
import pandas as pd
import tifffile

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QTextEdit, QMessageBox,
)
from PySide6.QtCore import QObject, QThread, Signal


# ── Run params (for resume / overwrite detection) ──

_PARAM_KEYS = [
    'blockDiam', 'fixedThresh', 'dustCorrection',
    'shiftThresh', 'fftStride', 'downsample',
    'magnification', 'magParams', 'copyRaw',
]

_RUN_PARAMS_FILE = 'run_params.json'


def _extract_run_params(state):
    """Extract the processing-relevant params from the state dict."""
    return {k: state.get(k) for k in _PARAM_KEYS}


def _save_run_params(outdir, params):
    path = os.path.join(outdir, _RUN_PARAMS_FILE)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)


def _load_run_params(outdir):
    path = os.path.join(outdir, _RUN_PARAMS_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _well_already_processed(outdir, well_id):
    """Check if a well has existing output files from a previous run."""
    return os.path.exists(os.path.join(outdir, f'{well_id}_processed.tif'))


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

        # outdir already points to <plate_outdir>/processedImages;
        # timelapse_processing expects the *parent* and creates
        # processedImages/ inside it.
        plate_outdir = os.path.dirname(outdir)
        masks, biomass, od_mean = timelapse_processing(
            images=stack,
            block_diameter=params['blockDiam'],
            ntimepoints=stack.shape[2],
            shift_thresh=params['shiftThresh'],
            fixed_thresh=params['fixedThresh'],
            dust_correction=params['dustCorrection'],
            outdir=plate_outdir,
            filename=well_id,
            image_records=None,
            fftStride=params.get('fftStride', 6),
            downsample=params.get('downsample', 4),
            skip_overlay=not params.get('saveOverlays', True),
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


def _track_one_well(plate_name, row, tracking_params=None):
    """Run colony tracking on a single well using trackAndSave."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    well_id = row['well']
    raw_path = row['registered_raw']
    mask_path = row['masks']
    biomass_path = row.get('biomass', '')
    if tracking_params is None:
        tracking_params = {}

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
            min_colony_area=tracking_params.get('minColonyAreaPx'),
            prop_radius=tracking_params.get('propRadiusPx'),
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

# Directories that contain pipeline outputs, never raw images.
_OUTPUT_DIR_NAMES = {
    'processedimages', 'processed_images', 'processed_images_py',
    'numerical_data', 'numerical_data_py',
    'results_images', 'results_data',
}

# Raw BF frame: WELL_MAG_..._FRAMENUM.tif  (e.g. A1_03_1_1_Bright Field_001.tif)
# Frame number is always _NNN at the end before .tif
_RAW_FRAME_RE = re.compile(r'^[A-P]\d+_\d+_.+_\d{3}\.tif$', re.IGNORECASE)


def _is_output_dir(name):
    return name.lower() in _OUTPUT_DIR_NAMES


def _is_raw_frame(filename):
    """True if *filename* looks like a raw single-frame BF image."""
    return bool(_RAW_FRAME_RE.match(filename))


def _list_raw_tifs(directory):
    """Return sorted, deduplicated list of raw BF frame paths in *directory*.

    Uses os.listdir (single syscall, full listing) instead of os.scandir
    or glob to avoid partial results from macOS SMB directory caching.
    """
    try:
        names = os.listdir(directory)
    except (PermissionError, OSError):
        return []
    seen = set()
    result = []
    for name in sorted(names):
        if name not in seen and _is_raw_frame(name):
            seen.add(name)
            result.append(os.path.join(directory, name))
    return result


def _resolve_tif_dir(root, max_depth=2):
    """Find the directory containing raw TIF images, up to *max_depth* levels below *root*.

    Skips known output directories (processedImages, Numerical_data_py, etc.)
    and ignores processed/registered stacks.  Uses os.listdir to avoid
    incomplete results from macOS SMB directory caching.
    Returns the resolved directory path, or *root* if nothing is found.
    """
    try:
        names = os.listdir(root)
    except (PermissionError, OSError):
        return root

    # Check if root itself has raw TIFs
    if any(_is_raw_frame(n) for n in names):
        return root

    # Walk one level at a time up to max_depth
    dirs_at_level = [root]
    for _ in range(max_depth):
        next_level = []
        for d in dirs_at_level:
            try:
                entries = os.listdir(d)
            except (PermissionError, OSError):
                continue
            for name in entries:
                if name.startswith('.') or _is_output_dir(name):
                    continue
                child = os.path.join(d, name)
                if os.path.isdir(child):
                    next_level.append(child)
        for d in next_level:
            try:
                if any(_is_raw_frame(n) for n in os.listdir(d)):
                    return d
            except (PermissionError, OSError):
                continue
        dirs_at_level = next_level

    return root


def discover_wells(plate_path, mag_setting='all'):
    """Find wells and their BF image files, filtered by selected magnifications.

    Returns (resolved_plate_path, wells_dict).
    resolved_plate_path is the directory that actually contains the TIF files
    (may be plate_path itself or a child directory).
    """
    resolved = _resolve_tif_dir(plate_path, max_depth=2)
    raw_tifs = _list_raw_tifs(resolved)

    bf_files = [f for f in raw_tifs if 'Bright Field' in f or 'Bright_Field' in f]
    candidates = bf_files if bf_files else raw_tifs

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


def _compute_outdir(plate_path, resolved_plate, output_root):
    """Compute the processedImages/ path for a plate.

    Drawer given:  <root>/<drawer>/processedImages/  (sibling of plate)
    Plate given:   <root>/<plate>/processedImages/   (inside plate)
    No output root: same logic but relative to source location.
    """
    is_drawer = (resolved_plate != plate_path)
    plate_name = os.path.basename(resolved_plate) if is_drawer else os.path.basename(plate_path)
    drawer_name = os.path.basename(plate_path) if is_drawer else None

    if output_root:
        if is_drawer:
            return os.path.join(output_root, drawer_name, 'processedImages')
        else:
            return os.path.join(output_root, plate_name, 'processedImages')
    else:
        if is_drawer:
            return os.path.join(plate_path, 'processedImages')
        else:
            return os.path.join(resolved_plate, 'processedImages')


# ── Qt Worker ──

class ProcessingWorker(QObject):
    overall_progress = Signal(int, int, str)  # done, total, description
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, state_dict, stop_event, resume=False):
        super().__init__()
        self._state = state_dict
        self._stop = stop_event
        self._resume = resume
        self._overall_done = 0
        self._total_tasks = 0

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
        output_root = s.get('outputDir', '')
        mag_setting = s.get('magnification', 'all')

        # Pre-scan all plates to compute total task count for the progress bar
        do_whole = s.get('wholeImageFeats', False)
        do_tracking = s.get('colonyTracking', False) or s.get('colonyFeats', False)
        do_colony_feats = s.get('colonyFeats', False)
        n_stages = 1 + int(do_whole) + int(do_tracking) + int(do_colony_feats)

        total_tasks = 0
        for plate_path in plates:
            resolved, pre_wells = discover_wells(plate_path, mag_setting)
            pre_outdir = _compute_outdir(plate_path, resolved, output_root)
            if self._resume:
                n = sum(1 for wid in pre_wells
                        if not _well_already_processed(pre_outdir, wid))
            else:
                n = len(pre_wells)
            total_tasks += n * n_stages

        self._overall_done = 0
        self._total_tasks = max(total_tasks, 1)
        self.overall_progress.emit(0, self._total_tasks, 'Starting…')

        plate_outdirs = []   # processedImages/ paths, one per plate — for master CSV
        drawer_map = {}      # plate_name → drawer_name (or plate_name if no drawer)

        for plate_idx, plate_path in enumerate(plates):
            if self._stop.is_set():
                self.log.emit('Cancelled by user.')
                return

            resolved_plate, wells = discover_wells(plate_path, mag_setting)
            is_drawer = (resolved_plate != plate_path)

            # plate_name = actual plate dir name (used in index CSV / feature files)
            # drawer_name = parent dir name when user selected a drawer
            plate_name = os.path.basename(resolved_plate) if is_drawer else os.path.basename(plate_path)
            drawer_name = os.path.basename(plate_path) if is_drawer else None

            self.log.emit(f'\n{"="*60}')
            if drawer_name:
                self.log.emit(f'Plate {plate_idx+1}/{total_plates}: {drawer_name} / {plate_name}')
            else:
                self.log.emit(f'Plate {plate_idx+1}/{total_plates}: {plate_name}')
            self.log.emit(f'{"="*60}')

            self.log.emit(f'  Found {len(wells)} wells (mag={mag_setting})')
            if not wells:
                self.log.emit(f'  No wells found, skipping.')
                continue

            outdir = _compute_outdir(plate_path, resolved_plate, output_root)
            # If drawer, also create the plate subdir in output for structure
            if is_drawer and output_root:
                os.makedirs(os.path.join(
                    output_root, drawer_name, plate_name), exist_ok=True)
            os.makedirs(outdir, exist_ok=True)
            self.log.emit(f'  Output dir: {outdir}')

            plate_outdirs.append(outdir)
            drawer_map[plate_name] = drawer_name if drawer_name else plate_name

            # Save run params so future runs can detect changes
            run_params = _extract_run_params(s)
            _save_run_params(outdir, run_params)

            # Resume: skip wells that already have output from a previous run
            well_items = list(wells.items())
            if self._resume:
                skipped = []
                remaining = []
                for well_id, files in well_items:
                    if _well_already_processed(outdir, well_id):
                        skipped.append(well_id)
                    else:
                        remaining.append((well_id, files))
                if skipped:
                    self.log.emit(f'  Resuming: skipping {len(skipped)} already-processed wells')
                well_items = remaining

            index = {}
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
                    self._submit_tracking, plate_name, s
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

            # ── Per-plate numericalData/ CSVs ──
            try:
                from multiWellAnalysis.processing.master_csv import assemble_plate_numerical_data
                assemble_plate_numerical_data(outdir, log_fn=self.log.emit)
            except Exception as e:
                self.log.emit(f'  [numericalData] ERROR: {e}')

        # ── Assemble master CSVs across all plates ──
        if output_root and plate_outdirs and not self._stop.is_set():
            self.log.emit(f'\n{"="*60}\nAssembling master CSVs…')
            try:
                from multiWellAnalysis.processing.master_csv import assemble_master_csvs
                assemble_master_csvs(
                    plate_outdirs, drawer_map, output_root,
                    log_fn=self.log.emit,
                )
            except Exception as e:
                self.log.emit(f'  [master CSV] ERROR: {e}')

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
                self._overall_done += 1
                desc = (f'{stage_name} · Plate {plate_idx+1}/{total_plates} · {plate_name}'
                        f' · {well_id} ({done_count}/{total})')
                self.overall_progress.emit(self._overall_done, self._total_tasks, desc)

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
            'saveOverlays': state.get('saveOverlays', True),
        }
        mag_params = state.get('magParams', {})
        if mag and mag in mag_params:
            params.update(mag_params[mag])

        return pool.submit(_process_one_well, plate_path, outdir, well_id, well_files, params)

    def _submit_whole_image(self, pool, well_id, row, outdir, plate_name):
        if 'registered_raw' not in row:
            return None
        return pool.submit(_whole_image_one_well, plate_name, {**row, 'well': well_id})

    def _submit_tracking(self, pool, well_id, row, outdir, plate_name, state):
        if 'registered_raw' not in row:
            return None
        m = re.match(r'^[A-P]\d+(_\d+)$', well_id)
        mag = m.group(1) if m else ''

        tracking_params = {
            'minColonyAreaPx': state.get('minColonyAreaPx', 200),
            'propRadiusPx': state.get('propRadiusPx', 25),
        }
        mag_params = state.get('magParams', {})
        if mag and mag in mag_params:
            mp = mag_params[mag]
            if 'minColonyAreaPx' in mp:
                tracking_params['minColonyAreaPx'] = mp['minColonyAreaPx']
            if 'propRadiusPx' in mp:
                tracking_params['propRadiusPx'] = mp['propRadiusPx']

        return pool.submit(_track_one_well, plate_name, {**row, 'well': well_id}, tracking_params)

    def _submit_colony_feats(self, pool, well_id, row, outdir, plate_name):
        if 'tracked_labels' not in row:
            return None
        return pool.submit(_colony_feats_one_well, plate_name, {**row, 'well': well_id})

    def _save_index(self, index, outdir, plate_name, plate_path):
        if not index:
            return
        index_path = os.path.join(outdir, 'index.csv')

        # Merge with any existing index rows (from a prior resumed run)
        existing = {}
        if os.path.exists(index_path):
            try:
                import csv as _csv
                with open(index_path, newline='') as f:
                    for row in _csv.DictReader(f):
                        existing[row['well']] = row
            except Exception:
                pass

        # Build new rows, overwriting existing entries for re-processed wells
        new_rows = {}
        for well_id, row in index.items():
            m = re.match(r'^[A-P]\d+(_\d+)$', well_id)
            mag = m.group(1) if m else ''
            full_row = {'plate': plate_name, 'plate_path': plate_path, 'well': well_id, 'mag': mag}
            full_row.update(row)
            new_rows[well_id] = full_row

        merged = {**existing, **new_rows}

        all_keys = ['plate', 'plate_path', 'well', 'mag']
        extra_keys = set()
        for row in merged.values():
            extra_keys.update(row.keys())
        extra_keys -= set(all_keys)
        all_keys.extend(sorted(extra_keys))

        with open(index_path, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for well_id in sorted(merged):
                writer.writerow(merged[well_id])

        self.log.emit(f'\n  Index saved: {index_path}')


# ── Qt Tab ──

class RunTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread = None
        self._worker = None
        self._stop_event = threading.Event()
        self._run_start_time = None
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

        self.status_label = QLabel('Ready')
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%v / %m  (%p%)')
        layout.addWidget(self.progress_bar)

        self.eta_label = QLabel('')
        self.eta_label.setStyleSheet('color: gray; font-size: 11px;')
        layout.addWidget(self.eta_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, stretch=1)

    def _start(self):
        plates = self.state.get('plates', [])
        if not plates:
            self.log_text.append('ERROR: No plates selected. Go to Setup tab.')
            return

        state_dict = self.state.to_dict()
        output_root = state_dict.get('outputDir', '')
        run_params = _extract_run_params(state_dict)
        resume = False

        # Check each plate for existing output
        plates_with_output = []
        for plate_path in plates:
            resolved = _resolve_tif_dir(plate_path, max_depth=2)
            outdir = _compute_outdir(plate_path, resolved, output_root)
            saved = _load_run_params(outdir)
            has_files = os.path.isdir(outdir) and any(
                f.endswith('.tif') for f in os.listdir(outdir))
            if has_files:
                plates_with_output.append((plate_path, saved))

        if plates_with_output:
            # Check if params match for all plates that have output
            all_match = all(saved == run_params for _, saved in plates_with_output
                           if saved is not None)
            any_saved = any(saved is not None for _, saved in plates_with_output)

            if any_saved and all_match:
                reply = QMessageBox.question(
                    self, 'Resume previous run?',
                    f'{len(plates_with_output)} plate(s) already have processed '
                    f'output with the same parameters.\n\n'
                    f'Resume and skip already-processed wells?',
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                )
                if reply == QMessageBox.Cancel:
                    return
                resume = (reply == QMessageBox.Yes)
            else:
                if any_saved:
                    msg = (f'{len(plates_with_output)} plate(s) already have '
                           f'processed output with DIFFERENT parameters.\n\n'
                           f'Continuing will overwrite existing results.')
                else:
                    msg = (f'{len(plates_with_output)} plate(s) already have '
                           f'processed output.\n\n'
                           f'Continuing will overwrite existing results.')
                reply = QMessageBox.warning(
                    self, 'Overwrite existing output?', msg,
                    QMessageBox.Ok | QMessageBox.Cancel,
                )
                if reply == QMessageBox.Cancel:
                    return

        self.log_text.clear()
        self._stop_event.clear()
        self._run_start_time = time.perf_counter()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.eta_label.setText('')
        self.status_label.setText('Starting…')
        self.progress_bar.setValue(0)

        self._thread = QThread()
        self._worker = ProcessingWorker(state_dict, self._stop_event, resume=resume)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.overall_progress.connect(self._on_overall_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _stop(self):
        self._stop_event.set()
        self.log_text.append('Stopping...')
        self.stop_btn.setEnabled(False)

    def _on_overall_progress(self, done, total, desc):
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(done)
        self.status_label.setText(desc)
        if done > 0 and self._run_start_time is not None:
            elapsed = time.perf_counter() - self._run_start_time
            eta_secs = elapsed / done * (total - done) if done < total else 0
            self.eta_label.setText(
                f'Elapsed: {_fmt_time(elapsed)}  ·  ETA: {_fmt_time(eta_secs)}'
            )

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
        self.status_label.setText('Complete')
        if self._run_start_time is not None:
            elapsed = time.perf_counter() - self._run_start_time
            self.eta_label.setText(f'Total time: {_fmt_time(elapsed)}')

        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
