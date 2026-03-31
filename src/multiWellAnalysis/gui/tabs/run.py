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


def _fmtTime(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    elif seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f'{h}h{m:02d}m'


_paramKeys = [
    'blockDiam', 'fixedThresh', 'dustCorrection',
    'shiftThresh', 'fftStride', 'downsample',
    'magnification', 'magParams', 'copyRaw',
]

_runParamsFile = 'run_params.json'


def _extractRunParams(state):
    return {k: state.get(k) for k in _paramKeys}


def _saveRunParams(outdir, params):
    path = os.path.join(outdir, _runParamsFile)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)


def _loadRunParams(outdir):
    path = os.path.join(outdir, _runParamsFile)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _wellAlreadyProcessed(outdir, wellId):
    return os.path.exists(os.path.join(outdir, f'{wellId}_processed.tif'))


def _processOneWell(platePath, outdir, wellId, wellFiles, params):
    """Run timelapse processing on a single well. Returns index row dict."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    from multiWellAnalysis.processing.analysis_main import timelapseProcessing

    try:
        t0 = time.perf_counter()

        if isinstance(wellFiles, str):
            raw = tifffile.imread(wellFiles)
            stack = raw[np.newaxis].astype(np.float32) if raw.ndim == 2 else raw.astype(np.float32)
            del raw
        else:
            first = tifffile.imread(wellFiles[0])
            h, w = first.shape[:2]
            stack = np.empty((len(wellFiles), h, w), dtype=np.float32)
            stack[0] = first.astype(np.float32)
            del first
            for fi in range(1, len(wellFiles)):
                stack[fi] = tifffile.imread(wellFiles[fi]).astype(np.float32)

        if stack.ndim == 3 and stack.shape[0] < stack.shape[2]:
            stack = np.transpose(stack, (1, 2, 0))

        plateOutdir = os.path.dirname(outdir)
        masks, biomass, odMean = timelapseProcessing(
            images=stack,
            blockDiameter=params['blockDiam'],
            ntimepoints=stack.shape[2],
            shiftThresh=params['shiftThresh'],
            fixedThresh=params['fixedThresh'],
            dustCorrection=params['dustCorrection'],
            outdir=plateOutdir,
            filename=wellId,
            imageRecords=None,
            fftStride=params.get('fftStride', 6),
            downsample=params.get('downsample', 4),
            skipOverlay=not params.get('saveOverlays', True),
            workers=1,
        )
        del stack

        biomassPath = os.path.join(outdir, f'{wellId}_biomass.csv')
        pd.DataFrame({'frame': range(len(biomass)), 'biomass': biomass}).to_csv(
            biomassPath, index=False
        )

        elapsed = time.perf_counter() - t0
        return {
            'well': wellId,
            'status': 'done',
            'elapsed': elapsed,
            'registered_raw': os.path.join(outdir, f'{wellId}_registered_raw.tif'),
            'processed': os.path.join(outdir, f'{wellId}_processed.tif'),
            'masks': os.path.join(outdir, f'{wellId}_masks.npz'),
            'biomass': biomassPath,
        }
    except Exception as e:
        return {'well': wellId, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _trackOneWell(plateName, row, trackingParams=None):
    """Run colony tracking on a single well using trackAndSave."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    wellId = row['well']
    rawPath = row['registered_raw']
    maskPath = row['masks']
    biomassPath = row.get('biomass', '')
    if trackingParams is None:
        trackingParams = {}

    try:
        if not os.path.exists(rawPath) or not os.path.exists(maskPath):
            return {'well': wellId, 'status': 'skipped', 'reason': 'missing files'}

        t0 = time.perf_counter()

        rawStack = tifffile.imread(rawPath)
        if rawStack.ndim == 3 and rawStack.shape[0] < rawStack.shape[1]:
            rawStack = np.transpose(rawStack, (1, 2, 0))

        maskData = np.load(maskPath)
        maskKey = 'masks' if 'masks' in maskData else list(maskData.keys())[0]
        maskStack = maskData[maskKey]

        biomass = None
        if biomassPath and os.path.exists(biomassPath):
            bdf = pd.read_csv(biomassPath)
            if 'biomass' in bdf.columns:
                biomass = bdf['biomass'].values

        outdir = os.path.dirname(rawPath)

        from multiWellAnalysis.colony.runTrackingGUI import trackAndSave
        npzPath = trackAndSave(
            rawStack, maskStack, outdir,
            plateName, wellId,
            biomass=biomass,
            minColonyArea=trackingParams.get('minColonyAreaPx'),
            propRadius=trackingParams.get('propRadiusPx'),
        )

        elapsed = time.perf_counter() - t0

        if npzPath:
            return {
                'well': wellId,
                'status': 'done',
                'elapsed': elapsed,
                'tracked_labels': npzPath,
            }
        else:
            return {'well': wellId, 'status': 'skipped', 'reason': 'no tracking output'}

    except Exception as e:
        return {'well': wellId, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _wholeImageOneWell(plateName, row):
    """Run whole-image feature extraction on a single well."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    wellId = row['well']
    try:
        from multiWellAnalysis.wholeImage.runWholeImageGUI import extractWholeImageFeatures
        outdir = os.path.dirname(row['processed'])
        t0 = time.perf_counter()
        status = extractWholeImageFeatures(
            row['processed'], plateName, wellId, outdir
        )
        elapsed = time.perf_counter() - t0
        featsFiles = glob.glob(os.path.join(outdir, f'{wellId}_wholeImage_*.csv'))
        featsPath = featsFiles[0] if featsFiles else ''
        return {
            'well': wellId,
            'status': 'done' if featsPath else status,
            'elapsed': elapsed,
            'whole_image_feats': featsPath,
        }
    except Exception as e:
        return {'well': wellId, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _colonyFeatsOneWell(plateName, row):
    """Run colony feature extraction on a single well."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    wellId = row['well']
    try:
        from multiWellAnalysis.colony.runColonyFeatsGUI import extractAndSave

        labelsPath = row['tracked_labels']
        rawPath = row['registered_raw']
        outdir = os.path.dirname(rawPath)

        data = np.load(labelsPath)
        rawStack = tifffile.imread(rawPath)
        if rawStack.ndim == 3 and rawStack.shape[0] < rawStack.shape[1]:
            rawStack = np.transpose(rawStack, (1, 2, 0))

        labels = data['labels']
        frames = data['frames']
        wasTracked = bool(data['wasTracked']) if 'wasTracked' in data else True

        t0 = time.perf_counter()
        colonyDf, wellDf = extractAndSave(
            rawStack, labels, frames,
            plateName, wellId, wasTracked,
            labelsPath, rawPath,
            outdir=outdir,
        )
        elapsed = time.perf_counter() - t0

        colonyFiles = glob.glob(os.path.join(outdir, f'{wellId}_colonyFeatures_*.csv'))
        aggFiles = glob.glob(os.path.join(outdir, f'{wellId}_wellColonyFeatures_*.csv'))

        return {
            'well': wellId,
            'status': 'done',
            'elapsed': elapsed,
            'colony_feats': colonyFiles[0] if colonyFiles else '',
            'well_colony_feats': aggFiles[0] if aggFiles else '',
        }
    except Exception as e:
        return {'well': wellId, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


_outputDirNames = {
    'processedimages', 'processed_images', 'processed_images_py',
    'numerical_data', 'numerical_data_py',
    'results_images', 'results_data',
}

_rawFrameRe = re.compile(r'^[A-P]\d+_\d+_.+_\d{3}\.tif$', re.IGNORECASE)


def _isOutputDir(name):
    return name.lower() in _outputDirNames


def _isRawFrame(filename):
    return bool(_rawFrameRe.match(filename))


def _listRawTifs(directory):
    """Return sorted, deduplicated list of raw BF frame paths in directory."""
    try:
        names = os.listdir(directory)
    except (PermissionError, OSError):
        return []
    seen = set()
    result = []
    for name in sorted(names):
        if name not in seen and _isRawFrame(name):
            seen.add(name)
            result.append(os.path.join(directory, name))
    return result


def _resolveTifDir(root, maxDepth=2):
    """Find the first directory containing raw TIF images, up to maxDepth levels below root."""
    try:
        names = os.listdir(root)
    except (PermissionError, OSError):
        return root

    if any(_isRawFrame(n) for n in names):
        return root

    dirsAtLevel = [root]
    for _ in range(maxDepth):
        nextLevel = []
        for d in dirsAtLevel:
            try:
                entries = os.listdir(d)
            except (PermissionError, OSError):
                continue
            for name in entries:
                if name.startswith('.') or _isOutputDir(name):
                    continue
                child = os.path.join(d, name)
                if os.path.isdir(child):
                    nextLevel.append(child)
        for d in nextLevel:
            try:
                if any(_isRawFrame(n) for n in os.listdir(d)):
                    return d
            except (PermissionError, OSError):
                continue
        dirsAtLevel = nextLevel

    return root


def _resolveAllTifDirs(root, maxDepth=2):
    """Find ALL directories containing raw TIF images under root.

    Unlike _resolveTifDir which returns only the first match, this returns
    every plate directory found — needed when root is a drawer containing
    multiple plates.  Returns [(platePath, resolvedDir), ...].
    """
    try:
        names = os.listdir(root)
    except (PermissionError, OSError):
        return [(root, root)]

    if any(_isRawFrame(n) for n in names):
        return [(root, root)]

    found = []
    dirsAtLevel = [root]
    for _ in range(maxDepth):
        nextLevel = []
        for d in dirsAtLevel:
            try:
                entries = os.listdir(d)
            except (PermissionError, OSError):
                continue
            for name in entries:
                if name.startswith('.') or _isOutputDir(name):
                    continue
                child = os.path.join(d, name)
                if os.path.isdir(child):
                    nextLevel.append(child)
        for d in sorted(nextLevel):
            try:
                if any(_isRawFrame(n) for n in os.listdir(d)):
                    found.append((root, d))
            except (PermissionError, OSError):
                continue
        dirsAtLevel = nextLevel

    return found if found else [(root, root)]


def discoverWells(platePath, magSetting='all'):
    """Find wells and their BF image files, filtered by selected magnifications.

    platePath should be the directory containing TIF files (already resolved).
    Returns (resolvedPlatePath, wellsDict).
    """
    rawTifs = _listRawTifs(platePath)
    if rawTifs:
        resolved = platePath
    else:
        resolved = _resolveTifDir(platePath, maxDepth=2)
        rawTifs = _listRawTifs(resolved)

    bfFiles = [f for f in rawTifs if 'Bright Field' in f or 'Bright_Field' in f]
    candidates = bfFiles if bfFiles else rawTifs

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

    if magSetting == 'all':
        selectedMags = None
    elif isinstance(magSetting, str):
        selectedMags = {magSetting}
    else:
        selectedMags = set(magSetting)

    wells = {}
    for (well, mag), files in sorted(groups.items()):
        if selectedMags is not None and mag not in selectedMags:
            continue
        key = f'{well}{mag}' if mag else well
        wells[key] = sorted(files)

    return resolved, wells


def _computeOutdir(userPath, resolvedPlate, outputRoot):
    """Compute the processedImages/ path for a plate.

    Drawer given:  output/<drawer>/<plate>/processedImages/
    Plate given:   output/<plate>/processedImages/
    No output root: <resolvedPlate>/processedImages/
    """
    isDrawer = (resolvedPlate != userPath)
    plateName = os.path.basename(resolvedPlate)
    drawerName = os.path.basename(userPath) if isDrawer else None

    if outputRoot:
        if isDrawer:
            return os.path.join(outputRoot, drawerName, plateName, 'processedImages')
        else:
            return os.path.join(outputRoot, plateName, 'processedImages')
    else:
        return os.path.join(resolvedPlate, 'processedImages')


class ProcessingWorker(QObject):
    overallProgress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)
    def __init__(self, stateDict, stopEvent):
        super().__init__()
        self._state = stateDict
        self._stop = stopEvent
        self._overallDone = 0
        self._totalTasks = 1

    def run(self):
        try:
            self._runPipeline()
        except Exception as e:
            self.error.emit(f'{e}\n{traceback.format_exc()}')
        finally:
            self.finished.emit()

    def _runPipeline(self):
        s = self._state
        nWorkers = s.get('workers', 4)
        outputRoot = s.get('outputDir', '')
        magSetting = s.get('magnification', 'all')

        doWhole = s.get('wholeImageFeats', False)
        doTracking = s.get('colonyTracking', False) or s.get('colonyFeats', False)
        doColonyFeats = s.get('colonyFeats', False)
        nStages = 1 + int(doWhole) + int(doTracking) + int(doColonyFeats)

        enabled = ['biomass']
        if doWhole: enabled.append('whole-image')
        if doTracking: enabled.append('tracking')
        if doColonyFeats: enabled.append('colony-feats')
        self.log.emit(f'Enabled stages: {", ".join(enabled)} ({nStages} total)')

        self._overallDone = 0
        self._totalTasks = 1
        self.overallProgress.emit(0, 1, 'Starting…')

        plateOutdirs = []
        drawerMap = {}
        runParams = _extractRunParams(s)
        plateIdx = 0

        for platePath in s['plates']:
            expanded = _resolveAllTifDirs(platePath, maxDepth=2)

            for userPath, resolvedPlate in expanded:
                if self._stop.is_set():
                    self.log.emit('Cancelled by user.')
                    return

                _, wells = discoverWells(resolvedPlate, magSetting)
                isDrawer = (resolvedPlate != userPath)
                plateName = os.path.basename(resolvedPlate)
                drawerName = os.path.basename(userPath) if isDrawer else None

                self.log.emit(f'\n{"="*60}')
                if drawerName:
                    self.log.emit(f'Plate {plateIdx+1}: {drawerName} / {plateName}')
                else:
                    self.log.emit(f'Plate {plateIdx+1}: {plateName}')
                self.log.emit(f'{"="*60}')

                self.log.emit(f'  Found {len(wells)} wells (mag={magSetting})')
                if not wells:
                    self.log.emit(f'  No wells found, skipping.')
                    plateIdx += 1
                    continue

                outdir = _computeOutdir(userPath, resolvedPlate, outputRoot)
                os.makedirs(outdir, exist_ok=True)
                self.log.emit(f'  Output dir: {outdir}')

                plateOutdirs.append(outdir)
                drawerMap[plateName] = drawerName if drawerName else plateName

                # per-plate resume: check if output already exists with same params
                saved = _loadRunParams(outdir)
                resume = False
                if saved is not None and saved == runParams:
                    resume = True
                _saveRunParams(outdir, runParams)

                wellItems = list(wells.items())
                index = {}
                if resume:
                    # load previously-done wells into index so later stages can run on them
                    existingIndex = os.path.join(outdir, 'index.csv')
                    if os.path.exists(existingIndex):
                        try:
                            import csv as _csv
                            with open(existingIndex, newline='') as f:
                                for row in _csv.DictReader(f):
                                    wid = row.get('well', '')
                                    if wid:
                                        index[wid] = {k: v for k, v in row.items()
                                                      if k not in ('plate', 'plate_path', 'well', 'mag')}
                        except Exception:
                            pass

                    skipped = []
                    remaining = []
                    for wellId, files in wellItems:
                        if _wellAlreadyProcessed(outdir, wellId):
                            skipped.append(wellId)
                        else:
                            remaining.append((wellId, files))
                    if skipped:
                        self.log.emit(f'  Resuming: skipping {len(skipped)} already-processed wells')
                    wellItems = remaining

                # update total tasks incrementally as we discover wells
                self._totalTasks += len(wellItems) * nStages
                self.overallProgress.emit(self._overallDone, self._totalTasks, f'Processing {plateName}…')

                totalWells = len(wellItems)

                self.log.emit(f'\n  --- Stage 1: Processing ({totalWells} wells, {nWorkers} workers) ---')
                self._runStageParallel(
                    plateName, plateIdx, 0, 'Processing',
                    wellItems, index, outdir, nWorkers,
                    self._submitProcessing, resolvedPlate, s
                )

                if s.get('wholeImageFeats') and index:
                    self.log.emit(f'\n  --- Stage 2: Whole-image features ({len(index)} wells) ---')
                    self._runStageParallel(
                        plateName, plateIdx, 0, 'Whole-image',
                        list(index.items()), index, outdir, nWorkers,
                        self._submitWholeImage, plateName
                    )

                if (s.get('colonyTracking') or s.get('colonyFeats')) and index:
                    self.log.emit(f'\n  --- Stage 3: Colony tracking ({len(index)} wells) ---')
                    self._runStageParallel(
                        plateName, plateIdx, 0, 'Tracking',
                        list(index.items()), index, outdir, nWorkers,
                        self._submitTracking, plateName, s
                    )

                if s.get('colonyFeats') and index:
                    trackable = [(k, v) for k, v in index.items() if 'tracked_labels' in v]
                    if trackable:
                        self.log.emit(f'\n  --- Stage 4: Colony features ({len(trackable)} wells) ---')
                        self._runStageParallel(
                            plateName, plateIdx, 0, 'Colony feats',
                            trackable, index, outdir, nWorkers,
                            self._submitColonyFeats, plateName
                        )

                # log index summary before saving
                indexCols = set()
                for row in index.values():
                    indexCols.update(row.keys())
                self.log.emit(f'\n  Index: {len(index)} wells, columns: {sorted(indexCols)}')

                self._saveIndex(index, outdir, plateName, resolvedPlate)

                try:
                    from multiWellAnalysis.processing.master_csv import assemblePlateNumericalData
                    assemblePlateNumericalData(outdir, logFn=self.log.emit)
                except Exception as e:
                    self.log.emit(f'  [numericalData] ERROR: {e}')

                plateIdx += 1

        if outputRoot and plateOutdirs and not self._stop.is_set():
            self.log.emit(f'\n{"="*60}\nAssembling master CSVs…')
            try:
                from multiWellAnalysis.processing.master_csv import assembleMasterCsvs
                assembleMasterCsvs(
                    plateOutdirs, drawerMap, outputRoot,
                    logFn=self.log.emit,
                )
            except Exception as e:
                self.log.emit(f'  [master CSV] ERROR: {e}')

    def _runStageParallel(self, plateName, plateIdx, totalPlates, stageName,
                          items, index, outdir, nWorkers, submitFn, *submitArgs):
        total = len(items)

        pool = ProcessPoolExecutor(max_workers=nWorkers)
        try:
            futures = {}
            for wellId, data in items:
                if self._stop.is_set():
                    break
                fut = submitFn(pool, wellId, data, outdir, *submitArgs)
                if fut is not None:
                    futures[fut] = wellId

            doneCount = 0
            for fut in as_completed(futures):
                if self._stop.is_set():
                    # cancel all pending futures and kill workers immediately
                    for f in futures:
                        f.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    self.log.emit('Stopped — cancelled remaining wells.')
                    return

                wellId = futures[fut]
                doneCount += 1
                self._overallDone += 1
                desc = (f'{stageName} · {plateName}'
                        f' · {wellId} ({doneCount}/{total})')
                self.overallProgress.emit(self._overallDone, self._totalTasks, desc)

                try:
                    result = fut.result()
                except Exception as e:
                    self.log.emit(f'  {wellId} {stageName} EXCEPTION: {e}')
                    continue

                if result['status'] == 'done':
                    elapsed = result.get('elapsed', 0)
                    self.log.emit(f'  {wellId} done ({elapsed:.1f}s)')
                    if wellId not in index:
                        index[wellId] = {}
                    for k, v in result.items():
                        if k not in ('well', 'status', 'elapsed'):
                            index[wellId][k] = v
                elif result['status'] == 'error':
                    self.log.emit(f'  {wellId} ERROR: {result.get("error", "unknown")}')
                else:
                    self.log.emit(f'  {wellId} {result["status"]}: {result.get("reason", "")}')
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _submitProcessing(self, pool, wellId, wellFiles, outdir, platePath, state):
        m = re.match(r'^[A-P]\d+(_\d+)$', wellId)
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
        magParams = state.get('magParams', {})
        if mag and mag in magParams:
            params.update(magParams[mag])

        return pool.submit(_processOneWell, platePath, outdir, wellId, wellFiles, params)

    def _submitWholeImage(self, pool, wellId, row, outdir, plateName):
        if 'registered_raw' not in row:
            return None
        return pool.submit(_wholeImageOneWell, plateName, {**row, 'well': wellId})

    def _submitTracking(self, pool, wellId, row, outdir, plateName, state):
        if 'registered_raw' not in row:
            return None
        m = re.match(r'^[A-P]\d+(_\d+)$', wellId)
        mag = m.group(1) if m else ''

        trackingParams = {
            'minColonyAreaPx': state.get('minColonyAreaPx', 200),
            'propRadiusPx': state.get('propRadiusPx', 25),
        }
        magParams = state.get('magParams', {})
        if mag and mag in magParams:
            mp = magParams[mag]
            if 'minColonyAreaPx' in mp:
                trackingParams['minColonyAreaPx'] = mp['minColonyAreaPx']
            if 'propRadiusPx' in mp:
                trackingParams['propRadiusPx'] = mp['propRadiusPx']

        return pool.submit(_trackOneWell, plateName, {**row, 'well': wellId}, trackingParams)

    def _submitColonyFeats(self, pool, wellId, row, outdir, plateName):
        if 'tracked_labels' not in row:
            return None
        return pool.submit(_colonyFeatsOneWell, plateName, {**row, 'well': wellId})

    def _saveIndex(self, index, outdir, plateName, platePath):
        if not index:
            return
        indexPath = os.path.join(outdir, 'index.csv')

        existing = {}
        if os.path.exists(indexPath):
            try:
                import csv as _csv
                with open(indexPath, newline='') as f:
                    for row in _csv.DictReader(f):
                        existing[row['well']] = row
            except Exception:
                pass

        newRows = {}
        for wellId, row in index.items():
            m = re.match(r'^[A-P]\d+(_\d+)$', wellId)
            mag = m.group(1) if m else ''
            fullRow = {'plate': plateName, 'plate_path': platePath, 'well': wellId, 'mag': mag}
            fullRow.update(row)
            newRows[wellId] = fullRow

        merged = {**existing, **newRows}

        allKeys = ['plate', 'plate_path', 'well', 'mag']
        extraKeys = set()
        for row in merged.values():
            extraKeys.update(row.keys())
        extraKeys -= set(allKeys)
        allKeys.extend(sorted(extraKeys))

        with open(indexPath, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=allKeys, extrasaction='ignore')
            writer.writeheader()
            for wellId in sorted(merged):
                writer.writerow(merged[wellId])

        self.log.emit(f'\n  Index saved: {indexPath}')


class RunTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread = None
        self._worker = None
        self._stopEvent = threading.Event()
        self._runStartTime = None
        self._buildUi()

    def _buildUi(self):
        layout = QVBoxLayout(self)

        btnRow = QHBoxLayout()
        self.startBtn = QPushButton('Start')
        self.startBtn.clicked.connect(self._start)
        btnRow.addWidget(self.startBtn)

        self.stopBtn = QPushButton('Stop')
        self.stopBtn.setEnabled(False)
        self.stopBtn.clicked.connect(self._stop)
        btnRow.addWidget(self.stopBtn)
        btnRow.addStretch()
        layout.addLayout(btnRow)

        self.statusLabel = QLabel('Ready')
        layout.addWidget(self.statusLabel)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.progressBar.setFormat('%v / %m  (%p%)')
        layout.addWidget(self.progressBar)

        self.etaLabel = QLabel('')
        self.etaLabel.setStyleSheet('color: gray; font-size: 11px;')
        layout.addWidget(self.etaLabel)

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        layout.addWidget(self.logText, stretch=1)

    def _start(self):
        plates = self.state.get('plates', [])
        if not plates:
            self.logText.append('ERROR: No plates selected. Go to Setup tab.')
            return

        stateDict = self.state.to_dict()

        self.logText.clear()
        self._stopEvent.clear()
        self._runStartTime = time.perf_counter()

        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.etaLabel.setText('')
        self.statusLabel.setText('Scanning plates…')
        self.progressBar.setValue(0)

        self._thread = QThread()
        self._worker = ProcessingWorker(stateDict, self._stopEvent)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.overallProgress.connect(self._onOverallProgress)
        self._worker.log.connect(self._onLog)
        self._worker.finished.connect(self._onFinished)
        self._worker.error.connect(self._onError)

        self._thread.start()

    def _stop(self):
        self._stopEvent.set()
        self.logText.append('Stopping...')
        self.stopBtn.setEnabled(False)

    def _onOverallProgress(self, done, total, desc):
        self.progressBar.setMaximum(max(total, 1))
        self.progressBar.setValue(done)
        self.statusLabel.setText(desc)
        if done > 0 and self._runStartTime is not None:
            elapsed = time.perf_counter() - self._runStartTime
            etaSecs = elapsed / done * (total - done) if done < total else 0
            self.etaLabel.setText(
                f'Elapsed: {_fmtTime(elapsed)}  ·  ETA: {_fmtTime(etaSecs)}'
            )

    def _onLog(self, msg):
        self.logText.append(msg)
        sb = self.logText.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _onError(self, msg):
        self.logText.append(f'ERROR: {msg}')

    def _onFinished(self):
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        stopped = self._stopEvent.is_set()
        if stopped:
            self.logText.append('\nStopped by user.')
            self.statusLabel.setText('Stopped')
        else:
            self.logText.append('\nDone.')
            self.progressBar.setValue(self.progressBar.maximum())
            self.statusLabel.setText('Complete')
        if self._runStartTime is not None:
            elapsed = time.perf_counter() - self._runStartTime
            self.etaLabel.setText(f'Total time: {_fmtTime(elapsed)}')

        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
