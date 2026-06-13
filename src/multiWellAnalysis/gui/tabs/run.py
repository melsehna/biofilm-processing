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
from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from ..buildinfo import buildRecord


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
    'magnification', 'magParams',
]

_runParamsFile = 'run_params.json'

# Metadata key carrying the pipeline build/version provenance. Underscore-
# prefixed and deliberately NOT in `_paramKeys`, so `_paramsMatch` ignores it
# (a code-version bump records provenance without forcing a full reprocess on
# resume — the warning at the resume site surfaces the drift instead).
_versionStampKey = '_pipelineVersion'


def _extractRunParams(state):
    return {k: state.get(k) for k in _paramKeys}


def _saveRunParams(outdir, params):
    path = os.path.join(outdir, _runParamsFile)
    payload = {**params, _versionStampKey: buildRecord()}
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def _loadRunParams(outdir):
    path = os.path.join(outdir, _runParamsFile)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _paramsMatch(saved, current):
    """Lenient comparison: saved having extra keys we don't track is OK.

    Strict `saved == current` would break cross-GUI resume because the two
    GUIs historically wrote slightly different key sets (e.g. microTyper-Vision's
    `copyRaw` doesn't exist here). What we actually care about is whether the
    *current* preprocessing knobs were the ones used to produce the existing
    outputs. So: every key in `current` must exist in `saved` with the same
    value. Extra keys in `saved` are ignored.
    """
    if not isinstance(saved, dict):
        return False
    for k, v in current.items():
        if k not in saved or saved[k] != v:
            return False
    return True


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

        # Preserve source dtype (typically uint16) so the downstream
        # _toBitDepthScaled() can use np.iinfo(dtype).max for exact scaling.
        # Casting to float32 here would force _toBitDepthScaled into its
        # observed-max heuristic, which misclassifies dim uint16 wells
        # (max < 256) as uint8 and over-scales by 256×.
        if isinstance(wellFiles, str):
            raw = tifffile.imread(wellFiles)
            stack = raw[np.newaxis] if raw.ndim == 2 else raw
            del raw
        else:
            first = tifffile.imread(wellFiles[0])
            h, w = first.shape[:2]
            stack = np.empty((len(wellFiles), h, w), dtype=first.dtype)
            stack[0] = first
            del first
            for fi in range(1, len(wellFiles)):
                stack[fi] = tifffile.imread(wellFiles[fi])

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
            saveProcessedVideo=params.get('saveProcessedVideo', False),
            saveFpHalf=params.get('saveFpHalf', True),
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
            min_colony_area=trackingParams.get('minColonyAreaPx'),
            prop_radius=trackingParams.get('propRadiusPx'),
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
    """Run whole-image feature extraction on a single well.

    REQUIRES the fixed-fpMean rendering (`_processed_fpHalf.tif`). Intensity and
    Haralick features are computed on the display image, and the adaptive
    `_processed.tif` render drifts batch-to-batch (different fpMean offset, and
    it clips differently), corrupting cross-batch feature comparisons. The
    adaptive render is therefore retired as a feature input — error loudly
    rather than silently mixing renders across batches. See CLAUDE.md "fpMean
    policy" and ISSUES.md.
    """
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    wellId = row['well']
    try:
        from multiWellAnalysis.wholeImage.runWholeImageGUI import extractWholeImageFeatures
        outdir = os.path.dirname(row['processed'])
        fpHalfPath = os.path.join(outdir, f'{wellId}_processed_fpHalf.tif')
        if not os.path.exists(fpHalfPath):
            return {'well': wellId, 'status': 'error',
                    'error': f'missing fixed-fpMean render {wellId}_processed_fpHalf.tif'
                             ' — whole-image features require it (adaptive '
                             '_processed.tif retired as a feature input). Reprocess '
                             'this well with saveFpHalf=True.'}
        inputPath = fpHalfPath
        t0 = time.perf_counter()
        status = extractWholeImageFeatures(
            inputPath, plateName, wellId, outdir
        )
        elapsed = time.perf_counter() - t0
        featsPath = os.path.join(outdir, f'{wellId}_wholeImageFeatures.csv')
        featsPath = featsPath if os.path.exists(featsPath) else ''
        return {
            'well': wellId,
            'status': 'done' if featsPath else status,
            'elapsed': elapsed,
            'whole_image_feats': featsPath,
        }
    except Exception as e:
        return {'well': wellId, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _colonyFeatsOneWell(plateName, row):
    """Run colony feature extraction on a single well.

    REQUIRES the fixed-fpMean processed rendering (`_processed_fpHalf.tif`) for
    intensity. The `_registered_raw.tif` fallback is retired: raw-stack
    intensities are uncorrected and drift batch-to-batch, so mixing raw-read
    wells with fpHalf-rendered wells corrupts cross-batch comparisons. Error
    loudly if the fixed render is missing. See CLAUDE.md "fpMean policy" and
    ISSUES.md.
    """
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    wellId = row['well']
    try:
        from multiWellAnalysis.colony.runColonyFeatsGUI import extractAndSave

        labelsPath = row['tracked_labels']
        rawPath = row['registered_raw']
        outdir = os.path.dirname(rawPath)
        fpHalfPath = os.path.join(outdir, f'{wellId}_processed_fpHalf.tif')
        if not os.path.exists(fpHalfPath):
            return {'well': wellId, 'status': 'error',
                    'error': f'missing fixed-fpMean render {wellId}_processed_fpHalf.tif'
                             ' — colony intensity features require it (registered-raw '
                             'fallback retired). Reprocess this well with '
                             'saveFpHalf=True.'}
        intensityPath = fpHalfPath

        data = np.load(labelsPath)
        rawStack = tifffile.imread(intensityPath)
        if rawStack.ndim == 3 and rawStack.shape[0] < rawStack.shape[1]:
            rawStack = np.transpose(rawStack, (1, 2, 0))

        labels = data['labels']
        frames = data['frames']
        wasTracked = bool(data['wasTracked']) if 'wasTracked' in data else True

        pxToUmRaw = row.get('pxToUm', '')
        if pxToUmRaw in ('', None):
            return {'well': wellId, 'status': 'error',
                    'error': 'missing pxToUm — run Setup → Detect from files '
                             'to probe TIFF metadata for this plate'}
        pxToUm = float(pxToUmRaw)
        if pxToUm <= 0:
            return {'well': wellId, 'status': 'error',
                    'error': f'invalid pxToUm={pxToUm!r}'}

        t0 = time.perf_counter()
        # Pass intensityPath (actual file read) for provenance, not rawPath.
        colonyDf, wellDf = extractAndSave(
            rawStack, labels, frames,
            plateName, wellId, wasTracked,
            labelsPath, intensityPath,
            outdir=outdir,
            pxToUm=pxToUm,
        )
        elapsed = time.perf_counter() - t0

        colonyPath = os.path.join(outdir, f'{wellId}_perColonyFeatures.csv')
        aggPath = os.path.join(outdir, f'{wellId}_wellColonyFeatures.csv')

        return {
            'well': wellId,
            'status': 'done',
            'elapsed': elapsed,
            'colony_feats': colonyPath if os.path.exists(colonyPath) else '',
            'well_colony_feats': aggPath if os.path.exists(aggPath) else '',
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

        nasMirror = bool(s.get('nasMirrorEnabled', False)) and bool(s.get('nasMirrorDir', '').strip())
        self._stagingAutoCreated = False
        if nasMirror:
            nasMirrorDir = s['nasMirrorDir'].strip()
            self.log.emit(f'\nNAS mirror enabled — outputs will be rsynced to {nasMirrorDir} '
                          f'after each plate and the local copy deleted.')
            if not self._preflightNasMirror(outputRoot, nasMirrorDir):
                return
            # Auto-stage to a fresh local dir if the user-provided outputDir is
            # empty or appears to live on the same mount as nasMirrorDir
            # (in which case mirroring would be a same-mount copy and provide
            # no speedup).
            outputRoot, self._stagingAutoCreated = self._resolveLocalStagingDir(
                outputRoot, nasMirrorDir,
            )
            self.log.emit(f'  [NAS mirror] local staging dir: {outputRoot}')

        doWhole = s.get('wholeImageFeats', False)
        doTracking = s.get('colonyTracking', False) or s.get('colonyFeats', False)
        doColonyFeats = s.get('colonyFeats', False)
        nStages = 1 + int(doWhole) + int(doTracking) + int(doColonyFeats)

        enabled = ['biomass']
        if doWhole: enabled.append('whole-image')
        if doTracking: enabled.append('tracking')
        if doColonyFeats: enabled.append('colony-feats')
        self.log.emit(f'Enabled stages: {", ".join(enabled)} ({nStages} total)')
        self.log.emit(f'  wholeImageFeats={s.get("wholeImageFeats")}, '
                      f'colonyTracking={s.get("colonyTracking")}, '
                      f'colonyFeats={s.get("colonyFeats")}, '
                      f'saveOverlays={s.get("saveOverlays")}, '
                      f'saveProcessedVideo={s.get("saveProcessedVideo")}, '
                      f'saveFpHalf={s.get("saveFpHalf")}')

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

                # Look up per-plate metadata cached by Setup tab's scan;
                # fall back to probing if this plate was never detected
                # (e.g. drawer-expanded plates, configs loaded without a
                # fresh detection click).
                plateMeta = s.get('plateMeta', {})
                suffixMeta = plateMeta.get(userPath) or plateMeta.get(resolvedPlate)
                if not suffixMeta:
                    self.log.emit(f'  No cached metadata for this plate — probing now')
                    from multiWellAnalysis.processing.image_metadata import probePlateMeta
                    suffixMeta = probePlateMeta(resolvedPlate, logFn=self.log.emit)

                outdir = _computeOutdir(userPath, resolvedPlate, outputRoot)
                os.makedirs(outdir, exist_ok=True)
                self.log.emit(f'  Output dir: {outdir}')

                plateOutdirs.append(outdir)
                drawerMap[plateName] = drawerName if drawerName else plateName

                # per-plate resume: check if output already exists with same params
                saved = _loadRunParams(outdir)
                resume = False
                if _paramsMatch(saved, runParams):
                    resume = True
                    # Surface pipeline-version drift: resuming over outputs that
                    # a different code version produced mixes feature provenance.
                    savedVer = (saved or {}).get(_versionStampKey) or {}
                    curVer = buildRecord()
                    if savedVer.get('gitCommit') != curVer.get('gitCommit') or \
                            savedVer.get('version') != curVer.get('version'):
                        self.log.emit(
                            f'  WARNING: resuming over outputs from a different '
                            f'pipeline version (existing: {savedVer.get("build", "unstamped")} '
                            f'| current: {curVer.get("build")}). Features may mix '
                            f'across versions; consider a clean reprocess.')
                _saveRunParams(outdir, runParams)

                wellItems = list(wells.items())

                # Pre-populate index with per-well metadata (pxToUm, objective).
                # Suffixes with no metadata entry are stored with empty values;
                # colony feats will fail loudly rather than silently using a
                # wrong default.
                import re as _re
                index = {}
                missingMeta = set()
                for wellId in wells:
                    m = _re.search(r'(_\d+)$', wellId)
                    suffix = m.group(1) if m else ''
                    meta = suffixMeta.get(suffix)
                    if meta is None:
                        missingMeta.add(suffix)
                        index[wellId] = {'pxToUm': '', 'objective': ''}
                    else:
                        index[wellId] = {
                            'pxToUm': meta['pxToUm'],
                            'objective': meta['objective'],
                        }
                if missingMeta:
                    self.log.emit(
                        f'  WARNING: no metadata for suffixes {sorted(missingMeta)} — '
                        f'colony-feature extraction will fail for these wells'
                    )

                if resume:
                    # load previously-done wells into index so later stages can run on them
                    existingIndex = os.path.join(outdir, 'index.csv')
                    if os.path.exists(existingIndex):
                        try:
                            import csv as _csv
                            with open(existingIndex, newline='') as f:
                                for row in _csv.DictReader(f):
                                    wid = row.get('well', '')
                                    if not wid:
                                        continue
                                    target = index.setdefault(wid, {})
                                    for k, v in row.items():
                                        if k in ('plate', 'plate_path', 'well', 'mag'):
                                            continue
                                        # Never overwrite freshly-probed metadata
                                        # with possibly-stale CSV values
                                        if k in ('pxToUm', 'objective') and target.get(k) not in ('', None):
                                            continue
                                        target[k] = v
                        except Exception:
                            pass

                    # Disk reconciliation: index.csv is only written at end of
                    # plate, so a stop mid-stage-1 leaves no CSV — but the
                    # wells that DID finish have _processed.tif on disk. Fill
                    # their standard output paths into the index so stages 2-4
                    # see them instead of silently skipping with KeyError.
                    for wellId in list(index):
                        target = index[wellId]
                        if target.get('processed'):
                            continue
                        candidates = {
                            'processed':      os.path.join(outdir, f'{wellId}_processed.tif'),
                            'registered_raw': os.path.join(outdir, f'{wellId}_registered_raw.tif'),
                            'masks':          os.path.join(outdir, f'{wellId}_masks.npz'),
                            'biomass':        os.path.join(outdir, f'{wellId}_biomass.csv'),
                        }
                        if os.path.exists(candidates['processed']):
                            for k, p in candidates.items():
                                if os.path.exists(p):
                                    target[k] = p

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

                if self._stop.is_set():
                    plateIdx += 1
                    continue

                if s.get('wholeImageFeats'):
                    if index:
                        wellsWithProcessed = sum(1 for r in index.values() if 'processed' in r)
                        self.log.emit(f'\n  --- Stage 2: Whole-image features ({len(index)} wells, {wellsWithProcessed} with processed TIF) ---')
                        self._runStageParallel(
                            plateName, plateIdx, 0, 'Whole-image',
                            list(index.items()), index, outdir, nWorkers,
                            self._submitWholeImage, plateName
                        )
                    else:
                        self.log.emit(f'\n  Stage 2 skipped: no wells in index')
                else:
                    self.log.emit(f'\n  Stage 2 skipped: wholeImageFeats={s.get("wholeImageFeats")}')

                if self._stop.is_set():
                    plateIdx += 1
                    continue

                if s.get('colonyTracking') or s.get('colonyFeats'):
                    if index:
                        self.log.emit(f'\n  --- Stage 3: Colony tracking ({len(index)} wells) ---')
                        self._runStageParallel(
                            plateName, plateIdx, 0, 'Tracking',
                            list(index.items()), index, outdir, nWorkers,
                            self._submitTracking, plateName, s
                        )
                    else:
                        self.log.emit(f'\n  Stage 3 skipped: no wells in index')

                if self._stop.is_set():
                    plateIdx += 1
                    continue

                if s.get('colonyFeats'):
                    trackable = [(k, v) for k, v in index.items() if 'tracked_labels' in v]
                    if trackable:
                        self.log.emit(f'\n  --- Stage 4: Colony features ({len(trackable)} wells) ---')
                        self._runStageParallel(
                            plateName, plateIdx, 0, 'Colony feats',
                            trackable, index, outdir, nWorkers,
                            self._submitColonyFeats, plateName
                        )
                    else:
                        self.log.emit(f'\n  Stage 4 skipped: no tracked labels in index')

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

                if nasMirror:
                    # Sync the WHOLE plate dir (parent of processedImages) so
                    # numericalData/ and run_params.json get mirrored too. The
                    # old code synced just processedImages, which both lost
                    # numericalData on NAS AND left the plate dir partially
                    # populated locally → disk accumulating across plates.
                    plateDirLocal = os.path.dirname(outdir)
                    nasPlateDir = self._computeNasPlateDir(
                        outputRoot, plateDirLocal, s['nasMirrorDir'].strip(),
                    )
                    if self._syncPlateToNas(plateDirLocal, nasPlateDir):
                        # plateOutdirs entries are processedImages paths, but
                        # the local copy is gone now. Repoint at NAS so the
                        # final master CSV finds the data.
                        nasProcessedImages = os.path.join(nasPlateDir, 'processedImages')
                        for i, p in enumerate(plateOutdirs):
                            if p == outdir:
                                plateOutdirs[i] = nasProcessedImages
                                break

                plateIdx += 1

        if outputRoot and plateOutdirs and not self._stop.is_set():
            self.log.emit(f'\n{"="*60}\nAssembling master CSVs…')
            masterOk = False
            try:
                from multiWellAnalysis.processing.master_csv import assembleMasterCsvs
                assembleMasterCsvs(
                    plateOutdirs, drawerMap, outputRoot,
                    logFn=self.log.emit,
                )
                masterOk = True
            except Exception as e:
                self.log.emit(f'  [master CSV] ERROR: {e}')

            if masterOk:
                self._exportConditionsCsvs(outputRoot, s)

            if masterOk and (s.get('umapStatic') or s.get('umapInteractive')):
                self._runUmapStep(outputRoot, s)

            # After master CSV + UMAP, mirror the top-level master CSVs and
            # any other outputRoot-level artifacts to NAS. Per-plate dirs were
            # already mirrored + deleted, so what's left at outputRoot is just
            # the master_*.csv files plus the embeddings/ directory if any.
            if nasMirror:
                self._syncOutputRootToNas(outputRoot, s['nasMirrorDir'].strip())
                # If the staging dir was auto-created under home, tear it down
                # entirely after the final sync. User-provided outputDirs are
                # left alone.
                if self._stagingAutoCreated:
                    if self._forceDeleteDir(outputRoot):
                        self.log.emit(f'  [NAS mirror] cleaned up auto-staging dir: {outputRoot}')
                    else:
                        self.log.emit(f'  [NAS mirror] ERROR: failed to clean auto-staging dir '
                                      f'{outputRoot}; manual cleanup: rm -rf "{outputRoot}"')

    def _preflightNasMirror(self, outputRoot, nasMirrorDir):
        """Sanity checks before starting a NAS-mirror run.

        Returns True if OK to proceed, False to abort (and emits a log line).
        Note: outputRoot may be empty here — _resolveLocalStagingDir will
        auto-create one if so. We only fail if rsync is missing or NAS is
        unwritable.
        """
        import shutil
        if shutil.which('rsync') is None:
            self.log.emit('  ERROR: rsync not found on PATH; install rsync or disable NAS mirror.')
            return False
        if outputRoot and outputRoot.rstrip('/') == nasMirrorDir.rstrip('/'):
            self.log.emit(f'  ERROR: outputDir and nasMirrorDir are the same ({outputRoot}); '
                          f'NAS mirror mode requires a distinct local staging dir.')
            return False
        try:
            os.makedirs(nasMirrorDir, exist_ok=True)
            probe = os.path.join(nasMirrorDir, '.mtv_nas_write_probe')
            with open(probe, 'w') as f:
                f.write('ok')
            os.remove(probe)
        except Exception as e:
            self.log.emit(f'  ERROR: NAS mirror dir {nasMirrorDir!r} not writable: {e}')
            return False

        # Disk-space check happens against the resolved staging dir, not
        # outputRoot — which may be empty at this point. Defer to the
        # post-resolve helper.
        return True

    def _onSameMount(self, a, b):
        """Are paths a and b on the same filesystem mount? Used to detect
        when a user-provided outputDir is on the NAS (which would defeat
        the purpose of NAS-mirror staging)."""
        try:
            return os.stat(a).st_dev == os.stat(b).st_dev
        except Exception:
            return False

    def _resolveLocalStagingDir(self, userOutputRoot, nasMirrorDir):
        """Decide what local path to use as the NAS-mirror staging area.

        Logic:
          - If userOutputRoot is set AND exists AND is on a different mount
            from nasMirrorDir → use it as-is (assume user picked a fast
            local disk deliberately).
          - Otherwise → auto-create a fresh dir under $HOME and use that.

        Returns (resolvedPath, autoCreatedBool). autoCreatedBool is True
        when the dir was auto-created, so the caller knows to delete it
        at the very end of the run.
        """
        import datetime, shutil
        if userOutputRoot and os.path.isdir(userOutputRoot):
            if not self._onSameMount(userOutputRoot, nasMirrorDir):
                # Verify space on user's chosen dir
                freeGb = shutil.disk_usage(userOutputRoot).free / (1024 ** 3)
                if freeGb < 20:
                    self.log.emit(f'  [NAS mirror] WARNING: only {freeGb:.1f} GB free '
                                  f'at {userOutputRoot}; per-plate sync should keep up '
                                  f'but headroom is tight.')
                return userOutputRoot, False
            self.log.emit(f'  [NAS mirror] outputDir {userOutputRoot} is on the same mount '
                          f'as nasMirrorDir — auto-creating local staging dir under home '
                          f'instead so the mirror actually buys you speed.')
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        staging = os.path.expanduser(f'~/biofilm-staging-{stamp}')
        os.makedirs(staging, exist_ok=True)
        # Sanity-check free space at auto-created location
        freeGb = shutil.disk_usage(staging).free / (1024 ** 3)
        if freeGb < 20:
            self.log.emit(f'  [NAS mirror] WARNING: only {freeGb:.1f} GB free at '
                          f'{staging}; per-plate sync may not keep up. Consider '
                          f'pointing outputDir at a larger local disk.')
        else:
            self.log.emit(f'  [NAS mirror] auto-created staging dir has {freeGb:.0f} GB free.')
        return staging, True

    def _computeNasPlateDir(self, outputRoot, localPlateDir, nasMirrorDir):
        """Compute the NAS-side path that mirrors a local plate dir.

        Preserves the relative structure under outputRoot, so a local
        outputRoot/<drawer>/<plate>/processedImages becomes
        nasMirrorDir/<drawer>/<plate>/processedImages.
        """
        rel = os.path.relpath(localPlateDir, outputRoot)
        return os.path.join(nasMirrorDir, rel)

    def _syncPlateToNas(self, localPlateDir, nasPlateDir):
        """rsync local plate dir → NAS, then delete the local copy on success.

        Returns True if sync + delete succeeded, False otherwise.
        On failure, local dir is preserved so a re-run can pick up.
        """
        import shutil, subprocess
        if not os.path.isdir(localPlateDir):
            self.log.emit(f'  [NAS sync] skip — local plate dir missing: {localPlateDir}')
            return False
        os.makedirs(os.path.dirname(nasPlateDir.rstrip('/')) or nasPlateDir, exist_ok=True)
        self.log.emit(f'  [NAS sync] {localPlateDir} → {nasPlateDir}')
        # Trailing slashes matter: rsync src/ → dst means "contents of src into dst"
        srcArg = localPlateDir.rstrip('/') + '/'
        dstArg = nasPlateDir.rstrip('/') + '/'
        try:
            result = subprocess.run(
                ['rsync', '-a', '--info=progress2', srcArg, dstArg],
                capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                self.log.emit(f'  [NAS sync] rsync FAILED (rc={result.returncode}): '
                              f'{result.stderr.strip()[:500]}')
                return False
        except subprocess.TimeoutExpired:
            self.log.emit(f'  [NAS sync] rsync timed out (>1h) for {localPlateDir}')
            return False
        except Exception as e:
            self.log.emit(f'  [NAS sync] rsync exception: {e}')
            return False
        # Robust delete: try shutil.rmtree first, fall back to `rm -rf`,
        # then verify the dir is actually gone. The whole point of NAS
        # mirror mode is bounded local disk, so loud-failing on a stuck
        # local copy is the right behavior.
        self.log.emit(f'  [NAS sync] rsync OK — deleting local copy: {localPlateDir}')
        deleted = self._forceDeleteDir(localPlateDir)
        if deleted:
            self.log.emit(f'  [NAS sync] local copy deleted: {localPlateDir}')
        else:
            self.log.emit(f'  [NAS sync] ERROR: rsync OK but LOCAL COPY STILL PRESENT at '
                          f'{localPlateDir}. NAS data is safe but disk will fill up. '
                          f'Manual cleanup: rm -rf "{localPlateDir}"')
        return True

    def _forceDeleteDir(self, path):
        """Robust recursive delete. Tries shutil.rmtree first; if that fails
        or leaves anything behind, falls back to `rm -rf`. Returns True only
        when the directory is verified gone."""
        import shutil, subprocess
        if not os.path.exists(path):
            return True
        try:
            shutil.rmtree(path)
        except Exception as e:
            self.log.emit(f'    [delete] shutil.rmtree failed ({e}) — falling back to rm -rf')
        if os.path.exists(path):
            try:
                subprocess.run(['rm', '-rf', path], check=True, timeout=600)
            except Exception as e:
                self.log.emit(f'    [delete] rm -rf failed: {e}')
        return not os.path.exists(path)

    def _syncOutputRootToNas(self, outputRoot, nasMirrorDir):
        """Mirror outputRoot-level files (master CSVs, etc.) to NAS at the
        end of a run. Skips already-mirrored per-plate subdirectories."""
        import subprocess
        self.log.emit(f'\n[NAS sync] mirroring outputRoot-level artifacts → {nasMirrorDir}')
        srcArg = outputRoot.rstrip('/') + '/'
        dstArg = nasMirrorDir.rstrip('/') + '/'
        try:
            # update-only rsync; per-plate dirs were already deleted locally
            # so only the master_*.csv and any embeddings/ etc. remain
            result = subprocess.run(
                ['rsync', '-a', srcArg, dstArg],
                capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                self.log.emit(f'  [NAS sync] outputRoot rsync FAILED: '
                              f'{result.stderr.strip()[:500]}')
                return
            self.log.emit('  [NAS sync] outputRoot mirror complete.')
        except Exception as e:
            self.log.emit(f'  [NAS sync] outputRoot rsync exception: {e}')

    def _exportConditionsCsvs(self, outputRoot, s):
        """Write <outputRoot>/<plateName>/conditions.csv for each plate that
        has conditions defined in the GUI."""
        conditions = s.get('conditions', {})
        if not conditions:
            return
        for platePath, plateConds in conditions.items():
            if not plateConds:
                continue
            plateName = os.path.basename(os.path.normpath(platePath))
            plateOutDir = os.path.join(outputRoot, plateName)
            if not os.path.isdir(plateOutDir):
                continue
            rows = []
            for condName, wells in plateConds.items():
                for w in wells:
                    rows.append({'wellId': w, 'condition': condName})
            if not rows:
                continue
            try:
                import pandas as pd
                csvPath = os.path.join(plateOutDir, 'conditions.csv')
                pd.DataFrame(rows).to_csv(csvPath, index=False)
                self.log.emit(f'  [conditions] wrote {csvPath} ({len(rows)} rows)')
            except Exception as e:
                self.log.emit(f'  [conditions] ERROR writing for {plateName}: {e}')

    def _runUmapStep(self, outputRoot, s):
        """Generate UMAP outputs after master CSV assembly. Failures are logged
        and isolated — they don't abort the run summary."""
        masterPath = os.path.join(outputRoot, 'master_frame_features.csv')
        if not os.path.exists(masterPath):
            self.log.emit('  [UMAP] master_frame_features.csv missing, skipping.')
            return

        self.log.emit(f'\n{"="*60}\nGenerating UMAPs…')
        try:
            import pandas as pd
            from multiWellAnalysis.analysis.runner import runUmap
        except ImportError as e:
            self.log.emit(
                f'  [UMAP] umap-learn not installed — '
                f'install with `pip install -e ".[umap]"` and re-run. ({e})'
            )
            return

        try:
            mags = sorted(pd.read_csv(masterPath, usecols=['mag'])['mag']
                          .dropna().unique())
        except Exception as e:
            self.log.emit(f'  [UMAP] could not read master CSV mags: {e}')
            return

        plates = self._state.get('plates', [])
        plateDirMap = {os.path.basename(os.path.normpath(p)): p for p in plates}
        # state['conditions'] is keyed by full platePath; runner expects plateId
        rawConditions = self._state.get('conditions', {}) or {}
        conditionsByPlate = {os.path.basename(os.path.normpath(p)): c
                             for p, c in rawConditions.items()}
        columnName = self._state.get('umapColumnName') or None

        for mag in mags:
            if self._stop.is_set():
                self.log.emit('  [UMAP] stopped before completing.')
                return
            try:
                runUmap(
                    outputRoot,
                    magnification=mag,
                    doStatic=self._state.get('umapStatic', False),
                    doInteractive=self._state.get('umapInteractive', False),
                    plateDirMap=plateDirMap,
                    conditionsByPlate=conditionsByPlate,
                    columnName=columnName,
                    plateMeta=self._state.get('plateMeta'),
                    logFn=self.log.emit,
                )
            except Exception as e:
                self.log.emit(f'  [UMAP {mag}] ERROR: {e}')

    def _runStageParallel(self, plateName, plateIdx, totalPlates, stageName,
                          items, index, outdir, nWorkers, submitFn, *submitArgs):
        total = len(items)

        with ProcessPoolExecutor(max_workers=nWorkers) as pool:
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
                    for f in futures:
                        f.cancel()
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
            'saveProcessedVideo': state.get('saveProcessedVideo', False),
            'saveFpHalf': state.get('saveFpHalf', True),
        }
        magParams = state.get('magParams', {})
        if mag and mag in magParams:
            params.update(magParams[mag])

        return pool.submit(_processOneWell, platePath, outdir, wellId, wellFiles, params)

    def _submitWholeImage(self, pool, wellId, row, outdir, plateName):
        if 'processed' not in row:
            self.log.emit(f'  {wellId} whole-image skipped: no processed TIF in index')
            return None
        return pool.submit(_wholeImageOneWell, plateName, {**row, 'well': wellId})

    def _submitTracking(self, pool, wellId, row, outdir, plateName, state):
        if 'registered_raw' not in row:
            self.log.emit(f'  {wellId} tracking skipped: no registered_raw in index')
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

        self.openAnalysisBtn = QPushButton('Open analysis folder')
        self.openAnalysisBtn.setToolTip(
            'Opens the <outputRoot>/analysis directory containing UMAP outputs.'
        )
        self.openAnalysisBtn.clicked.connect(self._openAnalysisFolder)
        btnRow.addWidget(self.openAnalysisBtn)

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

    def _openAnalysisFolder(self):
        outputRoot = self.state.get('outputDir', '')
        if not outputRoot:
            QMessageBox.information(
                self, 'No output directory',
                'Set an output directory in the Setup tab first.'
            )
            return
        target = os.path.join(outputRoot, 'analysis')
        if not os.path.isdir(target):
            target = outputRoot  # fall back to outputRoot if analysis/ not yet made
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

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
