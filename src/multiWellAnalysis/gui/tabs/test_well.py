import os
import glob
import re
import threading
import numpy as np
from collections import defaultdict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSlider,
    QPushButton, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

import tifffile

from multiWellAnalysis.gui.tabs.preview import (
    discoverWellsWithMag, MAG_SUFFIXES,
)


def _makeLabelCmap(nLabels):
    """Create a random colormap for label visualization."""
    rng = np.random.RandomState(42)
    colors = rng.rand(max(nLabels + 1, 2), 3)
    colors[0] = [0, 0, 0]
    return ListedColormap(colors)


def _makeColorTable(nLabels):
    """Return (nLabels+1, 3) float64 color array; index 0 is black."""
    rng = np.random.RandomState(42)
    table = np.zeros((max(nLabels + 1, 2), 3), dtype=np.float64)
    table[1:] = rng.rand(max(nLabels, 1), 3)
    return table


class TestWellTab(QWidget):
    _wellResult = Signal(object)
    _runLog = Signal(str)
    _runProgress = Signal(str, int, int)  # stage, current, total
    _runFinished = Signal(object)  # result dict or None

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._wellResult.connect(self._onWellsDiscovered)
        self._runLog.connect(self._onLog)
        self._runProgress.connect(self._onProgress)
        self._runFinished.connect(self._onRunFinished)
        self._wellEntries = []
        self._filteredEntries = []
        self._result = None
        self._running = False
        self._stopEvent = threading.Event()
        self._colorTable = None   # precomputed per result

        self._renderTimer = QTimer(self)
        self._renderTimer.setSingleShot(True)
        self._renderTimer.setInterval(60)  # 60 ms debounce
        self._renderTimer.timeout.connect(self._render)

        self._buildUi()
        self._connectSignals()

    def _buildUi(self):
        layout = QVBoxLayout(self)

        selRow = QHBoxLayout()
        selRow.addWidget(QLabel('Plate:'))
        self.plateCombo = QComboBox()
        selRow.addWidget(self.plateCombo, stretch=1)

        selRow.addWidget(QLabel('Mag:'))
        self.magCombo = QComboBox()
        selRow.addWidget(self.magCombo)

        selRow.addWidget(QLabel('Well:'))
        self.wellCombo = QComboBox()
        selRow.addWidget(self.wellCombo, stretch=1)
        layout.addLayout(selRow)

        runRow = QHBoxLayout()
        self.runBtn = QPushButton('Run Full Pipeline on Well')
        runRow.addWidget(self.runBtn)
        self.stopBtn = QPushButton('Stop')
        self.stopBtn.setEnabled(False)
        runRow.addWidget(self.stopBtn)
        self.statusLabel = QLabel('')
        self.statusLabel.setStyleSheet('color: gray; font-size: 11px;')
        self.statusLabel.setWordWrap(True)
        runRow.addWidget(self.statusLabel, stretch=1)
        layout.addLayout(runRow)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        layout.addWidget(self.progressBar)

        frameRow = QHBoxLayout()
        frameRow.addWidget(QLabel('Frame:'))
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setRange(0, 0)
        frameRow.addWidget(self.frameSlider, stretch=1)
        self.frameLabel = QLabel('0 / 0')
        frameRow.addWidget(self.frameLabel)
        layout.addLayout(frameRow)

        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.axRaw = self.figure.add_subplot(1, 3, 1)
        self.axLabels = self.figure.add_subplot(1, 3, 2)
        self.axOverlay = self.figure.add_subplot(1, 3, 3)

        for ax in self.figure.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # Persistent image handles — updated via set_data() to avoid full relayout
        _blank2 = np.zeros((2, 2), dtype=np.float32)
        _blank3 = np.zeros((2, 2, 3), dtype=np.float32)
        self._imRaw = self.axRaw.imshow(_blank2, cmap='gray')
        self._imLabels = self.axLabels.imshow(_blank2, interpolation='nearest')
        self._imOverlay = self.axOverlay.imshow(_blank3)
        self.axRaw.set_title('Raw')
        self.axLabels.set_title('Tracked Labels')
        self.axOverlay.set_title('Colony Overlay')

        self.figure.tight_layout()
        layout.addWidget(self.canvas, stretch=1)

    def _connectSignals(self):
        self.plateCombo.currentIndexChanged.connect(self._onPlateChanged)
        self.magCombo.currentIndexChanged.connect(self._onMagChanged)
        self.wellCombo.currentIndexChanged.connect(self._onWellChanged)
        self.frameSlider.valueChanged.connect(self._onFrameChanged)
        self.runBtn.clicked.connect(self._runPipeline)
        self.stopBtn.clicked.connect(self._stopPipeline)
        self.state.changed.connect(self._onStateChanged)

    def _onStateChanged(self):
        if not self.isVisible():
            self._stale = True
            return
        self._stale = False
        plates = self.state.get('plates', [])
        currentPlates = [
            self.plateCombo.itemData(i) for i in range(self.plateCombo.count())
        ]
        if plates != currentPlates:
            self._populatePlates()
        elif self._wellEntries:
            magSetting = self.state.get('magnification', 'all')
            if magSetting != getattr(self, '_lastMagSetting', None):
                self._onWellsDiscovered(self._wellEntries)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_stale', False):
            self._stale = False
            self._onStateChanged()

    def _populatePlates(self):
        prevPlate = self.plateCombo.currentData()
        self.plateCombo.blockSignals(True)
        self.plateCombo.clear()
        restoreIdx = 0
        for i, p in enumerate(self.state.get('plates', [])):
            self.plateCombo.addItem(os.path.basename(p), p)
            if p == prevPlate:
                restoreIdx = i
        self.plateCombo.blockSignals(False)
        if self.plateCombo.count() > 0:
            self.plateCombo.setCurrentIndex(restoreIdx)
            self._onPlateChanged(restoreIdx)

    def _onPlateChanged(self, idx):
        platePath = self.plateCombo.currentData()
        if not platePath:
            self._wellEntries = []
            self.wellCombo.clear()
            return

        self.magCombo.clear()
        self.magCombo.setEnabled(False)
        self.wellCombo.clear()
        self.wellCombo.addItem('Scanning...')
        self.wellCombo.setEnabled(False)

        cached = self.state.cache_get(('wells', platePath))
        if cached is not None:
            self._onWellsDiscovered(cached)
            return

        def _scan():
            try:
                entries = discoverWellsWithMag(
                    platePath,
                    plateMeta=self.state.get('plateMeta', {}).get(platePath, {}),
                )
                self.state.cache_set(('wells', platePath), entries)
                return entries
            except Exception:
                return []

        threading.Thread(target=lambda: self._wellResult.emit(_scan()), daemon=True).start()

    def _onWellsDiscovered(self, entries):
        self._wellEntries = entries or []
        self.wellCombo.setEnabled(True)
        self.magCombo.setEnabled(True)

        allMags = sorted({mag for _, _, mag, _ in self._wellEntries if mag})
        magSetting = self.state.get('magnification', 'all')
        if magSetting == 'all':
            mags = allMags
        elif isinstance(magSetting, list):
            mags = [m for m in allMags if m in magSetting]
        else:
            mags = [m for m in allMags if m == magSetting]
        self._lastMagSetting = magSetting

        platePath = self.plateCombo.currentData()
        plateMeta = self.state.get('plateMeta', {}).get(platePath, {}) if platePath else {}
        prevMag = self.magCombo.currentData()
        self.magCombo.blockSignals(True)
        self.magCombo.clear()
        if not mags:
            self.magCombo.addItem('(none)', '')
        else:
            restoreIdx = 0
            for i, mag in enumerate(mags):
                obj = plateMeta.get(mag, {}).get('objective')
                magLabel = f'{obj}x' if obj else MAG_SUFFIXES.get(mag, mag)
                self.magCombo.addItem(magLabel, mag)
                if mag == prevMag:
                    restoreIdx = i
            self.magCombo.setCurrentIndex(restoreIdx)
        self.magCombo.blockSignals(False)

        self._populateWellsForMag()

    def _onMagChanged(self, idx):
        self._populateWellsForMag()

    def _populateWellsForMag(self):
        selectedMag = self.magCombo.currentData() or ''
        filtered = [(label, well, mag, source)
                     for label, well, mag, source in self._wellEntries
                     if mag == selectedMag]

        prevWell = self.wellCombo.currentData()
        self.wellCombo.blockSignals(True)
        self.wellCombo.clear()
        restoreIdx = 0
        for i, (label, well, mag, source) in enumerate(filtered):
            self.wellCombo.addItem(well, i)
            if well == prevWell:
                restoreIdx = i
        self.wellCombo.blockSignals(False)

        self._filteredEntries = filtered

        if self.wellCombo.count() > 0:
            self.wellCombo.setCurrentIndex(restoreIdx)

    def _onWellChanged(self, idx):
        self._result = None
        self._colorTable = None
        self._clearCanvas()

    def _getSelectedWell(self):
        """Return (plate_path, well_id, mag, source) or None."""
        idx = self.wellCombo.currentIndex()
        platePath = self.plateCombo.currentData()
        if idx < 0 or idx >= len(self._filteredEntries) or not platePath:
            return None
        label, well, mag, source = self._filteredEntries[idx]
        return platePath, well, mag, source

    def _stopPipeline(self):
        self._stopEvent.set()
        self.stopBtn.setEnabled(False)
        self.statusLabel.setText('Stopping...')

    def _runPipeline(self):
        if self._running:
            return

        sel = self._getSelectedWell()
        if not sel:
            self.statusLabel.setText('Select a plate, mag, and well first')
            return

        platePath, wellId, mag, source = sel
        self._running = True
        self._stopEvent.clear()
        self.runBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.progressBar.setValue(0)
        self.statusLabel.setText(f'Running pipeline on {wellId}...')

        s = self.state.to_dict()

        magParams = s.get('magParams', {})
        if mag and mag in magParams:
            s.update(magParams[mag])

        stop = self._stopEvent

        def _work():
            try:
                import tempfile
                from multiWellAnalysis.processing.analysis_main import timelapseProcessing
                from multiWellAnalysis.colony.runTrackingGUI import (
                    trackColoniesAllFrames, findSeedFrame,
                )

                if stop.is_set():
                    self._runFinished.emit(None)
                    return

                self._runLog.emit(f'Loading images for {wellId}...')
                self._runProgress.emit('Loading', 0, 5)

                if isinstance(source, str):
                    raw = tifffile.imread(source)
                    if raw.ndim == 2:
                        stack = raw[np.newaxis].astype(np.float32)
                    else:
                        stack = raw.astype(np.float32)
                    del raw
                else:
                    first = tifffile.imread(source[0])
                    h, w = first.shape[:2]
                    stack = np.empty((len(source), h, w), dtype=np.float32)
                    stack[0] = first.astype(np.float32)
                    del first
                    for fi in range(1, len(source)):
                        if stop.is_set():
                            self._runFinished.emit(None)
                            return
                        self._runLog.emit(f'Loading frame {fi+1}/{len(source)}...')
                        stack[fi] = tifffile.imread(source[fi]).astype(np.float32)

                # ensure (H, W, T)
                if stack.ndim == 3 and stack.shape[0] < stack.shape[2]:
                    stack = np.transpose(stack, (1, 2, 0))

                if stop.is_set():
                    self._runFinished.emit(None)
                    return

                self._runProgress.emit('Processing', 1, 5)
                ntimepoints = stack.shape[2]
                self._runLog.emit(f'Step 1/3: Processing {ntimepoints} frames...')

                with tempfile.TemporaryDirectory() as tmpdir:
                    masks, biomass, _ = timelapseProcessing(
                        images=stack,
                        blockDiameter=s['blockDiam'],
                        ntimepoints=ntimepoints,
                        shiftThresh=s['shiftThresh'],
                        fixedThresh=s['fixedThresh'],
                        dustCorrection=s['dustCorrection'],
                        outdir=tmpdir,
                        filename=wellId,
                        imageRecords=None,
                        fftStride=s.get('fftStride', 6),
                        downsample=s.get('downsample', 4),
                        skipOverlay=True,
                        workers=1,
                        progressFn=lambda msg: self._runLog.emit(f'  {msg}'),
                    )
                    import os as _os
                    procDir = _os.path.join(tmpdir, 'processedImages')
                    rawPath = _os.path.join(procDir, f'{wellId}_registered_raw.tif')
                    if _os.path.exists(rawPath):
                        registeredRaw = tifffile.imread(rawPath)
                        if registeredRaw.ndim == 3 and registeredRaw.shape[0] < registeredRaw.shape[1]:
                            registeredRaw = np.transpose(registeredRaw, (1, 2, 0))
                    else:
                        registeredRaw = stack

                if stop.is_set():
                    self._runFinished.emit(None)
                    return

                ntimepoints = masks.shape[2]

                self._runLog.emit('Step 2/3: Colony tracking...')
                self._runProgress.emit('Tracking', 3, 5)

                seedFrame = findSeedFrame(biomass)
                if seedFrame is None:
                    maskAreas = np.array([masks[..., t].sum() for t in range(ntimepoints)], dtype=float)
                    if maskAreas.max() > 0:
                        seedFrame = findSeedFrame(maskAreas / maskAreas.max())
                if seedFrame is None:
                    seedFrame = 0

                peakFrame = int(np.argmax([masks[..., t].sum() for t in range(ntimepoints)]))
                self._runLog.emit(f'  Seed frame: {seedFrame}, Peak frame: {peakFrame}')

                labelsByFrame, _, reason, frames = trackColoniesAllFrames(
                    registeredRaw, masks, seedFrame, peakFrame,
                    min_area=s.get('minColonyAreaPx', 200),
                    prop_radius=s.get('propRadiusPx', 25),
                )

                if stop.is_set():
                    self._runFinished.emit(None)
                    return

                self._runLog.emit(f'Step 3/3: Building result ({reason})')
                self._runProgress.emit('Building result', 4, 5)

                if labelsByFrame and frames:
                    firstLabel = labelsByFrame[list(labelsByFrame.keys())[0]]
                    lh, lw = firstLabel.shape[:2]
                    labelStack = np.zeros((lh, lw, len(frames)), dtype=np.int32)
                    for i, fIdx in enumerate(frames):
                        if fIdx in labelsByFrame:
                            labelStack[:, :, i] = labelsByFrame[fIdx][:lh, :lw]
                else:
                    labelStack = None
                    frames = list(range(ntimepoints))

                result = {
                    'raw_stack': registeredRaw,
                    'label_stack': labelStack,
                    'frames': frames,
                    'well_id': wellId,
                }

                self._runProgress.emit('Done', 5, 5)
                self._runFinished.emit(result)

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._runLog.emit(f'Error: {e}\n{tb}')
                print(tb)
                self._runFinished.emit(None)

        threading.Thread(target=_work, daemon=True).start()

    def _onLog(self, msg):
        self.statusLabel.setText(msg)

    def _onProgress(self, stage, current, total):
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(current)
        self.progressBar.setFormat(f'{stage} ({current}/{total})')

    def _onRunFinished(self, result):
        self._running = False
        self.runBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self._result = result

        if result is None:
            if self._stopEvent.is_set():
                self.statusLabel.setText('Stopped by user')
            return

        self.statusLabel.setText(f'Done — {result["well_id"]}')

        # Precompute color table from the maximum label across the whole stack
        labelStack = result.get('label_stack')
        if labelStack is not None and labelStack.size > 0:
            nMax = int(labelStack.max())
            self._colorTable = _makeColorTable(nMax)
        else:
            self._colorTable = None

        nFrames = len(result['frames']) if result['frames'] else 0
        if nFrames > 0:
            self.frameSlider.blockSignals(True)
            self.frameSlider.setRange(0, nFrames - 1)
            self.frameSlider.setValue(0)
            self.frameSlider.blockSignals(False)
            self.frameLabel.setText(f'0 / {nFrames - 1}')

        self._render()

    def _onFrameChanged(self, val):
        n = len(self._result['frames']) if self._result and self._result['frames'] else 0
        self.frameLabel.setText(f'{val} / {max(0, n - 1)}')
        self._renderTimer.start()  # debounce — fires _render after 60 ms of silence

    def _clearCanvas(self):
        _blank2 = np.zeros((2, 2), dtype=np.float32)
        _blank3 = np.zeros((2, 2, 3), dtype=np.float32)
        self._imRaw.set_data(_blank2)
        self._imLabels.set_data(_blank2)
        self._imOverlay.set_data(_blank3)
        self.axRaw.set_title('Raw')
        self.axLabels.set_title('Tracked Labels')
        self.axOverlay.set_title('Colony Overlay')
        self.canvas.draw_idle()

    def _render(self):
        if self._result is None:
            self.axRaw.set_title('Raw\n(run pipeline first)')
            self.axLabels.set_title('Tracked Labels')
            self.axOverlay.set_title('Colony Overlay')
            self.canvas.draw_idle()
            return

        frameIdx = self.frameSlider.value()

        rawStack = self._result.get('raw_stack')
        if rawStack is None:
            self.axRaw.set_title('No data')
            self.canvas.draw_idle()
            return

        fi = min(frameIdx, rawStack.shape[2] - 1)
        raw = rawStack[:, :, fi].astype(np.float64)

        self._imRaw.set_data(raw)
        self._imRaw.autoscale()
        self.axRaw.set_title('Raw')

        labelStack = self._result.get('label_stack')
        if labelStack is not None and labelStack.shape[2] > 0:
            fi2 = min(frameIdx, labelStack.shape[2] - 1)
            labelFrame = labelStack[:, :, fi2]
            nTracked = int(labelFrame.max())

            # Update label image
            cmap = _makeLabelCmap(nTracked) if nTracked > 0 else 'gray'
            self._imLabels.set_data(labelFrame)
            self._imLabels.set_cmap(cmap)
            self._imLabels.set_clim(0, max(nTracked, 1))
            self.axLabels.set_title(f'Tracked Labels\n{nTracked} colonies', fontsize=9)

            # Build overlay — vectorized: no per-colony loop
            rmax = raw.max()
            rawNorm = (raw / rmax) if rmax > 0 else raw.copy()
            overlay = np.stack([rawNorm, rawNorm, rawNorm], axis=-1)

            if nTracked > 0 and self._colorTable is not None:
                h = min(overlay.shape[0], labelFrame.shape[0])
                w = min(overlay.shape[1], labelFrame.shape[1])
                lf = labelFrame[:h, :w]
                # Clamp label indices to color table size
                lf_clamped = np.clip(lf, 0, len(self._colorTable) - 1)
                colMap = self._colorTable[lf_clamped]   # (H, W, 3)
                mask3d = (lf > 0)[:, :, np.newaxis]
                overlay[:h, :w] = np.where(mask3d,
                                            overlay[:h, :w] * 0.5 + colMap * 0.5,
                                            overlay[:h, :w])

            self._imOverlay.set_data(overlay)
            self._imOverlay.set_clim(0, 1)
            self.axOverlay.set_title('Colony Overlay', fontsize=9)
        else:
            self.axLabels.set_title('Tracked Labels\n(no results)', fontsize=9)
            self.axOverlay.set_title('Colony Overlay\n(no results)', fontsize=9)

        self.canvas.draw_idle()
