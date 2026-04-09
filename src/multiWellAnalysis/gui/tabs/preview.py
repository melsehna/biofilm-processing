import os
import glob
import re
import threading
import numpy as np
from collections import defaultdict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSlider,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import cv2
import tifffile

from multiWellAnalysis.processing.preprocessing import normalizeLocalContrast

MAG_SUFFIXES = {'_02': '4x', '_03': '10x', '_04': '20x', '_05': '40x'}


def discoverWellsWithMag(plateDir):
    """Find well+mag combinations from TIF filenames.

    Returns list of (display_label, well_id, mag_suffix, file_list_or_path) tuples.
    For plates without magnification suffixes, mag_suffix is ''.
    """
    if not plateDir or not os.path.isdir(plateDir):
        return []

    from multiWellAnalysis.gui.tabs.run import _resolveTifDir, _listRawTifs
    resolved = _resolveTifDir(plateDir, maxDepth=2)
    rawTifs = _listRawTifs(resolved)

    # fallback: if _listRawTifs found nothing, grab all .tif files directly
    if not rawTifs:
        try:
            rawTifs = sorted(
                os.path.join(resolved, f) for f in os.listdir(resolved)
                if f.lower().endswith('.tif')
            )
        except (PermissionError, OSError):
            pass

    bfFiles = [f for f in rawTifs if 'Bright Field' in f or 'Bright_Field' in f]

    candidates = bfFiles if bfFiles else rawTifs
    if candidates:
        groups = defaultdict(lambda: defaultdict(list))
        for f in candidates:
            base = os.path.basename(f)
            m = re.match(r'^([A-P]\d+)(_\d+)_', base)
            if m:
                well, mag = m.group(1), m.group(2)
                groups[mag][well].append(f)

        if groups:
            # Probe one TIFF per suffix to get actual objective from metadata
            from multiWellAnalysis.processing.image_metadata import readCytationMeta
            suffixObjective = {}
            for mag, wellDict in groups.items():
                for files in wellDict.values():
                    if files:
                        try:
                            meta = readCytationMeta(files[0])
                            suffixObjective[mag] = meta['objective']
                        except Exception:
                            pass
                        break

            result = []
            for mag in sorted(groups):
                for well in sorted(groups[mag]):
                    files = sorted(groups[mag][well])
                    obj = suffixObjective.get(mag)
                    magLabel = f'{obj}x' if obj else MAG_SUFFIXES.get(mag, mag)
                    label = f'{well} ({magLabel})'
                    result.append((label, well, mag, files))
            return result

    # Fallback: no magnification suffixes — group raw TIFs by well
    wells = defaultdict(list)
    for f in rawTifs:
        name = os.path.basename(f)
        m = re.match(r'^([A-P]\d{1,2})[_.]', name)
        if m:
            wells[m.group(1)].append(f)

    result = []
    for well in sorted(wells):
        result.append((well, well, '', sorted(wells[well])))
    return result


def loadFrame(source, frameIdx):
    """Load a single frame from a TIFF stack or list of files."""
    if isinstance(source, str):
        img = tifffile.imread(source)
        if img.ndim == 3:
            if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                nFrames = img.shape[0]
                frameIdx = min(frameIdx, nFrames - 1)
                return img[frameIdx].astype(np.float64), nFrames
            elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
                nFrames = img.shape[2]
                frameIdx = min(frameIdx, nFrames - 1)
                return img[:, :, frameIdx].astype(np.float64), nFrames
            else:
                nFrames = img.shape[0]
                frameIdx = min(frameIdx, nFrames - 1)
                return img[frameIdx].astype(np.float64), nFrames
        return img.astype(np.float64), 1
    elif isinstance(source, list):
        frameIdx = min(frameIdx, len(source) - 1)
        img = tifffile.imread(source[frameIdx])
        return img.astype(np.float64), len(source)
    return None, 0


class PreviewTab(QWidget):
    _wellResult = Signal(object)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._wellResult.connect(self._onWellsDiscovered)
        self._debounceTimer = QTimer()
        self._debounceTimer.setSingleShot(True)
        self._debounceTimer.setInterval(300)
        self._debounceTimer.timeout.connect(self._render)
        self._currentSource = None
        self._currentMag = ''
        self._currentWell = ''
        self._nFrames = 0
        self._wellEntries = []
        self._filteredEntries = []
        self._lastMagSetting = None
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

        frameRow = QHBoxLayout()
        frameRow.addWidget(QLabel('Frame:'))
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setRange(0, 0)
        frameRow.addWidget(self.frameSlider, stretch=1)
        self.frameLabel = QLabel('0 / 0')
        frameRow.addWidget(self.frameLabel)
        layout.addLayout(frameRow)

        self.paramsLabel = QLabel('')
        self.paramsLabel.setStyleSheet('color: gray; font-size: 11px;')
        layout.addWidget(self.paramsLabel)

        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.axRaw = self.figure.add_subplot(1, 3, 1)
        self.axProc = self.figure.add_subplot(1, 3, 2)
        self.axMask = self.figure.add_subplot(1, 3, 3)

        for ax in self.figure.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        self.figure.tight_layout()
        layout.addWidget(self.canvas, stretch=1)

        self.refreshBtn = QPushButton('Refresh')
        layout.addWidget(self.refreshBtn)

    def _connectSignals(self):
        self.plateCombo.currentIndexChanged.connect(self._onPlateChanged)
        self.magCombo.currentIndexChanged.connect(self._onMagChanged)
        self.wellCombo.currentIndexChanged.connect(self._onWellChanged)
        self.frameSlider.valueChanged.connect(self._onFrameChanged)
        self.refreshBtn.clicked.connect(self._refreshAll)
        self.state.changed.connect(self._onStateChanged)

    def _getParamsForMag(self, mag):
        """Get parameters with per-magnification overrides applied."""
        blockDiam = self.state.get('blockDiam', 101)
        fixedThresh = self.state.get('fixedThresh', 0.04)
        dustCorrection = self.state.get('dustCorrection', True)
        minColonyArea = self.state.get('minColonyAreaPx', 200)

        magParams = self.state.get('magParams', {})
        if mag and mag in magParams:
            overrides = magParams[mag]
            blockDiam = overrides.get('blockDiam', blockDiam)
            fixedThresh = overrides.get('fixedThresh', fixedThresh)
            dustCorrection = overrides.get('dustCorrection', dustCorrection)
            minColonyArea = overrides.get('minColonyAreaPx', minColonyArea)

        return blockDiam, fixedThresh, dustCorrection, minColonyArea

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
        else:
            magSetting = self.state.get('magnification', 'all')
            if magSetting != self._lastMagSetting and self._wellEntries:
                self._onWellsDiscovered(self._wellEntries)
            else:
                self._scheduleRender()

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
            self._loadSource()
            self._scheduleRender()
            return

        self.magCombo.clear()
        self.magCombo.setEnabled(False)
        self.wellCombo.clear()
        self.wellCombo.addItem('Scanning...')
        self.wellCombo.setEnabled(False)

        def _scan():
            try:
                return discoverWellsWithMag(platePath)
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

        suffixObjective = self.state.get('suffixObjective', {})
        prevMag = self.magCombo.currentData()
        self.magCombo.blockSignals(True)
        self.magCombo.clear()
        if not mags:
            self.magCombo.addItem('(none)', '')
        else:
            restoreIdx = 0
            for i, mag in enumerate(mags):
                obj = suffixObjective.get(mag)
                magLabel = f'{obj}x' if obj else MAG_SUFFIXES.get(mag, mag)
                self.magCombo.addItem(magLabel, mag)
                if mag == prevMag:
                    restoreIdx = i
            self.magCombo.setCurrentIndex(restoreIdx)
        self.magCombo.blockSignals(False)

        self._lastMagSetting = self.state.get('magnification', 'all')
        self._populateWellsForMag()

    def _onMagChanged(self, idx):
        self._populateWellsForMag()

    def _populateWellsForMag(self):
        """Filter well combo to show only wells for the selected magnification."""
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
        self._loadSource()
        self._scheduleRender()

    def _onWellChanged(self, idx):
        self._loadSource()
        self._scheduleRender()

    def _loadSource(self):
        idx = self.wellCombo.currentIndex()
        platePath = self.plateCombo.currentData()
        filtered = self._filteredEntries

        if idx < 0 or idx >= len(filtered) or not platePath:
            self._currentSource = None
            self._currentMag = ''
            self._currentWell = ''
            self._nFrames = 0
            return

        label, well, mag, source = filtered[idx]
        self._currentSource = source
        self._currentMag = mag
        self._currentWell = well

        if self._currentSource is not None:
            _, self._nFrames = loadFrame(self._currentSource, 0)
            oldVal = self.frameSlider.value()
            self.frameSlider.blockSignals(True)
            self.frameSlider.setRange(0, max(0, self._nFrames - 1))
            if oldVal <= self._nFrames - 1:
                self.frameSlider.setValue(oldVal)
            else:
                self.frameSlider.setValue(0)
            self.frameSlider.blockSignals(False)
            self.frameLabel.setText(
                f'{self.frameSlider.value()} / {max(0, self._nFrames - 1)}'
            )
        else:
            self._nFrames = 0
            self.frameSlider.setRange(0, 0)

    def _onFrameChanged(self, val):
        self.frameLabel.setText(f'{val} / {max(0, self._nFrames - 1)}')
        self._scheduleRender()

    def _scheduleRender(self):
        self._debounceTimer.start()

    def _refreshAll(self):
        self._populatePlates()

    def _render(self):
        for ax in self.figure.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        if self._currentSource is None:
            self.axRaw.set_title('No image')
            self.paramsLabel.setText('')
            self.canvas.draw()
            return

        frameIdx = self.frameSlider.value()
        raw, _ = loadFrame(self._currentSource, frameIdx)
        if raw is None:
            self.axRaw.set_title('Could not load')
            self.paramsLabel.setText('')
            self.canvas.draw()
            return

        mag = self._currentMag
        blockDiam, fixedThresh, dustCorrection, minColonyArea = self._getParamsForMag(mag)

        magParams = self.state.get('magParams', {})
        if mag and mag in magParams:
            self.paramsLabel.setText(
                f'Using per-mag overrides for {mag}: {magParams[mag]}'
            )
        else:
            self.paramsLabel.setText(
                f'Using global parameters'
                + (f' (mag {mag})' if mag else '')
            )

        rawScaled = raw.astype(np.float32)
        rmax = rawScaled.max()
        if rmax > 0:
            rawScaled /= rmax

        processed = normalizeLocalContrast(rawScaled, blockDiam)
        sigma = 2.0
        blurred = cv2.GaussianBlur(
            processed, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )

        maskLive = blurred > fixedThresh

        # invert exactly as the saved _processed.tif: normalize [min,max]→[0,1] then invert
        pmin, pmax = float(processed.min()), float(processed.max())
        if pmax > pmin:
            display = 1.0 - (processed - pmin) / (pmax - pmin)
        else:
            display = np.zeros_like(processed, dtype=np.float32)

        self.axRaw.imshow(raw, cmap='gray')
        self.axRaw.set_title('Raw')

        self.axProc.imshow(display, cmap='gray')
        self.axProc.set_title(
            f'Preprocessed\nblockDiam={blockDiam}',
            fontsize=9,
        )

        overlay = np.stack([display, display, display], axis=-1)
        overlay[maskLive] = [0, 1, 1]
        self.axMask.imshow(overlay)
        self.axMask.set_title(
            f'Mask Overlay\nthresh={fixedThresh}  dust={dustCorrection}',
            fontsize=9,
        )

        self.figure.tight_layout()
        self.canvas.draw()
