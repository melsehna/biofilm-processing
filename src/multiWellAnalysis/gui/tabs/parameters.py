import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QLineEdit, QComboBox,
    QPushButton, QListWidget,
)


def _maxWorkers():
    cpus = os.cpu_count() or 4
    return max(1, int(cpus * 0.75))


class _CollapsibleGroupBox(QGroupBox):
    """QGroupBox that hides its child widgets when its title checkbox is off.

    Default is collapsed. Click the title to expand/collapse. Useful for
    tucking away advanced/rarely-changed parameters so they don't clutter
    the main form but are still discoverable.
    """
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._onToggle)

    def setLayout(self, layout):
        super().setLayout(layout)
        self._setChildrenVisible(self.isChecked())

    def _onToggle(self, checked):
        self._setChildrenVisible(checked)

    def _setChildrenVisible(self, visible):
        layout = self.layout()
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget()
            if w:
                w.setVisible(visible)


class ParametersTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._buildUi()
        self._connectSignals()

    def _buildUi(self):
        layout = QVBoxLayout(self)

        analysisGroup = QGroupBox('Analysis')
        analysisForm = QFormLayout()

        self.doBiomass = QCheckBox('Biofilm biomass (preprocessing + registration + masking)')
        self.doBiomass.setChecked(True)
        self.doBiomass.setEnabled(False)  # always on — base pipeline
        analysisForm.addRow(self.doBiomass)

        self.saveOverlays = QCheckBox('Mask overlay videos (.mp4)')
        self.saveOverlays.setChecked(self.state.get('saveOverlays', True))
        analysisForm.addRow(self.saveOverlays)

        self.saveProcessedVideo = QCheckBox('Processed videos (.mp4, no mask overlay)')
        self.saveProcessedVideo.setChecked(self.state.get('saveProcessedVideo', False))
        analysisForm.addRow(self.saveProcessedVideo)

        self.saveFpHalf = QCheckBox('Also save fixed-fpMean=0.5 outputs (_fpHalf.tif/.mp4)')
        self.saveFpHalf.setChecked(self.state.get('saveFpHalf', False))
        self.saveFpHalf.setToolTip(
            'When checked, every processed .tif (and .mp4 if those are enabled) '
            'is also written with fixed fpMean=0.5 alongside the adaptive '
            'rendering, suffixed _fpHalf. Lets you compare cross-batch '
            'consistency. See ISSUES.md / JULIA_REFERENCE_COMPARISON.md '
            'in microTyper-Vision for motivation.'
        )
        analysisForm.addRow(self.saveFpHalf)

        self.wholeImage = QCheckBox('Whole-image texture features')
        self.wholeImage.setChecked(self.state.get('wholeImageFeats', False))
        analysisForm.addRow(self.wholeImage)

        self.colonyTracking = QCheckBox('Colony tracking')
        self.colonyTracking.setChecked(self.state.get('colonyTracking', False))
        analysisForm.addRow(self.colonyTracking)

        self.colonyFeats = QCheckBox('Colony-level feature extraction (requires tracking)')
        self.colonyFeats.setChecked(self.state.get('colonyFeats', False))
        analysisForm.addRow(self.colonyFeats)

        analysisGroup.setLayout(analysisForm)
        layout.addWidget(analysisGroup)

        preprocGroup = QGroupBox('Preprocessing Parameters')
        preprocForm = QFormLayout()

        self.blockDiam = QSpinBox()
        self.blockDiam.setRange(11, 501)
        self.blockDiam.setSingleStep(2)
        self.blockDiam.setValue(self.state.get('blockDiam', 101))
        preprocForm.addRow('Block diameter (odd):', self.blockDiam)

        self.fixedThresh = QDoubleSpinBox()
        self.fixedThresh.setRange(0.0, 1.0)
        self.fixedThresh.setDecimals(4)
        self.fixedThresh.setSingleStep(0.001)
        self.fixedThresh.setValue(self.state.get('fixedThresh', 0.04))
        preprocForm.addRow('Fixed threshold:', self.fixedThresh)

        self.dustCorrection = QCheckBox('Dust correction')
        self.dustCorrection.setChecked(self.state.get('dustCorrection', True))
        preprocForm.addRow(self.dustCorrection)

        preprocGroup.setLayout(preprocForm)
        layout.addWidget(preprocGroup)

        # ── Advanced (registration) — collapsed by default ──────────────────
        advGroup = _CollapsibleGroupBox('Advanced (registration)')
        advForm = QFormLayout()

        self.fftStride = QSpinBox()
        self.fftStride.setRange(1, 30)
        self.fftStride.setValue(self.state.get('fftStride', 6))
        self.fftStride.setToolTip(
            'Keyframe spacing for phase-correlation registration. 1 = register '
            'every frame (most accurate, slowest). Higher values speed up '
            'phase 1 but let sub-pixel drift accumulate between keyframes, '
            'which can cause downstream colony-label flips.'
        )
        advForm.addRow('FFT stride (keyframe step):', self.fftStride)

        self.downsample = QSpinBox()
        self.downsample.setRange(1, 16)
        self.downsample.setValue(self.state.get('downsample', 4))
        self.downsample.setToolTip(
            'Downsampling factor applied to each frame BEFORE the FFT phase '
            'correlation. 1 = full resolution (most precise, slowest). 4 '
            '(default) is a good trade-off; quadratic FFT cost means 1 is '
            '~16x slower than 4 per FFT call.'
        )
        advForm.addRow('FFT downsample factor:', self.downsample)

        self.shiftThresh = QSpinBox()
        self.shiftThresh.setRange(1, 1000)
        self.shiftThresh.setValue(self.state.get('shiftThresh', 50))
        self.shiftThresh.setToolTip(
            'Maximum per-step shift (in pixels) the registrar will accept '
            'from one FFT before rejecting it as a spurious peak. Raise if '
            'frames legitimately drift far between keyframes; lower if you '
            'have transient artifacts that fool the registrar.'
        )
        advForm.addRow('Shift threshold (px):', self.shiftThresh)

        advGroup.setLayout(advForm)
        layout.addWidget(advGroup)

        magGroup = QGroupBox('Per-Magnification Overrides')
        magLayout = QVBoxLayout()

        magHint = QLabel(
            'Save current preprocessing values as overrides for a specific magnification. '
            'Magnifications without overrides use the global values above.'
        )
        magHint.setWordWrap(True)
        magHint.setStyleSheet('color: gray; font-size: 11px;')
        magLayout.addWidget(magHint)

        magBtnRow = QHBoxLayout()
        self.magOverrideCombo = QComboBox()
        self.magOverrideCombo.setMinimumWidth(150)
        magBtnRow.addWidget(QLabel('Magnification:'))
        magBtnRow.addWidget(self.magOverrideCombo)

        saveOverrideBtn = QPushButton('Save override')
        saveOverrideBtn.clicked.connect(self._saveMagOverride)
        magBtnRow.addWidget(saveOverrideBtn)

        loadOverrideBtn = QPushButton('Load override')
        loadOverrideBtn.clicked.connect(self._loadMagOverride)
        magBtnRow.addWidget(loadOverrideBtn)

        delOverrideBtn = QPushButton('Delete')
        delOverrideBtn.clicked.connect(self._deleteMagOverride)
        magBtnRow.addWidget(delOverrideBtn)
        magBtnRow.addStretch()
        magLayout.addLayout(magBtnRow)

        self.magOverridesList = QListWidget()
        self.magOverridesList.setMaximumHeight(80)
        magLayout.addWidget(self.magOverridesList)

        magGroup.setLayout(magLayout)
        layout.addWidget(magGroup)

        self._refreshMagCombo()
        self._refreshMagOverridesList()
        self.state.changed.connect(self._onStateChangedMag)

        self.colonyParamsGroup = QGroupBox('Colony Tracking Parameters')
        colonyForm = QFormLayout()

        self.minColonyArea = QSpinBox()
        self.minColonyArea.setRange(10, 5000)
        self.minColonyArea.setValue(self.state.get('minColonyAreaPx', 200))
        colonyForm.addRow('Min colony area (px):', self.minColonyArea)

        self.propRadius = QSpinBox()
        self.propRadius.setRange(1, 99999)
        self.propRadius.setValue(self.state.get('propRadiusPx', 25))
        colonyForm.addRow('Propagation radius (px):', self.propRadius)

        self.colonyParamsGroup.setLayout(colonyForm)
        self.colonyParamsGroup.setVisible(
            self.state.get('colonyTracking', False)
            or self.state.get('colonyFeats', False)
        )
        layout.addWidget(self.colonyParamsGroup)

        umapGroup = QGroupBox('UMAP Generation')
        umapForm = QFormLayout()

        umapHint = QLabel(
            'Embeds wells from master_frame_features.csv with UMAP. '
            'Coloring uses a per-plate "*layout*.csv" sidecar if present, '
            'else falls back to the Conditions tab.'
        )
        umapHint.setWordWrap(True)
        umapHint.setStyleSheet('color: gray; font-size: 11px;')
        umapForm.addRow(umapHint)

        self.umapStatic = QCheckBox('Generate static UMAP (canonical PNG + 3x3 grid)')
        self.umapStatic.setChecked(self.state.get('umapStatic', False))
        umapForm.addRow(self.umapStatic)

        self.umapInteractive = QCheckBox('Generate interactive UMAP (HTML viewer)')
        self.umapInteractive.setChecked(self.state.get('umapInteractive', False))
        umapForm.addRow(self.umapInteractive)

        self.umapColumnName = QLineEdit()
        self.umapColumnName.setPlaceholderText('blank = first non-wellId column')
        self.umapColumnName.setText(self.state.get('umapColumnName', ''))
        umapForm.addRow('Color by column:', self.umapColumnName)

        umapGroup.setLayout(umapForm)
        layout.addWidget(umapGroup)

        perfGroup = QGroupBox('Performance')
        perfForm = QFormLayout()

        cap = _maxWorkers()
        self.workers = QSpinBox()
        self.workers.setRange(1, cap)
        self.workers.setValue(min(self.state.get('workers', 4), cap))
        perfForm.addRow('Workers:', self.workers)

        coresLabel = QLabel(f'(max {cap}, from {os.cpu_count()} cores)')
        coresLabel.setStyleSheet('color: gray; font-size: 11px;')
        perfForm.addRow('', coresLabel)

        perfGroup.setLayout(perfForm)
        layout.addWidget(perfGroup)

        outputGroup = QGroupBox('Saved Outputs (Advanced)')
        outputForm = QFormLayout()

        # NOTE: saveRegistered / saveProcessed / saveMasks are stored
        # in state and respected by the dependency-enforcement logic, but
        # post-run file cleanup is not yet implemented — the pipeline always
        # writes all outputs.  These checkboxes are placeholders for that feature.

        self.saveRegistered = QCheckBox('Keep registered raw stacks (.tif)')
        self.saveRegistered.setChecked(self.state.get('saveRegistered', True))
        outputForm.addRow(self.saveRegistered)

        self.saveProcessed = QCheckBox('Keep processed images (.tif)')
        self.saveProcessed.setChecked(self.state.get('saveProcessed', True))
        outputForm.addRow(self.saveProcessed)

        self.saveMasks = QCheckBox('Keep binary masks (.npz)')
        self.saveMasks.setChecked(self.state.get('saveMasks', True))
        outputForm.addRow(self.saveMasks)


        outputGroup.setLayout(outputForm)
        layout.addWidget(outputGroup)

        layout.addStretch()

    def _connectSignals(self):
        self.saveOverlays.toggled.connect(
            lambda v: self.state.set('saveOverlays', v))
        self.wholeImage.toggled.connect(self._onWholeImage)
        self.colonyTracking.toggled.connect(self._onColonyTracking)
        self.colonyFeats.toggled.connect(self._onColonyFeats)

        self.blockDiam.valueChanged.connect(self._onBlockDiam)
        self.fixedThresh.valueChanged.connect(
            lambda v: self.state.set('fixedThresh', v))
        self.dustCorrection.toggled.connect(
            lambda v: self.state.set('dustCorrection', v))
        self.fftStride.valueChanged.connect(
            lambda v: self.state.set('fftStride', v))
        self.downsample.valueChanged.connect(
            lambda v: self.state.set('downsample', v))
        self.shiftThresh.valueChanged.connect(
            lambda v: self.state.set('shiftThresh', v))

        self.minColonyArea.valueChanged.connect(
            lambda v: self.state.set('minColonyAreaPx', v))
        self.propRadius.valueChanged.connect(
            lambda v: self.state.set('propRadiusPx', v))

        self.umapStatic.toggled.connect(
            lambda v: self.state.set('umapStatic', v))
        self.umapInteractive.toggled.connect(
            lambda v: self.state.set('umapInteractive', v))
        self.umapColumnName.editingFinished.connect(
            lambda: self.state.set('umapColumnName', self.umapColumnName.text().strip()))

        self.workers.valueChanged.connect(
            lambda v: self.state.set('workers', v))

        self.saveRegistered.toggled.connect(
            lambda v: self.state.set('saveRegistered', v))
        self.saveProcessed.toggled.connect(
            lambda v: self.state.set('saveProcessed', v))
        self.saveMasks.toggled.connect(
            lambda v: self.state.set('saveMasks', v))
        self.saveProcessedVideo.toggled.connect(
            lambda v: self.state.set('saveProcessedVideo', v))
        self.saveFpHalf.toggled.connect(
            lambda v: self.state.set('saveFpHalf', v))

        self.saveProcessed.toggled.connect(self._enforceOutputDeps)
        self.saveRegistered.toggled.connect(self._enforceOutputDeps)

    def _onBlockDiam(self, val):
        if val % 2 == 0:
            self.blockDiam.setValue(val + 1)
            return
        self.state.set('blockDiam', val)

    def _onWholeImage(self, checked):
        self.state.set('wholeImageFeats', checked)
        if checked and not self.saveProcessed.isChecked():
            self.saveProcessed.setChecked(True)

    def _onColonyTracking(self, checked):
        self.state.set('colonyTracking', checked)
        if not checked and self.colonyFeats.isChecked():
            self.colonyTracking.setChecked(True)
            return
        if checked:
            if not self.saveRegistered.isChecked():
                self.saveRegistered.setChecked(True)
            if not self.saveMasks.isChecked():
                self.saveMasks.setChecked(True)
        self.colonyParamsGroup.setVisible(
            checked or self.colonyFeats.isChecked()
        )

    def _onColonyFeats(self, checked):
        self.state.set('colonyFeats', checked)
        if checked:
            if not self.colonyTracking.isChecked():
                self.colonyTracking.setChecked(True)
            if not self.saveRegistered.isChecked():
                self.saveRegistered.setChecked(True)
            if not self.saveMasks.isChecked():
                self.saveMasks.setChecked(True)
        self.colonyParamsGroup.setVisible(
            checked or self.colonyTracking.isChecked()
        )

    def _enforceOutputDeps(self):
        """Prevent unchecking outputs that active features depend on."""
        if self.wholeImage.isChecked() and not self.saveProcessed.isChecked():
            self.saveProcessed.setChecked(True)
        if self.colonyFeats.isChecked() or self.colonyTracking.isChecked():
            if not self.saveRegistered.isChecked():
                self.saveRegistered.setChecked(True)
            if not self.saveMasks.isChecked():
                self.saveMasks.setChecked(True)
        if self.colonyFeats.isChecked() and not self.colonyTracking.isChecked():
            self.colonyTracking.setChecked(True)

    def _onStateChangedMag(self):
        """Refresh mag combo when magnifications change in Setup tab."""
        self._refreshMagCombo()

    def _refreshMagCombo(self):
        magSetting = self.state.get('magnification', 'all')
        mags = []
        if isinstance(magSetting, list):
            mags = magSetting
        elif isinstance(magSetting, str) and magSetting != 'all':
            mags = [magSetting]

        for m in self.state.get('magParams', {}):
            if m not in mags:
                mags.append(m)

        # Build "4x (_02)" style labels from plateMeta; when plates disagree
        # on a suffix, show "4x/10x (_02)". The combo's stored data is still
        # the bare suffix — the label is display-only.
        plateMeta = self.state.get('plateMeta', {})
        suffixObjs = {}
        for meta in plateMeta.values():
            for suf, m in meta.items():
                obj = m.get('objective')
                if obj is not None:
                    suffixObjs.setdefault(suf, set()).add(obj)

        prev = self.magOverrideCombo.currentData()
        self.magOverrideCombo.blockSignals(True)
        self.magOverrideCombo.clear()
        for m in sorted(set(mags)):
            objs = suffixObjs.get(m)
            if objs:
                objLabel = '/'.join(f'{o}x' for o in sorted(objs))
                label = f'{objLabel} ({m})'
            else:
                label = m
            self.magOverrideCombo.addItem(label, m)
        idx = self.magOverrideCombo.findData(prev)
        if idx >= 0:
            self.magOverrideCombo.setCurrentIndex(idx)
        self.magOverrideCombo.blockSignals(False)

    def _refreshMagOverridesList(self):
        self.magOverridesList.clear()
        magParams = self.state.get('magParams', {})
        for mag, params in sorted(magParams.items()):
            parts = [f'{k}={v}' for k, v in sorted(params.items())]
            self.magOverridesList.addItem(f'{mag}: {", ".join(parts)}')

    def _saveMagOverride(self):
        mag = self.magOverrideCombo.currentData()
        if not mag:
            return
        magParams = self.state.get('magParams', {})
        magParams[mag] = {
            'blockDiam': self.blockDiam.value(),
            'fixedThresh': self.fixedThresh.value(),
            'dustCorrection': self.dustCorrection.isChecked(),
            'minColonyAreaPx': self.minColonyArea.value(),
            'propRadiusPx': self.propRadius.value(),
        }
        self.state.set('magParams', magParams)
        self._refreshMagOverridesList()

    def _loadMagOverride(self):
        """Load a saved override's values into the parameter widgets for editing."""
        mag = self.magOverrideCombo.currentData()
        if not mag:
            return
        magParams = self.state.get('magParams', {})
        if mag not in magParams:
            return
        p = magParams[mag]
        for w in [self.blockDiam, self.fixedThresh, self.dustCorrection,
                  self.minColonyArea, self.propRadius]:
            w.blockSignals(True)
        self.blockDiam.setValue(p.get('blockDiam', self.state.get('blockDiam', 101)))
        self.fixedThresh.setValue(p.get('fixedThresh', self.state.get('fixedThresh', 0.04)))
        self.dustCorrection.setChecked(p.get('dustCorrection', self.state.get('dustCorrection', True)))
        self.minColonyArea.setValue(p.get('minColonyAreaPx', self.state.get('minColonyAreaPx', 200)))
        self.propRadius.setValue(p.get('propRadiusPx', self.state.get('propRadiusPx', 25)))
        for w in [self.blockDiam, self.fixedThresh, self.dustCorrection,
                  self.minColonyArea, self.propRadius]:
            w.blockSignals(False)

    def _deleteMagOverride(self):
        mag = self.magOverrideCombo.currentData()
        if not mag:
            return
        magParams = self.state.get('magParams', {})
        magParams.pop(mag, None)
        self.state.set('magParams', magParams)
        self._refreshMagOverridesList()

    def refreshFromState(self):
        """Sync all widgets to current state (call after loading a config)."""
        widgets = [
            self.saveOverlays, self.wholeImage, self.colonyTracking,
            self.colonyFeats, self.dustCorrection, self.saveRegistered,
            self.saveProcessed, self.saveMasks, self.saveProcessedVideo, self.saveFpHalf,
            self.blockDiam, self.fixedThresh,
            self.fftStride, self.downsample, self.shiftThresh,
            self.minColonyArea, self.propRadius, self.workers,
            self.umapStatic, self.umapInteractive, self.umapColumnName,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.saveOverlays.setChecked(self.state.get('saveOverlays', True))
        self.wholeImage.setChecked(self.state.get('wholeImageFeats', False))
        self.colonyTracking.setChecked(self.state.get('colonyTracking', False))
        self.colonyFeats.setChecked(self.state.get('colonyFeats', False))
        self.dustCorrection.setChecked(self.state.get('dustCorrection', True))
        self.saveRegistered.setChecked(self.state.get('saveRegistered', True))
        self.saveProcessed.setChecked(self.state.get('saveProcessed', True))
        self.saveMasks.setChecked(self.state.get('saveMasks', True))
        self.saveProcessedVideo.setChecked(self.state.get('saveProcessedVideo', False))
        self.saveFpHalf.setChecked(self.state.get('saveFpHalf', False))
        self.blockDiam.setValue(self.state.get('blockDiam', 101))
        self.fixedThresh.setValue(self.state.get('fixedThresh', 0.04))
        self.fftStride.setValue(self.state.get('fftStride', 6))
        self.downsample.setValue(self.state.get('downsample', 4))
        self.shiftThresh.setValue(self.state.get('shiftThresh', 50))
        self.minColonyArea.setValue(self.state.get('minColonyAreaPx', 200))
        self.propRadius.setValue(self.state.get('propRadiusPx', 25))
        self.workers.setValue(min(self.state.get('workers', 4), _maxWorkers()))
        self.umapStatic.setChecked(self.state.get('umapStatic', False))
        self.umapInteractive.setChecked(self.state.get('umapInteractive', False))
        self.umapColumnName.setText(self.state.get('umapColumnName', ''))

        for w in widgets:
            w.blockSignals(False)

        self.colonyParamsGroup.setVisible(
            self.state.get('colonyTracking', False) or self.state.get('colonyFeats', False)
        )
        self._refreshMagCombo()
        self._refreshMagOverridesList()
