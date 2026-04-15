import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QComboBox,
    QPushButton, QListWidget,
)


def _maxWorkers():
    cpus = os.cpu_count() or 4
    return max(1, int(cpus * 0.75))


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

        # NOTE: saveRegistered / saveProcessed / saveMasks / copyRaw are stored
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

        self.copyRaw = QCheckBox('Copy raw frames as stacked TIFF (.tif)')
        self.copyRaw.setChecked(self.state.get('copyRaw', False))
        outputForm.addRow(self.copyRaw)

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

        self.minColonyArea.valueChanged.connect(
            lambda v: self.state.set('minColonyAreaPx', v))
        self.propRadius.valueChanged.connect(
            lambda v: self.state.set('propRadiusPx', v))

        self.workers.valueChanged.connect(
            lambda v: self.state.set('workers', v))

        self.saveRegistered.toggled.connect(
            lambda v: self.state.set('saveRegistered', v))
        self.saveProcessed.toggled.connect(
            lambda v: self.state.set('saveProcessed', v))
        self.saveMasks.toggled.connect(
            lambda v: self.state.set('saveMasks', v))
        self.copyRaw.toggled.connect(
            lambda v: self.state.set('copyRaw', v))

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

        prev = self.magOverrideCombo.currentText()
        self.magOverrideCombo.blockSignals(True)
        self.magOverrideCombo.clear()
        for m in sorted(set(mags)):
            self.magOverrideCombo.addItem(m)
        idx = self.magOverrideCombo.findText(prev)
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
        mag = self.magOverrideCombo.currentText()
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
        mag = self.magOverrideCombo.currentText()
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
        mag = self.magOverrideCombo.currentText()
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
            self.saveProcessed, self.saveMasks, self.copyRaw,
            self.blockDiam, self.fixedThresh,
            self.minColonyArea, self.propRadius, self.workers,
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
        self.copyRaw.setChecked(self.state.get('copyRaw', False))
        self.blockDiam.setValue(self.state.get('blockDiam', 101))
        self.fixedThresh.setValue(self.state.get('fixedThresh', 0.04))
        self.minColonyArea.setValue(self.state.get('minColonyAreaPx', 200))
        self.propRadius.setValue(self.state.get('propRadiusPx', 25))
        self.workers.setValue(min(self.state.get('workers', 4), _maxWorkers()))

        for w in widgets:
            w.blockSignals(False)

        self.colonyParamsGroup.setVisible(
            self.state.get('colonyTracking', False) or self.state.get('colonyFeats', False)
        )
        self._refreshMagCombo()
        self._refreshMagOverridesList()
