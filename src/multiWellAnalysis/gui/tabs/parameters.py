import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel,
)


def _max_workers():
    cpus = os.cpu_count() or 4
    return max(1, int(cpus * 0.75))


class ParametersTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Analysis ─────────────────────────────────────────────
        analysis_group = QGroupBox('Analysis')
        analysis_form = QFormLayout()

        self.do_biomass = QCheckBox('Biofilm biomass (preprocessing + registration + masking)')
        self.do_biomass.setChecked(True)
        self.do_biomass.setEnabled(False)  # always on — base pipeline
        analysis_form.addRow(self.do_biomass)

        self.save_overlays = QCheckBox('Mask overlay videos (.mp4)')
        self.save_overlays.setChecked(self.state.get('saveOverlays', True))
        analysis_form.addRow(self.save_overlays)

        self.whole_image = QCheckBox('Whole-image texture features')
        self.whole_image.setChecked(self.state.get('wholeImageFeats', False))
        analysis_form.addRow(self.whole_image)

        self.colony_tracking = QCheckBox('Colony tracking')
        self.colony_tracking.setChecked(self.state.get('colonyTracking', False))
        analysis_form.addRow(self.colony_tracking)

        self.colony_feats = QCheckBox('Colony-level feature extraction (requires tracking)')
        self.colony_feats.setChecked(self.state.get('colonyFeats', False))
        analysis_form.addRow(self.colony_feats)

        analysis_group.setLayout(analysis_form)
        layout.addWidget(analysis_group)

        # ── Preprocessing Parameters ────────────────────────────
        preproc_group = QGroupBox('Preprocessing Parameters')
        preproc_form = QFormLayout()

        self.block_diam = QSpinBox()
        self.block_diam.setRange(11, 501)
        self.block_diam.setSingleStep(2)
        self.block_diam.setValue(self.state.get('blockDiam', 101))
        preproc_form.addRow('Block diameter (odd):', self.block_diam)

        self.fixed_thresh = QDoubleSpinBox()
        self.fixed_thresh.setRange(0.0, 1.0)
        self.fixed_thresh.setDecimals(4)
        self.fixed_thresh.setSingleStep(0.001)
        self.fixed_thresh.setValue(self.state.get('fixedThresh', 0.04))
        preproc_form.addRow('Fixed threshold:', self.fixed_thresh)

        self.shift_thresh = QSpinBox()
        self.shift_thresh.setRange(1, 500)
        self.shift_thresh.setValue(self.state.get('shiftThresh', 50))
        preproc_form.addRow('Shift threshold (px):', self.shift_thresh)

        self.fft_stride = QSpinBox()
        self.fft_stride.setRange(1, 20)
        self.fft_stride.setValue(self.state.get('fftStride', 6))
        preproc_form.addRow('FFT stride:', self.fft_stride)

        self.downsample = QSpinBox()
        self.downsample.setRange(1, 16)
        self.downsample.setValue(self.state.get('downsample', 4))
        preproc_form.addRow('Downsample factor:', self.downsample)

        self.dust_correction = QCheckBox('Dust correction')
        self.dust_correction.setChecked(self.state.get('dustCorrection', True))
        preproc_form.addRow(self.dust_correction)

        preproc_group.setLayout(preproc_form)
        layout.addWidget(preproc_group)

        # ── Colony Tracking Parameters ───────────────────────────
        self.colony_params_group = QGroupBox('Colony Tracking Parameters')
        colony_form = QFormLayout()

        self.min_colony_area = QSpinBox()
        self.min_colony_area.setRange(10, 5000)
        self.min_colony_area.setValue(self.state.get('minColonyAreaPx', 200))
        colony_form.addRow('Min colony area (px):', self.min_colony_area)

        self.prop_radius = QSpinBox()
        self.prop_radius.setRange(1, 200)
        self.prop_radius.setValue(self.state.get('propRadiusPx', 25))
        colony_form.addRow('Propagation radius (px):', self.prop_radius)

        self.colony_params_group.setLayout(colony_form)
        self.colony_params_group.setVisible(
            self.state.get('colonyTracking', False)
            or self.state.get('colonyFeats', False)
        )
        layout.addWidget(self.colony_params_group)

        # ── Performance ──────────────────────────────────────────
        perf_group = QGroupBox('Performance')
        perf_form = QFormLayout()

        cap = _max_workers()
        self.workers = QSpinBox()
        self.workers.setRange(1, cap)
        self.workers.setValue(min(self.state.get('workers', 4), cap))
        perf_form.addRow('Workers:', self.workers)

        cores_label = QLabel(f'(max {cap}, from {os.cpu_count()} cores)')
        cores_label.setStyleSheet('color: gray; font-size: 11px;')
        perf_form.addRow('', cores_label)

        perf_group.setLayout(perf_form)
        layout.addWidget(perf_group)

        # ── Saved Outputs (Advanced) ─────────────────────────────
        output_group = QGroupBox('Saved Outputs (Advanced)')
        output_form = QFormLayout()

        self.save_registered = QCheckBox('Keep registered raw stacks (.tif)')
        self.save_registered.setChecked(self.state.get('saveRegistered', True))
        output_form.addRow(self.save_registered)

        self.save_processed = QCheckBox('Keep processed images (.tif)')
        self.save_processed.setChecked(self.state.get('saveProcessed', True))
        output_form.addRow(self.save_processed)

        self.save_masks = QCheckBox('Keep binary masks (.npz)')
        self.save_masks.setChecked(self.state.get('saveMasks', True))
        output_form.addRow(self.save_masks)

        output_group.setLayout(output_form)
        layout.addWidget(output_group)

        layout.addStretch()

    def _connect_signals(self):
        # analysis
        self.save_overlays.toggled.connect(
            lambda v: self.state.set('saveOverlays', v))
        self.whole_image.toggled.connect(self._on_whole_image)
        self.colony_tracking.toggled.connect(self._on_colony_tracking)
        self.colony_feats.toggled.connect(self._on_colony_feats)

        # preprocessing
        self.block_diam.valueChanged.connect(self._on_block_diam)
        self.fixed_thresh.valueChanged.connect(
            lambda v: self.state.set('fixedThresh', v))
        self.shift_thresh.valueChanged.connect(
            lambda v: self.state.set('shiftThresh', v))
        self.fft_stride.valueChanged.connect(
            lambda v: self.state.set('fftStride', v))
        self.downsample.valueChanged.connect(
            lambda v: self.state.set('downsample', v))
        self.dust_correction.toggled.connect(
            lambda v: self.state.set('dustCorrection', v))

        # colony params
        self.min_colony_area.valueChanged.connect(
            lambda v: self.state.set('minColonyAreaPx', v))
        self.prop_radius.valueChanged.connect(
            lambda v: self.state.set('propRadiusPx', v))

        # performance
        self.workers.valueChanged.connect(
            lambda v: self.state.set('workers', v))

        # saved outputs
        self.save_registered.toggled.connect(
            lambda v: self.state.set('saveRegistered', v))
        self.save_processed.toggled.connect(
            lambda v: self.state.set('saveProcessed', v))
        self.save_masks.toggled.connect(
            lambda v: self.state.set('saveMasks', v))

        # re-check deps when user unchecks saved outputs
        self.save_processed.toggled.connect(self._enforce_output_deps)
        self.save_registered.toggled.connect(self._enforce_output_deps)

    def _on_block_diam(self, val):
        if val % 2 == 0:
            self.block_diam.setValue(val + 1)
            return
        self.state.set('blockDiam', val)

    def _on_whole_image(self, checked):
        self.state.set('wholeImageFeats', checked)
        if checked and not self.save_processed.isChecked():
            self.save_processed.setChecked(True)

    def _on_colony_tracking(self, checked):
        self.state.set('colonyTracking', checked)
        if checked:
            if not self.save_registered.isChecked():
                self.save_registered.setChecked(True)
            if not self.save_masks.isChecked():
                self.save_masks.setChecked(True)
        self.colony_params_group.setVisible(
            checked or self.colony_feats.isChecked()
        )

    def _on_colony_feats(self, checked):
        self.state.set('colonyFeats', checked)
        if checked and not self.colony_tracking.isChecked():
            self.colony_tracking.setChecked(True)
        self.colony_params_group.setVisible(
            checked or self.colony_tracking.isChecked()
        )

    def _enforce_output_deps(self):
        """Prevent unchecking outputs that active features depend on."""
        if self.whole_image.isChecked() and not self.save_processed.isChecked():
            self.save_processed.setChecked(True)
        if (self.colony_tracking.isChecked() or self.colony_feats.isChecked()):
            if not self.save_registered.isChecked():
                self.save_registered.setChecked(True)
            if not self.save_masks.isChecked():
                self.save_masks.setChecked(True)
