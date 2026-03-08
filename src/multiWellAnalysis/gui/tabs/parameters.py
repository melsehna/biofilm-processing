from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox,
)


class ParametersTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Preprocessing ──────────────────────────────────────────
        preproc_group = QGroupBox('Preprocessing')
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
        preproc_form.addRow('Shift threshold:', self.shift_thresh)

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

        # ── Outputs to Save ────────────────────────────────────────
        output_group = QGroupBox('Outputs to Save')
        output_form = QFormLayout()

        self.save_registered = QCheckBox('Registered raw stacks (.tif)')
        self.save_registered.setChecked(self.state.get('saveRegistered', True))
        output_form.addRow(self.save_registered)

        self.save_processed = QCheckBox('Processed images (.tif)')
        self.save_processed.setChecked(self.state.get('saveProcessed', True))
        output_form.addRow(self.save_processed)

        self.save_masks = QCheckBox('Binary masks (.npz)')
        self.save_masks.setChecked(self.state.get('saveMasks', True))
        output_form.addRow(self.save_masks)

        self.save_overlays = QCheckBox('Mask overlay videos')
        self.save_overlays.setChecked(self.state.get('saveOverlays', True))
        output_form.addRow(self.save_overlays)

        output_group.setLayout(output_form)
        layout.addWidget(output_group)

        # ── Feature Extraction ─────────────────────────────────────
        feat_group = QGroupBox('Feature Extraction')
        feat_form = QFormLayout()

        self.whole_image = QCheckBox('Whole-image texture features')
        self.whole_image.setChecked(self.state.get('wholeImageFeats', False))
        feat_form.addRow(self.whole_image)

        self.colony_tracking = QCheckBox('Colony tracking (+ save tracked labels)')
        self.colony_tracking.setChecked(self.state.get('colonyTracking', False))
        feat_form.addRow(self.colony_tracking)

        self.colony_feats = QCheckBox('Colony-level feature extraction')
        self.colony_feats.setChecked(self.state.get('colonyFeats', False))
        feat_form.addRow(self.colony_feats)

        feat_group.setLayout(feat_form)
        layout.addWidget(feat_group)

        # ── Colony Tracking Parameters ─────────────────────────────
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

        layout.addStretch()

    def _connect_signals(self):
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

        # outputs
        self.save_registered.toggled.connect(
            lambda v: self.state.set('saveRegistered', v))
        self.save_processed.toggled.connect(
            lambda v: self.state.set('saveProcessed', v))
        self.save_masks.toggled.connect(
            lambda v: self.state.set('saveMasks', v))
        self.save_overlays.toggled.connect(
            lambda v: self.state.set('saveOverlays', v))

        # features
        self.whole_image.toggled.connect(
            lambda v: self.state.set('wholeImageFeats', v))
        self.colony_tracking.toggled.connect(self._on_colony_tracking)
        self.colony_feats.toggled.connect(self._on_colony_feats)

        # colony params
        self.min_colony_area.valueChanged.connect(
            lambda v: self.state.set('minColonyAreaPx', v))
        self.prop_radius.valueChanged.connect(
            lambda v: self.state.set('propRadiusPx', v))

    def _on_block_diam(self, val):
        if val % 2 == 0:
            self.block_diam.setValue(val + 1)
            return
        self.state.set('blockDiam', val)

    def _on_colony_tracking(self, checked):
        self.state.set('colonyTracking', checked)
        self.colony_params_group.setVisible(
            checked or self.colony_feats.isChecked()
        )

    def _on_colony_feats(self, checked):
        self.state.set('colonyFeats', checked)
        # colony feats requires tracking
        if checked and not self.colony_tracking.isChecked():
            self.colony_tracking.setChecked(True)
        self.colony_params_group.setVisible(
            checked or self.colony_tracking.isChecked()
        )
