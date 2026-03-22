import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QComboBox,
    QPushButton, QListWidget,
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

        self.dust_correction = QCheckBox('Dust correction')
        self.dust_correction.setChecked(self.state.get('dustCorrection', True))
        preproc_form.addRow(self.dust_correction)

        preproc_group.setLayout(preproc_form)
        layout.addWidget(preproc_group)

        # ── Per-Magnification Overrides ──────────────────────────
        mag_group = QGroupBox('Per-Magnification Overrides')
        mag_layout = QVBoxLayout()

        mag_hint = QLabel(
            'Save current preprocessing values as overrides for a specific magnification. '
            'Magnifications without overrides use the global values above.'
        )
        mag_hint.setWordWrap(True)
        mag_hint.setStyleSheet('color: gray; font-size: 11px;')
        mag_layout.addWidget(mag_hint)

        mag_btn_row = QHBoxLayout()
        self.mag_override_combo = QComboBox()
        self.mag_override_combo.setMinimumWidth(150)
        mag_btn_row.addWidget(QLabel('Magnification:'))
        mag_btn_row.addWidget(self.mag_override_combo)

        save_override_btn = QPushButton('Save override')
        save_override_btn.clicked.connect(self._save_mag_override)
        mag_btn_row.addWidget(save_override_btn)

        load_override_btn = QPushButton('Load override')
        load_override_btn.clicked.connect(self._load_mag_override)
        mag_btn_row.addWidget(load_override_btn)

        del_override_btn = QPushButton('Delete')
        del_override_btn.clicked.connect(self._delete_mag_override)
        mag_btn_row.addWidget(del_override_btn)
        mag_btn_row.addStretch()
        mag_layout.addLayout(mag_btn_row)

        self.mag_overrides_list = QListWidget()
        self.mag_overrides_list.setMaximumHeight(80)
        mag_layout.addWidget(self.mag_overrides_list)

        mag_group.setLayout(mag_layout)
        layout.addWidget(mag_group)

        # populate mag combo from state
        self._refresh_mag_combo()
        self._refresh_mag_overrides_list()
        self.state.changed.connect(self._on_state_changed_mag)

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
        # If unchecking tracking while colony feats is on, re-enable
        if not checked and self.colony_feats.isChecked():
            self.colony_tracking.setChecked(True)
            return
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
        if checked:
            # Colony feats requires tracking, which requires registered + masks
            if not self.colony_tracking.isChecked():
                self.colony_tracking.setChecked(True)
            if not self.save_registered.isChecked():
                self.save_registered.setChecked(True)
            if not self.save_masks.isChecked():
                self.save_masks.setChecked(True)
        self.colony_params_group.setVisible(
            checked or self.colony_tracking.isChecked()
        )

    def _enforce_output_deps(self):
        """Prevent unchecking outputs that active features depend on."""
        if self.whole_image.isChecked() and not self.save_processed.isChecked():
            self.save_processed.setChecked(True)
        if self.colony_feats.isChecked() or self.colony_tracking.isChecked():
            if not self.save_registered.isChecked():
                self.save_registered.setChecked(True)
            if not self.save_masks.isChecked():
                self.save_masks.setChecked(True)
        if self.colony_feats.isChecked() and not self.colony_tracking.isChecked():
            self.colony_tracking.setChecked(True)

    # ── Per-magnification override methods ───────────────────
    def _on_state_changed_mag(self):
        """Refresh mag combo when magnifications change in Setup tab."""
        self._refresh_mag_combo()

    def _refresh_mag_combo(self):
        mag_setting = self.state.get('magnification', 'all')
        mags = []
        if isinstance(mag_setting, list):
            mags = mag_setting
        elif isinstance(mag_setting, str) and mag_setting != 'all':
            mags = [mag_setting]

        # Also check magParams for mags that were saved but may not be selected
        for m in self.state.get('magParams', {}):
            if m not in mags:
                mags.append(m)

        prev = self.mag_override_combo.currentText()
        self.mag_override_combo.blockSignals(True)
        self.mag_override_combo.clear()
        for m in sorted(set(mags)):
            self.mag_override_combo.addItem(m)
        # restore
        idx = self.mag_override_combo.findText(prev)
        if idx >= 0:
            self.mag_override_combo.setCurrentIndex(idx)
        self.mag_override_combo.blockSignals(False)

    def _refresh_mag_overrides_list(self):
        self.mag_overrides_list.clear()
        mag_params = self.state.get('magParams', {})
        for mag, params in sorted(mag_params.items()):
            parts = [f'{k}={v}' for k, v in sorted(params.items())]
            self.mag_overrides_list.addItem(f'{mag}: {", ".join(parts)}')

    def _save_mag_override(self):
        mag = self.mag_override_combo.currentText()
        if not mag:
            return
        mag_params = self.state.get('magParams', {})
        mag_params[mag] = {
            'blockDiam': self.block_diam.value(),
            'fixedThresh': self.fixed_thresh.value(),
            'dustCorrection': self.dust_correction.isChecked(),
            'minColonyAreaPx': self.min_colony_area.value(),
            'propRadiusPx': self.prop_radius.value(),
        }
        self.state.set('magParams', mag_params)
        self._refresh_mag_overrides_list()

    def _load_mag_override(self):
        """Load a saved override's values into the parameter widgets for editing."""
        mag = self.mag_override_combo.currentText()
        if not mag:
            return
        mag_params = self.state.get('magParams', {})
        if mag not in mag_params:
            return
        p = mag_params[mag]
        self.block_diam.blockSignals(True)
        self.fixed_thresh.blockSignals(True)
        self.dust_correction.blockSignals(True)
        self.min_colony_area.blockSignals(True)
        self.prop_radius.blockSignals(True)
        self.block_diam.setValue(p.get('blockDiam', self.state.get('blockDiam', 101)))
        self.fixed_thresh.setValue(p.get('fixedThresh', self.state.get('fixedThresh', 0.04)))
        self.dust_correction.setChecked(p.get('dustCorrection', self.state.get('dustCorrection', True)))
        self.min_colony_area.setValue(p.get('minColonyAreaPx', self.state.get('minColonyAreaPx', 200)))
        self.prop_radius.setValue(p.get('propRadiusPx', self.state.get('propRadiusPx', 25)))
        self.block_diam.blockSignals(False)
        self.fixed_thresh.blockSignals(False)
        self.dust_correction.blockSignals(False)
        self.min_colony_area.blockSignals(False)
        self.prop_radius.blockSignals(False)

    def _delete_mag_override(self):
        mag = self.mag_override_combo.currentText()
        if not mag:
            return
        mag_params = self.state.get('magParams', {})
        mag_params.pop(mag, None)
        self.state.set('magParams', mag_params)
        self._refresh_mag_overrides_list()
