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

from multiWellAnalysis.processing.preprocessing import normalize_local_contrast

MAG_SUFFIXES = {'_02': '4x', '_03': '10x', '_04': '20x', '_05': '40x'}


def discover_wells_with_mag(plate_dir):
    """Find well+mag combinations from TIF filenames.

    Returns list of (display_label, well_id, mag_suffix, file_list_or_path) tuples.
    For plates without magnification suffixes, mag_suffix is ''.
    """
    if not plate_dir or not os.path.isdir(plate_dir):
        return []

    from multiWellAnalysis.gui.tabs.run import _resolve_tif_dir, _list_raw_tifs
    resolved = _resolve_tif_dir(plate_dir, max_depth=2)
    raw_tifs = _list_raw_tifs(resolved)

    bf_files = [f for f in raw_tifs if 'Bright Field' in f or 'Bright_Field' in f]

    # Try magnification-aware grouping
    candidates = bf_files if bf_files else raw_tifs
    if candidates:
        groups = defaultdict(lambda: defaultdict(list))
        for f in candidates:
            base = os.path.basename(f)
            m = re.match(r'^([A-P]\d+)(_\d+)_', base)
            if m:
                well, mag = m.group(1), m.group(2)
                groups[mag][well].append(f)

        if groups:
            result = []
            for mag in sorted(groups):
                for well in sorted(groups[mag]):
                    files = sorted(groups[mag][well])
                    mag_label = MAG_SUFFIXES.get(mag, mag)
                    label = f'{well} ({mag_label})'
                    result.append((label, well, mag, files))
            return result

    # Fallback: no magnification suffixes — group raw TIFs by well
    wells = defaultdict(list)
    for f in raw_tifs:
        name = os.path.basename(f)
        m = re.match(r'^([A-P]\d{1,2})[_.]', name)
        if m:
            wells[m.group(1)].append(f)

    result = []
    for well in sorted(wells):
        result.append((well, well, '', sorted(wells[well])))
    return result


def load_frame(source, frame_idx):
    """Load a single frame from a TIFF stack or list of files."""
    if isinstance(source, str):
        img = tifffile.imread(source)
        if img.ndim == 3:
            if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                n_frames = img.shape[0]
                frame_idx = min(frame_idx, n_frames - 1)
                return img[frame_idx].astype(np.float64), n_frames
            elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
                n_frames = img.shape[2]
                frame_idx = min(frame_idx, n_frames - 1)
                return img[:, :, frame_idx].astype(np.float64), n_frames
            else:
                n_frames = img.shape[0]
                frame_idx = min(frame_idx, n_frames - 1)
                return img[frame_idx].astype(np.float64), n_frames
        return img.astype(np.float64), 1
    elif isinstance(source, list):
        frame_idx = min(frame_idx, len(source) - 1)
        img = tifffile.imread(source[frame_idx])
        return img.astype(np.float64), len(source)
    return None, 0


class PreviewTab(QWidget):
    _well_result = Signal(object)  # delivers well entries from background thread

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._well_result.connect(self._on_wells_discovered)
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._render)
        self._current_source = None
        self._current_mag = ''
        self._current_well = ''
        self._n_frames = 0
        self._well_entries = []  # list of (label, well_id, mag, source)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # selectors row
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel('Plate:'))
        self.plate_combo = QComboBox()
        sel_row.addWidget(self.plate_combo, stretch=1)

        sel_row.addWidget(QLabel('Mag:'))
        self.mag_combo = QComboBox()
        sel_row.addWidget(self.mag_combo)

        sel_row.addWidget(QLabel('Well:'))
        self.well_combo = QComboBox()
        sel_row.addWidget(self.well_combo, stretch=1)
        layout.addLayout(sel_row)

        # frame slider row
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel('Frame:'))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        frame_row.addWidget(self.frame_slider, stretch=1)
        self.frame_label = QLabel('0 / 0')
        frame_row.addWidget(self.frame_label)
        layout.addLayout(frame_row)

        # params display
        self.params_label = QLabel('')
        self.params_label.setStyleSheet('color: gray; font-size: 11px;')
        layout.addWidget(self.params_label)

        # matplotlib canvas: 1 row x 3 columns
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.ax_raw = self.figure.add_subplot(1, 3, 1)
        self.ax_proc = self.figure.add_subplot(1, 3, 2)
        self.ax_mask = self.figure.add_subplot(1, 3, 3)

        for ax in self.figure.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        self.figure.tight_layout()
        layout.addWidget(self.canvas, stretch=1)

        self.refresh_btn = QPushButton('Refresh')
        layout.addWidget(self.refresh_btn)

    def _connect_signals(self):
        self.plate_combo.currentIndexChanged.connect(self._on_plate_changed)
        self.mag_combo.currentIndexChanged.connect(self._on_mag_changed)
        self.well_combo.currentIndexChanged.connect(self._on_well_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.refresh_btn.clicked.connect(self._refresh_all)
        self.state.changed.connect(self._on_state_changed)

    def _get_params_for_mag(self, mag):
        """Get parameters with per-magnification overrides applied."""
        block_diam = self.state.get('blockDiam', 101)
        fixed_thresh = self.state.get('fixedThresh', 0.04)
        dust_correction = self.state.get('dustCorrection', True)
        min_colony_area = self.state.get('minColonyAreaPx', 200)

        mag_params = self.state.get('magParams', {})
        if mag and mag in mag_params:
            overrides = mag_params[mag]
            block_diam = overrides.get('blockDiam', block_diam)
            fixed_thresh = overrides.get('fixedThresh', fixed_thresh)
            dust_correction = overrides.get('dustCorrection', dust_correction)
            min_colony_area = overrides.get('minColonyAreaPx', min_colony_area)

        return block_diam, fixed_thresh, dust_correction, min_colony_area

    def _on_state_changed(self):
        if not self.isVisible():
            self._stale = True
            return
        self._stale = False
        plates = self.state.get('plates', [])
        current_plates = [
            self.plate_combo.itemData(i) for i in range(self.plate_combo.count())
        ]
        if plates != current_plates:
            self._populate_plates()
        else:
            self._schedule_render()

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_stale', False):
            self._stale = False
            self._on_state_changed()

    def _populate_plates(self):
        prev_plate = self.plate_combo.currentData()
        self.plate_combo.blockSignals(True)
        self.plate_combo.clear()
        restore_idx = 0
        for i, p in enumerate(self.state.get('plates', [])):
            self.plate_combo.addItem(os.path.basename(p), p)
            if p == prev_plate:
                restore_idx = i
        self.plate_combo.blockSignals(False)
        if self.plate_combo.count() > 0:
            self.plate_combo.setCurrentIndex(restore_idx)
            self._on_plate_changed(restore_idx)

    def _on_plate_changed(self, idx):
        plate_path = self.plate_combo.currentData()
        if not plate_path:
            self._well_entries = []
            self.well_combo.clear()
            self._load_source()
            self._schedule_render()
            return

        self.mag_combo.clear()
        self.mag_combo.setEnabled(False)
        self.well_combo.clear()
        self.well_combo.addItem('Scanning...')
        self.well_combo.setEnabled(False)

        def _scan():
            try:
                return discover_wells_with_mag(plate_path)
            except Exception:
                return []

        threading.Thread(target=lambda: self._well_result.emit(_scan()), daemon=True).start()

    def _on_wells_discovered(self, entries):
        self._well_entries = entries or []
        self.well_combo.setEnabled(True)
        self.mag_combo.setEnabled(True)

        # Extract unique magnifications
        mags = sorted({mag for _, _, mag, _ in self._well_entries if mag})

        prev_mag = self.mag_combo.currentData()
        self.mag_combo.blockSignals(True)
        self.mag_combo.clear()
        if not mags:
            self.mag_combo.addItem('(none)', '')
        else:
            restore_idx = 0
            for i, mag in enumerate(mags):
                mag_label = MAG_SUFFIXES.get(mag, mag)
                self.mag_combo.addItem(mag_label, mag)
                if mag == prev_mag:
                    restore_idx = i
            self.mag_combo.setCurrentIndex(restore_idx)
        self.mag_combo.blockSignals(False)

        self._populate_wells_for_mag()

    def _on_mag_changed(self, idx):
        self._populate_wells_for_mag()

    def _populate_wells_for_mag(self):
        """Filter well combo to show only wells for the selected magnification."""
        selected_mag = self.mag_combo.currentData() or ''
        filtered = [(label, well, mag, source)
                     for label, well, mag, source in self._well_entries
                     if mag == selected_mag]

        prev_well = self.well_combo.currentData()
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        restore_idx = 0
        for i, (label, well, mag, source) in enumerate(filtered):
            self.well_combo.addItem(well, i)  # store index into filtered
            if well == prev_well:
                restore_idx = i
        self.well_combo.blockSignals(False)

        # Store filtered list for _load_source
        self._filtered_entries = filtered

        if self.well_combo.count() > 0:
            self.well_combo.setCurrentIndex(restore_idx)
        self._load_source()
        self._schedule_render()

    def _on_well_changed(self, idx):
        self._load_source()
        self._schedule_render()

    def _load_source(self):
        idx = self.well_combo.currentIndex()
        plate_path = self.plate_combo.currentData()
        filtered = getattr(self, '_filtered_entries', [])

        if idx < 0 or idx >= len(filtered) or not plate_path:
            self._current_source = None
            self._current_mag = ''
            self._current_well = ''
            self._n_frames = 0
            return

        label, well, mag, source = filtered[idx]
        self._current_source = source
        self._current_mag = mag
        self._current_well = well

        if self._current_source is not None:
            _, self._n_frames = load_frame(self._current_source, 0)
            old_val = self.frame_slider.value()
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, self._n_frames - 1))
            if old_val <= self._n_frames - 1:
                self.frame_slider.setValue(old_val)
            else:
                self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(
                f'{self.frame_slider.value()} / {max(0, self._n_frames - 1)}'
            )
        else:
            self._n_frames = 0
            self.frame_slider.setRange(0, 0)

    def _on_frame_changed(self, val):
        self.frame_label.setText(f'{val} / {max(0, self._n_frames - 1)}')
        self._schedule_render()

    def _schedule_render(self):
        self._debounce_timer.start()

    def _refresh_all(self):
        self._populate_plates()

    def _render(self):
        for ax in self.figure.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        if self._current_source is None:
            self.ax_raw.set_title('No image')
            self.params_label.setText('')
            self.canvas.draw()
            return

        frame_idx = self.frame_slider.value()
        raw, _ = load_frame(self._current_source, frame_idx)
        if raw is None:
            self.ax_raw.set_title('Could not load')
            self.params_label.setText('')
            self.canvas.draw()
            return

        mag = self._current_mag
        block_diam, fixed_thresh, dust_correction, min_colony_area = self._get_params_for_mag(mag)

        mag_params = self.state.get('magParams', {})
        if mag and mag in mag_params:
            self.params_label.setText(
                f'Using per-mag overrides for {mag}: {mag_params[mag]}'
            )
        else:
            self.params_label.setText(
                f'Using global parameters'
                + (f' (mag {mag})' if mag else '')
            )

        # --- Top row: processing preview ---
        # Match pipeline exactly: scale to [0,1], normalize, blur with sigma=2.0

        raw_scaled = raw.astype(np.float32)
        rmax = raw_scaled.max()
        if rmax > 0:
            raw_scaled /= rmax

        processed = normalize_local_contrast(raw_scaled, block_diam)
        sigma = 2.0  # matches pipeline hardcoded sigma
        blurred = cv2.GaussianBlur(
            processed, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )

        mask_live = blurred > fixed_thresh

        # display raw
        self.ax_raw.imshow(raw, cmap='gray')
        self.ax_raw.set_title('Raw')

        # display preprocessed
        self.ax_proc.imshow(processed, cmap='gray')
        self.ax_proc.set_title(
            f'Preprocessed\nblockDiam={block_diam}',
            fontsize=9,
        )

        # mask overlay on preprocessed
        if processed.max() > 0:
            display = processed / processed.max()
        else:
            display = processed
        overlay = np.stack([display, display, display], axis=-1)
        overlay[mask_live] = [0, 1, 1]
        self.ax_mask.imshow(overlay)
        self.ax_mask.set_title(
            f'Mask Overlay\nthresh={fixed_thresh}  dust={dust_correction}',
            fontsize=9,
        )

        self.figure.tight_layout()
        self.canvas.draw()
