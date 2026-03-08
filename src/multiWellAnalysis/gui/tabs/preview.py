import os
import glob
import re
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSlider,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import tifffile
from scipy.ndimage import gaussian_filter

from multiWellAnalysis.processing.preprocessing import normalize_local_contrast


def discover_wells(plate_dir):
    """Find well IDs from TIF filenames in a plate directory."""
    if not plate_dir or not os.path.isdir(plate_dir):
        return []
    wells = set()
    # search both plate root and processedImages subdir
    search_dirs = [plate_dir]
    proc_dir = os.path.join(plate_dir, 'processedImages')
    if os.path.isdir(proc_dir):
        search_dirs.append(proc_dir)
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, '*.tif')):
            name = os.path.basename(f)
            m = re.match(r'^([A-H]\d{1,2})', name)
            if m:
                wells.add(m.group(1))
    return sorted(wells)


def find_well_images(plate_dir, well_id):
    """Find image file(s) for a given well. Returns path to a TIFF stack
    or a sorted list of single-frame TIFs."""
    proc_dir = os.path.join(plate_dir, 'processedImages')

    # prefer registered raw in processedImages (already aligned)
    for d in [proc_dir, plate_dir]:
        raw_path = os.path.join(d, f'{well_id}_registered_raw.tif')
        if os.path.exists(raw_path):
            return raw_path

    # check for a single multi-frame TIFF at plate root
    stack_path = os.path.join(plate_dir, f'{well_id}.tif')
    if os.path.exists(stack_path):
        return stack_path

    # check for processed stack
    proc_path = os.path.join(proc_dir, f'{well_id}_processed.tif')
    if os.path.exists(proc_path):
        return proc_path

    # check for single-frame files matching well_id prefix
    pattern = os.path.join(plate_dir, f'{well_id}_*.tif')
    files = sorted(glob.glob(pattern))
    if files:
        return files

    return None


def load_frame(source, frame_idx):
    """Load a single frame from a TIFF stack or list of files."""
    if isinstance(source, str):
        # single file, possibly multi-frame
        img = tifffile.imread(source)
        if img.ndim == 3:
            # determine axis order: (T, H, W) vs (H, W, T)
            if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                # (T, H, W)
                n_frames = img.shape[0]
                frame_idx = min(frame_idx, n_frames - 1)
                return img[frame_idx].astype(np.float64), n_frames
            elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
                # (H, W, T)
                n_frames = img.shape[2]
                frame_idx = min(frame_idx, n_frames - 1)
                return img[:, :, frame_idx].astype(np.float64), n_frames
            else:
                # ambiguous, assume (T, H, W)
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
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._render)
        self._current_source = None
        self._n_frames = 0
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # selectors row
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel('Plate:'))
        self.plate_combo = QComboBox()
        sel_row.addWidget(self.plate_combo, stretch=1)

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

        # matplotlib canvas: 3 panels
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_raw = self.figure.add_subplot(1, 3, 1)
        self.ax_proc = self.figure.add_subplot(1, 3, 2)
        self.ax_mask = self.figure.add_subplot(1, 3, 3)
        for ax in (self.ax_raw, self.ax_proc, self.ax_mask):
            ax.set_xticks([])
            ax.set_yticks([])
        self.figure.tight_layout()
        layout.addWidget(self.canvas, stretch=1)

        # refresh button
        self.refresh_btn = QPushButton('Refresh')
        layout.addWidget(self.refresh_btn)

    def _connect_signals(self):
        self.plate_combo.currentIndexChanged.connect(self._on_plate_changed)
        self.well_combo.currentIndexChanged.connect(self._on_well_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.refresh_btn.clicked.connect(self._refresh_all)
        self.state.changed.connect(self._on_state_changed)

    def _on_state_changed(self):
        """Repopulate plates if the list changed; debounce re-render for param changes."""
        plates = self.state.get('plates', [])
        current_plates = [
            self.plate_combo.itemData(i) for i in range(self.plate_combo.count())
        ]
        if plates != current_plates:
            self._populate_plates()
        else:
            # parameter change only — re-render without resetting selectors
            self._schedule_render()

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
        wells = discover_wells(plate_path) if plate_path else []

        prev_well = self.well_combo.currentText()
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        restore_idx = 0
        for i, w in enumerate(wells):
            self.well_combo.addItem(w)
            if w == prev_well:
                restore_idx = i
        self.well_combo.blockSignals(False)

        if self.well_combo.count() > 0:
            self.well_combo.setCurrentIndex(restore_idx)
        self._load_source()
        self._schedule_render()

    def _on_well_changed(self, idx):
        self._load_source()
        self._schedule_render()

    def _load_source(self):
        plate_path = self.plate_combo.currentData()
        well_id = self.well_combo.currentText()
        if not plate_path or not well_id:
            self._current_source = None
            self._n_frames = 0
            return
        self._current_source = find_well_images(plate_path, well_id)
        if self._current_source is not None:
            _, self._n_frames = load_frame(self._current_source, 0)
            old_val = self.frame_slider.value()
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, self._n_frames - 1))
            # keep current frame if within range, otherwise reset to 0
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
        for ax in (self.ax_raw, self.ax_proc, self.ax_mask):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        if self._current_source is None:
            self.ax_raw.set_title('No image')
            self.canvas.draw()
            return

        frame_idx = self.frame_slider.value()
        raw, _ = load_frame(self._current_source, frame_idx)
        if raw is None:
            self.ax_raw.set_title('Could not load')
            self.canvas.draw()
            return

        block_diam = self.state.get('blockDiam', 101)
        fixed_thresh = self.state.get('fixedThresh', 0.04)
        dust_correction = self.state.get('dustCorrection', True)

        # preprocess
        processed = normalize_local_contrast(raw, block_diam)
        sigma = 2.0 * (block_diam / 31.0)
        blurred = gaussian_filter(processed, sigma=sigma)

        # mask (computed from preprocessed image)
        mask = blurred > fixed_thresh

        # display raw
        self.ax_raw.imshow(raw, cmap='gray')
        self.ax_raw.set_title('Raw')

        # display preprocessed with parameters
        self.ax_proc.imshow(processed, cmap='gray')
        self.ax_proc.set_title(
            f'Preprocessed\nblockDiam={block_diam}',
            fontsize=9,
        )

        # overlay: mask on PREPROCESSED image (not raw) to avoid vignetting
        if processed.max() > 0:
            display = processed / processed.max()
        else:
            display = processed
        overlay = np.stack([display, display, display], axis=-1)
        overlay[mask] = [0, 1, 1]  # cyan
        self.ax_mask.imshow(overlay)
        self.ax_mask.set_title(
            f'Mask Overlay\nthresh={fixed_thresh}  dust={dust_correction}',
            fontsize=9,
        )

        self.figure.tight_layout()
        self.canvas.draw()
