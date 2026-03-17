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
from matplotlib.colors import ListedColormap

import tifffile
from scipy.ndimage import gaussian_filter

from multiWellAnalysis.processing.preprocessing import normalize_local_contrast


def discover_wells(plate_dir):
    """Find well IDs from TIF filenames in a plate directory."""
    if not plate_dir or not os.path.isdir(plate_dir):
        return []
    wells = set()
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
    """Find image file(s) for a given well."""
    proc_dir = os.path.join(plate_dir, 'processedImages')

    for d in [proc_dir, plate_dir]:
        raw_path = os.path.join(d, f'{well_id}_registered_raw.tif')
        if os.path.exists(raw_path):
            return raw_path

    stack_path = os.path.join(plate_dir, f'{well_id}.tif')
    if os.path.exists(stack_path):
        return stack_path

    proc_path = os.path.join(proc_dir, f'{well_id}_processed.tif')
    if os.path.exists(proc_path):
        return proc_path

    pattern = os.path.join(plate_dir, f'{well_id}_*.tif')
    files = sorted(glob.glob(pattern))
    if files:
        return files

    return None


def find_masks(plate_dir, well_id):
    """Find saved mask .npz for a well."""
    proc_dir = os.path.join(plate_dir, 'processedImages')
    for d in [proc_dir, plate_dir]:
        path = os.path.join(d, f'{well_id}_masks.npz')
        if os.path.exists(path):
            return path
    return None


def find_tracked_labels(plate_dir, well_id):
    """Find tracked labels .npz for a well."""
    proc_dir = os.path.join(plate_dir, 'processedImages')
    for d in [proc_dir, plate_dir]:
        matches = glob.glob(os.path.join(d, f'{well_id}_trackedLabels_*.npz'))
        if matches:
            return matches[0]
    return None


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


def load_mask_frame(mask_path, frame_idx):
    """Load a single mask frame from .npz."""
    if mask_path is None:
        return None
    data = np.load(mask_path)
    key = 'masks' if 'masks' in data else list(data.keys())[0]
    masks = data[key]
    if masks.ndim == 3:
        if masks.shape[2] < masks.shape[0] and masks.shape[2] < masks.shape[1]:
            # (H, W, T)
            frame_idx = min(frame_idx, masks.shape[2] - 1)
            return masks[:, :, frame_idx]
        else:
            # (T, H, W)
            frame_idx = min(frame_idx, masks.shape[0] - 1)
            return masks[frame_idx]
    return masks


def load_label_frame(label_path, frame_idx):
    """Load a single tracked label frame from .npz."""
    if label_path is None:
        return None, None
    data = np.load(label_path)
    if 'labels' not in data:
        return None, None
    labels = data['labels']
    frames = data.get('frames', np.arange(labels.shape[2] if labels.ndim == 3 else 1))
    if labels.ndim == 3:
        if labels.shape[2] < labels.shape[0] and labels.shape[2] < labels.shape[1]:
            # (H, W, T)
            idx = min(frame_idx, labels.shape[2] - 1)
            return labels[:, :, idx], frames
        else:
            # (T, H, W)
            idx = min(frame_idx, labels.shape[0] - 1)
            return labels[idx], frames
    return labels, frames


def _make_label_cmap(n_labels):
    """Create a random colormap for label visualization."""
    rng = np.random.RandomState(42)
    colors = rng.rand(max(n_labels + 1, 2), 3)
    colors[0] = [0, 0, 0]  # background = black
    return ListedColormap(colors)


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
        self._mask_path = None
        self._label_path = None
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

        # matplotlib canvas: 2 rows x 3 columns
        self.figure = Figure(figsize=(12, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)

        # top row: processing preview
        self.ax_raw = self.figure.add_subplot(2, 3, 1)
        self.ax_proc = self.figure.add_subplot(2, 3, 2)
        self.ax_mask = self.figure.add_subplot(2, 3, 3)

        # bottom row: colony preview
        self.ax_seg = self.figure.add_subplot(2, 3, 4)
        self.ax_labels = self.figure.add_subplot(2, 3, 5)
        self.ax_colony_overlay = self.figure.add_subplot(2, 3, 6)

        for ax in self.figure.axes:
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
        plates = self.state.get('plates', [])
        current_plates = [
            self.plate_combo.itemData(i) for i in range(self.plate_combo.count())
        ]
        if plates != current_plates:
            self._populate_plates()
        else:
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
            self._mask_path = None
            self._label_path = None
            return

        self._current_source = find_well_images(plate_path, well_id)
        self._mask_path = find_masks(plate_path, well_id)
        self._label_path = find_tracked_labels(plate_path, well_id)

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
        min_colony_area = self.state.get('minColonyAreaPx', 200)

        # --- Top row: processing preview ---

        # preprocess
        processed = normalize_local_contrast(raw, block_diam)
        sigma = 2.0 * (block_diam / 31.0)
        blurred = gaussian_filter(processed, sigma=sigma)

        # mask (live from parameters)
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
        overlay[mask_live] = [0, 1, 1]  # cyan
        self.ax_mask.imshow(overlay)
        self.ax_mask.set_title(
            f'Mask Overlay\nthresh={fixed_thresh}  dust={dust_correction}',
            fontsize=9,
        )

        # --- Bottom row: colony segmentation / tracking preview ---

        # Load saved mask if available, otherwise use live mask
        saved_mask = load_mask_frame(self._mask_path, frame_idx)
        mask_for_colony = saved_mask if saved_mask is not None else mask_live

        # Colony segmentation (live)
        try:
            from multiWellAnalysis.colony.segmentation import segmentColonies
            labels_seg, props = segmentColonies(
                raw, mask_for_colony, minColonyArea_px=min_colony_area
            )
            n_colonies = labels_seg.max()
            cmap = _make_label_cmap(n_colonies)
            self.ax_seg.imshow(labels_seg, cmap=cmap, interpolation='nearest')
            self.ax_seg.set_title(
                f'Colony Segmentation\n{n_colonies} colonies (minArea={min_colony_area})',
                fontsize=9,
            )
        except Exception as e:
            self.ax_seg.set_title(f'Segmentation error\n{e}', fontsize=8)

        # Tracked labels (from saved file)
        label_frame, frames = load_label_frame(self._label_path, frame_idx)
        if label_frame is not None:
            n_tracked = label_frame.max()
            cmap_tracked = _make_label_cmap(n_tracked)
            self.ax_labels.imshow(
                label_frame, cmap=cmap_tracked, interpolation='nearest'
            )
            self.ax_labels.set_title(
                f'Tracked Labels\n{n_tracked} colonies',
                fontsize=9,
            )
        else:
            self.ax_labels.set_title('Tracked Labels\n(not yet computed)', fontsize=9)

        # Colony overlay on raw
        if raw.max() > 0:
            raw_norm = raw / raw.max()
        else:
            raw_norm = raw
        colony_overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

        # Use tracked labels if available, else live segmentation
        overlay_labels = label_frame if label_frame is not None else labels_seg if 'labels_seg' in dir() else None
        if overlay_labels is not None and overlay_labels.max() > 0:
            n = overlay_labels.max()
            rng = np.random.RandomState(42)
            colors = rng.rand(n + 1, 3)
            colors[0] = [0, 0, 0]
            # Crop to matching dimensions
            h = min(colony_overlay.shape[0], overlay_labels.shape[0])
            w = min(colony_overlay.shape[1], overlay_labels.shape[1])
            for label_id in range(1, n + 1):
                region = overlay_labels[:h, :w] == label_id
                if region.any():
                    colony_overlay[:h, :w][region] = (
                        colony_overlay[:h, :w][region] * 0.5
                        + colors[label_id] * 0.5
                    )
            self.ax_colony_overlay.imshow(colony_overlay)
            self.ax_colony_overlay.set_title('Colony Overlay', fontsize=9)
        else:
            self.ax_colony_overlay.imshow(colony_overlay)
            self.ax_colony_overlay.set_title(
                'Colony Overlay\n(no colonies)', fontsize=9
            )

        self.figure.tight_layout()
        self.canvas.draw()
