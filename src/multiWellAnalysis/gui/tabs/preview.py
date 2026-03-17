import os
import glob
import re
import numpy as np
from collections import defaultdict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSlider,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

import cv2
import tifffile

from multiWellAnalysis.processing.preprocessing import normalize_local_contrast


def discover_wells_with_mag(plate_dir):
    """Find well+mag combinations from TIF filenames.

    Returns list of (display_label, well_id, mag_suffix, file_list_or_path) tuples.
    For plates without magnification suffixes, mag_suffix is ''.
    """
    if not plate_dir or not os.path.isdir(plate_dir):
        return []

    tif_files = sorted(glob.glob(os.path.join(plate_dir, '*.tif')))
    bf_files = [f for f in tif_files if 'Bright Field' in f or 'Bright_Field' in f]

    # Try magnification-aware grouping
    if bf_files:
        groups = defaultdict(lambda: defaultdict(list))
        for f in bf_files:
            base = os.path.basename(f)
            m = re.match(r'^([A-H]\d+)(_\d+)_', base)
            if m:
                well, mag = m.group(1), m.group(2)
                groups[mag][well].append(f)

        if groups:
            result = []
            for mag in sorted(groups):
                for well in sorted(groups[mag]):
                    files = sorted(groups[mag][well])
                    label = f'{well} (mag {mag})'
                    result.append((label, well, mag, files))
            return result

    # Fallback: no magnification suffixes
    # Check for processed files first
    proc_dir = os.path.join(plate_dir, 'processedImages')
    wells = set()

    for d in [proc_dir, plate_dir]:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, '*.tif')):
            name = os.path.basename(f)
            m = re.match(r'^([A-H]\d{1,2})', name)
            if m:
                wells.add(m.group(1))

    result = []
    for well in sorted(wells):
        source = _find_well_source(plate_dir, well)
        if source:
            result.append((well, well, '', source))
    return result


def _find_well_source(plate_dir, well_id):
    """Find image source for a well without mag suffix."""
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
    return files if files else None


def find_masks(plate_dir, well_id, mag=''):
    """Find saved mask .npz for a well (with optional mag suffix)."""
    label = f'{well_id}_{mag}' if mag else well_id
    proc_dir = os.path.join(plate_dir, 'processedImages')
    for d in [proc_dir, plate_dir]:
        for name in [f'{label}_masks.npz', f'{well_id}_masks.npz']:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    return None


def find_tracked_labels(plate_dir, well_id, mag=''):
    """Find tracked labels .npz for a well."""
    label = f'{well_id}_{mag}' if mag else well_id
    proc_dir = os.path.join(plate_dir, 'processedImages')
    for d in [proc_dir, plate_dir]:
        for prefix in [label, well_id]:
            matches = glob.glob(os.path.join(d, f'{prefix}_trackedLabels_*.npz'))
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
            frame_idx = min(frame_idx, masks.shape[2] - 1)
            return masks[:, :, frame_idx]
        else:
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
            idx = min(frame_idx, labels.shape[2] - 1)
            return labels[:, :, idx], frames
        else:
            idx = min(frame_idx, labels.shape[0] - 1)
            return labels[idx], frames
    return labels, frames


def _make_label_cmap(n_labels):
    """Create a random colormap for label visualization."""
    rng = np.random.RandomState(42)
    colors = rng.rand(max(n_labels + 1, 2), 3)
    colors[0] = [0, 0, 0]
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
        self._current_mag = ''
        self._current_well = ''
        self._n_frames = 0
        self._mask_path = None
        self._label_path = None
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

        # matplotlib canvas: 2 rows x 3 columns
        self.figure = Figure(figsize=(12, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.ax_raw = self.figure.add_subplot(2, 3, 1)
        self.ax_proc = self.figure.add_subplot(2, 3, 2)
        self.ax_mask = self.figure.add_subplot(2, 3, 3)
        self.ax_seg = self.figure.add_subplot(2, 3, 4)
        self.ax_labels = self.figure.add_subplot(2, 3, 5)
        self.ax_colony_overlay = self.figure.add_subplot(2, 3, 6)

        for ax in self.figure.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        self.figure.tight_layout()
        layout.addWidget(self.canvas, stretch=1)

        self.refresh_btn = QPushButton('Refresh')
        layout.addWidget(self.refresh_btn)

    def _connect_signals(self):
        self.plate_combo.currentIndexChanged.connect(self._on_plate_changed)
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
        self._well_entries = discover_wells_with_mag(plate_path) if plate_path else []

        prev_text = self.well_combo.currentText()
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        restore_idx = 0
        for i, (label, well, mag, source) in enumerate(self._well_entries):
            self.well_combo.addItem(label)
            if label == prev_text:
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
        idx = self.well_combo.currentIndex()
        plate_path = self.plate_combo.currentData()

        if idx < 0 or idx >= len(self._well_entries) or not plate_path:
            self._current_source = None
            self._current_mag = ''
            self._current_well = ''
            self._n_frames = 0
            self._mask_path = None
            self._label_path = None
            return

        label, well, mag, source = self._well_entries[idx]
        self._current_source = source
        self._current_mag = mag
        self._current_well = well
        self._mask_path = find_masks(plate_path, well, mag)
        self._label_path = find_tracked_labels(plate_path, well, mag)

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

        # --- Bottom row: colony segmentation / tracking preview ---

        saved_mask = load_mask_frame(self._mask_path, frame_idx)
        mask_for_colony = saved_mask if saved_mask is not None else mask_live

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
            labels_seg = None
            self.ax_seg.set_title(f'Segmentation error\n{e}', fontsize=8)

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
        if rmax > 0:
            raw_norm = raw / raw.max()
        else:
            raw_norm = raw.astype(np.float64)
        colony_overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

        overlay_labels = label_frame if label_frame is not None else labels_seg
        if overlay_labels is not None and overlay_labels.max() > 0:
            n = overlay_labels.max()
            rng = np.random.RandomState(42)
            colors = rng.rand(n + 1, 3)
            colors[0] = [0, 0, 0]
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
