import os
import glob
import re
import threading
import numpy as np
from collections import defaultdict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSlider,
    QPushButton, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

import tifffile

from multiWellAnalysis.gui.tabs.preview import (
    discover_wells_with_mag, MAG_SUFFIXES,
)


def _make_label_cmap(n_labels):
    """Create a random colormap for label visualization."""
    rng = np.random.RandomState(42)
    colors = rng.rand(max(n_labels + 1, 2), 3)
    colors[0] = [0, 0, 0]
    return ListedColormap(colors)


class TestWellTab(QWidget):
    _well_result = Signal(object)
    _run_log = Signal(str)
    _run_progress = Signal(str, int, int)  # stage, current, total
    _run_finished = Signal(object)  # result dict or None

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._well_result.connect(self._on_wells_discovered)
        self._run_log.connect(self._on_log)
        self._run_progress.connect(self._on_progress)
        self._run_finished.connect(self._on_run_finished)
        self._well_entries = []
        self._filtered_entries = []
        self._result = None  # stores last run result
        self._running = False
        self._stop_event = threading.Event()
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

        # run button + stop button + progress
        run_row = QHBoxLayout()
        self.run_btn = QPushButton('Run Full Pipeline on Well')
        run_row.addWidget(self.run_btn)
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        run_row.addWidget(self.stop_btn)
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: gray; font-size: 11px;')
        self.status_label.setWordWrap(True)
        run_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # frame slider
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel('Frame:'))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        frame_row.addWidget(self.frame_slider, stretch=1)
        self.frame_label = QLabel('0 / 0')
        frame_row.addWidget(self.frame_label)
        layout.addLayout(frame_row)

        # matplotlib canvas: 1 row x 3 columns
        # raw | tracked labels | colony overlay on raw
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.ax_raw = self.figure.add_subplot(1, 3, 1)
        self.ax_labels = self.figure.add_subplot(1, 3, 2)
        self.ax_overlay = self.figure.add_subplot(1, 3, 3)

        for ax in self.figure.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        self.figure.tight_layout()
        layout.addWidget(self.canvas, stretch=1)

    def _connect_signals(self):
        self.plate_combo.currentIndexChanged.connect(self._on_plate_changed)
        self.mag_combo.currentIndexChanged.connect(self._on_mag_changed)
        self.well_combo.currentIndexChanged.connect(self._on_well_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.run_btn.clicked.connect(self._run_pipeline)
        self.stop_btn.clicked.connect(self._stop_pipeline)
        self.state.changed.connect(self._on_state_changed)

    # ── State / plate / well selection (same pattern as PreviewTab) ──

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
        selected_mag = self.mag_combo.currentData() or ''
        filtered = [(label, well, mag, source)
                     for label, well, mag, source in self._well_entries
                     if mag == selected_mag]

        prev_well = self.well_combo.currentData()
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        restore_idx = 0
        for i, (label, well, mag, source) in enumerate(filtered):
            self.well_combo.addItem(well, i)
            if well == prev_well:
                restore_idx = i
        self.well_combo.blockSignals(False)

        self._filtered_entries = filtered

        if self.well_combo.count() > 0:
            self.well_combo.setCurrentIndex(restore_idx)

    def _on_well_changed(self, idx):
        self._result = None
        self._clear_canvas()

    # ── Pipeline execution ──

    def _get_selected_well(self):
        """Return (plate_path, well_id, mag, source) or None."""
        idx = self.well_combo.currentIndex()
        plate_path = self.plate_combo.currentData()
        if idx < 0 or idx >= len(self._filtered_entries) or not plate_path:
            return None
        label, well, mag, source = self._filtered_entries[idx]
        return plate_path, well, mag, source

    def _stop_pipeline(self):
        self._stop_event.set()
        self.stop_btn.setEnabled(False)
        self.status_label.setText('Stopping...')

    def _run_pipeline(self):
        if self._running:
            return

        sel = self._get_selected_well()
        if not sel:
            self.status_label.setText('Select a plate, mag, and well first')
            return

        plate_path, well_id, mag, source = sel
        self._running = True
        self._stop_event.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f'Running pipeline on {well_id}...')

        s = self.state.to_dict()

        # Apply per-mag overrides
        mag_params = s.get('magParams', {})
        if mag and mag in mag_params:
            s.update(mag_params[mag])

        stop = self._stop_event

        def _work():
            try:
                import tempfile
                from multiWellAnalysis.processing.analysis_main import timelapse_processing
                from multiWellAnalysis.colony.runTrackingGUI import (
                    trackColoniesAllFrames, findSeedFrame,
                )

                if stop.is_set():
                    self._run_finished.emit(None)
                    return

                # ── Load ──
                self._run_log.emit(f'Loading images for {well_id}...')
                self._run_progress.emit('Loading', 0, 5)

                if isinstance(source, str):
                    raw = tifffile.imread(source)
                    if raw.ndim == 2:
                        stack = raw[np.newaxis].astype(np.float32)
                    else:
                        stack = raw.astype(np.float32)
                    del raw
                else:
                    first = tifffile.imread(source[0])
                    h, w = first.shape[:2]
                    stack = np.empty((len(source), h, w), dtype=np.float32)
                    stack[0] = first.astype(np.float32)
                    del first
                    for fi in range(1, len(source)):
                        if stop.is_set():
                            self._run_finished.emit(None)
                            return
                        self._run_log.emit(f'Loading frame {fi+1}/{len(source)}...')
                        stack[fi] = tifffile.imread(source[fi]).astype(np.float32)

                # ensure (H, W, T)
                if stack.ndim == 3 and stack.shape[0] < stack.shape[2]:
                    stack = np.transpose(stack, (1, 2, 0))

                if stop.is_set():
                    self._run_finished.emit(None)
                    return

                # ── Step 1: Run the real pipeline (same as batch processing) ──
                self._run_progress.emit('Processing', 1, 5)
                ntimepoints = stack.shape[2]
                self._run_log.emit(f'Step 1/3: Processing {ntimepoints} frames...')

                with tempfile.TemporaryDirectory() as tmpdir:
                    masks, biomass, _ = timelapse_processing(
                        images=stack,
                        block_diameter=s['blockDiam'],
                        ntimepoints=ntimepoints,
                        shift_thresh=s['shiftThresh'],
                        fixed_thresh=s['fixedThresh'],
                        dust_correction=s['dustCorrection'],
                        outdir=tmpdir,
                        filename=well_id,
                        image_records=None,
                        fftStride=s.get('fftStride', 6),
                        downsample=s.get('downsample', 4),
                        skip_overlay=True,
                        workers=1,
                        progress_fn=lambda msg: self._run_log.emit(f'  {msg}'),
                    )
                    # stack was modified in-place by timelapse_processing
                    # (scaled, registered, cropped) — it IS the registered raw now.
                    # However timelapse_processing saves cropped stacks to disk;
                    # read them back so we have the cropped versions.
                    import os as _os
                    proc_dir = _os.path.join(tmpdir, 'processedImages')
                    raw_path = _os.path.join(proc_dir, f'{well_id}_registered_raw.tif')
                    if _os.path.exists(raw_path):
                        registered_raw = tifffile.imread(raw_path)
                        if registered_raw.ndim == 3 and registered_raw.shape[0] < registered_raw.shape[1]:
                            registered_raw = np.transpose(registered_raw, (1, 2, 0))
                    else:
                        # Fallback: use the in-place modified stack
                        registered_raw = stack

                if stop.is_set():
                    self._run_finished.emit(None)
                    return

                ntimepoints = masks.shape[2]

                # ── Step 2: Colony tracking (in memory) ──
                self._run_log.emit('Step 2/3: Colony tracking...')
                self._run_progress.emit('Tracking', 3, 5)

                # Find seed frame using biomass (same logic as trackAndSave)
                seed_frame = findSeedFrame(biomass)
                if seed_frame is None:
                    mask_areas = np.array([masks[..., t].sum() for t in range(ntimepoints)], dtype=float)
                    if mask_areas.max() > 0:
                        seed_frame = findSeedFrame(mask_areas / mask_areas.max())
                if seed_frame is None:
                    seed_frame = 0

                peak_frame = int(np.argmax([masks[..., t].sum() for t in range(ntimepoints)]))
                self._run_log.emit(f'  Seed frame: {seed_frame}, Peak frame: {peak_frame}')

                labels_by_frame, _, reason, frames = trackColoniesAllFrames(
                    registered_raw, masks, seed_frame, peak_frame,
                    min_area=s.get('minColonyAreaPx', 200),
                    prop_radius=s.get('propRadiusPx', 25),
                )

                if stop.is_set():
                    self._run_finished.emit(None)
                    return

                self._run_log.emit(f'Step 3/3: Building result ({reason})')
                self._run_progress.emit('Building result', 4, 5)

                # Build result for visualization (all in memory)
                if labels_by_frame and frames:
                    first_label = labels_by_frame[list(labels_by_frame.keys())[0]]
                    lh, lw = first_label.shape[:2]
                    label_stack = np.zeros((lh, lw, len(frames)), dtype=np.int32)
                    for i, f_idx in enumerate(frames):
                        if f_idx in labels_by_frame:
                            label_stack[:, :, i] = labels_by_frame[f_idx][:lh, :lw]
                else:
                    label_stack = None
                    frames = list(range(ntimepoints))

                result = {
                    'raw_stack': registered_raw,
                    'label_stack': label_stack,
                    'frames': frames,
                    'well_id': well_id,
                }

                self._run_progress.emit('Done', 5, 5)
                self._run_finished.emit(result)

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._run_log.emit(f'Error: {e}\n{tb}')
                print(tb)
                self._run_finished.emit(None)

        threading.Thread(target=_work, daemon=True).start()

    def _on_log(self, msg):
        self.status_label.setText(msg)

    def _on_progress(self, stage, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f'{stage} ({current}/{total})')

    def _on_run_finished(self, result):
        self._running = False
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._result = result

        if result is None:
            if self._stop_event.is_set():
                self.status_label.setText('Stopped by user')
            # else: error message already set by _on_log in the except block
            return

        self.status_label.setText(f'Done — {result["well_id"]}')

        n_frames = len(result['frames']) if result['frames'] else 0
        if n_frames > 0:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, n_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(f'0 / {n_frames - 1}')

        self._render()

    # ── Rendering ──

    def _on_frame_changed(self, val):
        n = len(self._result['frames']) if self._result and self._result['frames'] else 0
        self.frame_label.setText(f'{val} / {max(0, n - 1)}')
        self._render()

    def _clear_canvas(self):
        for ax in self.figure.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_raw.set_title('Raw')
        self.ax_labels.set_title('Tracked Labels')
        self.ax_overlay.set_title('Colony Overlay')
        self.canvas.draw()

    def _render(self):
        for ax in self.figure.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        if self._result is None:
            self.ax_raw.set_title('Raw\n(run pipeline first)')
            self.ax_labels.set_title('Tracked Labels')
            self.ax_overlay.set_title('Colony Overlay')
            self.canvas.draw()
            return

        frame_idx = self.frame_slider.value()

        # Get raw frame from in-memory stack
        raw_stack = self._result.get('raw_stack')
        if raw_stack is None:
            self.ax_raw.set_title('No data')
            self.canvas.draw()
            return
        fi = min(frame_idx, raw_stack.shape[2] - 1)
        raw = raw_stack[:, :, fi].astype(np.float64)

        self.ax_raw.imshow(raw, cmap='gray')
        self.ax_raw.set_title('Raw')

        label_stack = self._result.get('label_stack')
        if label_stack is not None and label_stack.shape[2] > 0:
            fi = min(frame_idx, label_stack.shape[2] - 1)
            label_frame = label_stack[:, :, fi]
            n_tracked = int(label_frame.max())
            cmap = _make_label_cmap(n_tracked)
            self.ax_labels.imshow(label_frame, cmap=cmap, interpolation='nearest')
            self.ax_labels.set_title(f'Tracked Labels\n{n_tracked} colonies', fontsize=9)

            # Colony overlay on raw
            rmax = raw.max()
            if rmax > 0:
                raw_norm = raw / rmax
            else:
                raw_norm = raw.astype(np.float64)
            overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

            if n_tracked > 0:
                rng = np.random.RandomState(42)
                colors = rng.rand(n_tracked + 1, 3)
                colors[0] = [0, 0, 0]
                h = min(overlay.shape[0], label_frame.shape[0])
                w = min(overlay.shape[1], label_frame.shape[1])
                for lid in range(1, n_tracked + 1):
                    region = label_frame[:h, :w] == lid
                    if region.any():
                        overlay[:h, :w][region] = (
                            overlay[:h, :w][region] * 0.5 + colors[lid] * 0.5
                        )

            self.ax_overlay.imshow(overlay)
            self.ax_overlay.set_title('Colony Overlay', fontsize=9)
        else:
            self.ax_labels.set_title('Tracked Labels\n(no results)', fontsize=9)
            self.ax_overlay.set_title('Colony Overlay\n(no results)', fontsize=9)

        self.figure.tight_layout()
        self.canvas.draw()
