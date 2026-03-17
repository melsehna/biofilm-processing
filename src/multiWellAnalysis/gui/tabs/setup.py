import os
import re
import glob
import threading
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer


def discover_plates(root_dir, depth=0):
    """Find candidate plate directories under root_dir.

    Lists only immediate subdirectories (single os.listdir call — no stat,
    no is_dir check, no file scanning). Instant even over SMB.

    Returns all direct children as candidates. The user unchecks non-plates.
    Use the 'Search deeper' button to recurse one level if the root contains
    experiment folders rather than plate folders.
    """
    if not root_dir or not os.path.isdir(root_dir):
        return []

    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        return []

    # Filter out hidden dirs and known output dirs by name only (no stat calls)
    skip = {
        'processedimages', 'processed_images_py', 'numerical_data_py',
        'numericaldata', 'plots', '__pycache__', '.git', 'checkpoints',
    }

    candidates = []
    for name in entries:
        if name.startswith('.') or name.startswith('~$'):
            continue
        if name.lower() in skip:
            continue
        # Skip obvious non-directory files by extension
        if '.' in name and name.rsplit('.', 1)[1].lower() in {
            'tif', 'tiff', 'csv', 'json', 'xlsx', 'xls', 'pdf', 'png',
            'jpg', 'mp4', 'npz', 'npy', 'log', 'txt', 'py', 'r', 'md',
        }:
            continue
        candidates.append(os.path.join(root_dir, name))

    if not candidates:
        # No subdirectories found — root itself is likely a plate
        return [root_dir]

    # If depth requested, recurse one level into each candidate
    if depth > 0:
        deeper = []
        for c in candidates:
            deeper.extend(discover_plates(c, depth=depth - 1))
        return sorted(deeper)

    return sorted(candidates)


class SetupTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._build_ui()
        self._connect_signals()

        # restore from state if already set
        root = self.state.get('rootDir', '')
        if root:
            self.root_edit.setText(root)
            self._refresh_plates()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # root directory row
        dir_label = QLabel('Root Directory (input data):')
        layout.addWidget(dir_label)

        dir_row = QHBoxLayout()
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText('Select directory containing plate folders...')
        dir_row.addWidget(self.root_edit)

        self.browse_btn = QPushButton('Browse...')
        dir_row.addWidget(self.browse_btn)

        self.refresh_btn = QPushButton('Refresh')
        dir_row.addWidget(self.refresh_btn)

        self.deeper_btn = QPushButton('Search deeper')
        self.deeper_btn.setToolTip('Look inside each folder for plate subdirectories')
        dir_row.addWidget(self.deeper_btn)
        layout.addLayout(dir_row)

        # output directory row
        outdir_label = QLabel('Output Directory:')
        layout.addWidget(outdir_label)

        outdir_row = QHBoxLayout()
        self.outdir_edit = QLineEdit()
        self.outdir_edit.setPlaceholderText('Select directory for processed outputs...')
        self.outdir_edit.setText(self.state.get('outputDir', ''))
        outdir_row.addWidget(self.outdir_edit)

        self.outdir_browse_btn = QPushButton('Browse...')
        outdir_row.addWidget(self.outdir_browse_btn)
        layout.addLayout(outdir_row)

        # plate list
        plates_label = QLabel('Discovered Plates:')
        layout.addWidget(plates_label)

        self.plate_list = QListWidget()
        layout.addWidget(self.plate_list, stretch=1)

        # magnification selector
        mag_label = QLabel('Magnifications (auto-detected from plates):')
        layout.addWidget(mag_label)

        self.mag_list = QListWidget()
        self.mag_list.setMaximumHeight(90)
        layout.addWidget(self.mag_list)

        # notes
        notes_label = QLabel('Notes:')
        layout.addWidget(notes_label)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.setPlaceholderText('Experiment notes...')
        layout.addWidget(self.notes_edit)

    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._browse)
        self.refresh_btn.clicked.connect(self._refresh_plates)
        self.deeper_btn.clicked.connect(self._refresh_plates_deeper)
        self.outdir_browse_btn.clicked.connect(self._browse_outdir)
        self.outdir_edit.editingFinished.connect(
            lambda: self.state.set('outputDir', self.outdir_edit.text().strip())
        )
        self.plate_list.itemChanged.connect(self._update_selected_plates)
        self.mag_list.itemChanged.connect(self._on_mag_changed)
        self.notes_edit.textChanged.connect(
            lambda: self.state.set('notes', self.notes_edit.toPlainText())
        )

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Root Directory')
        if path:
            self.root_edit.setText(path)
            self.state.set('rootDir', path)

            # auto-load config if it exists in root or output dir
            config_path = os.path.join(path, 'experiment_config.json')
            if os.path.exists(config_path):
                try:
                    self.state.load(config_path)
                    self.root_edit.setText(self.state.get('rootDir', path))
                    self.outdir_edit.setText(self.state.get('outputDir', ''))
                except Exception:
                    pass

            self._refresh_plates()

    def _browse_outdir(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if path:
            self.outdir_edit.setText(path)
            self.state.set('outputDir', path)

    def _refresh_plates(self, search_depth=0):
        root = self.root_edit.text().strip()
        self.plate_list.blockSignals(True)
        self.plate_list.clear()
        self.plate_list.blockSignals(False)
        self.refresh_btn.setEnabled(False)
        self.deeper_btn.setEnabled(False)
        self.refresh_btn.setText('Scanning...')

        def _scan():
            return discover_plates(root, depth=search_depth), root

        def _done(plates, root_used):
            self.refresh_btn.setEnabled(True)
            self.deeper_btn.setEnabled(True)
            self.refresh_btn.setText('Refresh')
            self.plate_list.blockSignals(True)
            self.plate_list.clear()
            for p in plates:
                try:
                    display = os.path.relpath(p, root_used)
                except ValueError:
                    display = os.path.basename(p)
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, p)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.plate_list.addItem(item)
            self.plate_list.blockSignals(False)
            self._update_selected_plates()

        self._run_in_background(_scan, _done)

    def _refresh_plates_deeper(self):
        self._refresh_plates(search_depth=1)

    def _update_selected_plates(self):
        selected = []
        for i in range(self.plate_list.count()):
            item = self.plate_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        self.state.set('plates', selected)
        self._scan_magnifications_async(selected)

    def _on_mag_changed(self, item=None):
        """Update state with currently checked magnifications."""
        selected = []
        for i in range(self.mag_list.count()):
            it = self.mag_list.item(i)
            if it.checkState() == Qt.Checked:
                selected.append(it.data(Qt.UserRole))
        total = self.mag_list.count()
        if not selected or len(selected) == total:
            self.state.set('magnification', 'all')
        elif len(selected) == 1:
            self.state.set('magnification', selected[0])
        else:
            self.state.set('magnification', selected)

    def _scan_magnifications_async(self, plates):
        """Detect magnifications from the first plate.

        Strategy (fast to slow, stops at first success):
        1. Read protocol.csv if it exists (tiny file, explicit mag labels)
        2. Sample a handful of filenames from scandir (parse mag suffix)
        """
        def _scan():
            all_mags = set()
            if not plates:
                return all_mags
            plate_path = plates[0]

            # Method 1: protocol.csv (instant — small file)
            protocol_path = os.path.join(plate_path, 'protocol.csv')
            if os.path.exists(protocol_path):
                try:
                    import csv
                    with open(protocol_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if (row.get('action') == 'Imaging Read'
                                    and row.get('channel') == 'Bright Field'
                                    and 'step' in row):
                                mag = row.get('magnification', '')
                                step = row.get('step', '')
                                label = str(mag) if mag else f'_{step}'
                                all_mags.add(label)
                except Exception:
                    pass
                if all_mags:
                    return all_mags

            # Method 2: sample filenames (stop after first 20 tifs found)
            try:
                tif_count = 0
                for entry in os.scandir(plate_path):
                    if tif_count >= 20:
                        break
                    if not entry.name.endswith('.tif'):
                        continue
                    tif_count += 1
                    if 'Bright Field' in entry.name or 'Bright_Field' in entry.name:
                        m = re.match(r'^[A-P]\d+(_\d+)_', entry.name)
                        if m:
                            all_mags.add(m.group(1))
            except PermissionError:
                pass

            return all_mags

        def _done(all_mags):
            if not all_mags:
                self.mag_list.clear()
                self.state.set('magnification', 'all')
                return

            saved = self.state.get('magnification', 'all')
            if isinstance(saved, str) and saved != 'all':
                saved = [saved]
            elif saved == 'all':
                saved = list(all_mags)

            self.mag_list.blockSignals(True)
            self.mag_list.clear()
            for mag in sorted(all_mags):
                item = QListWidgetItem(f'Magnification {mag}')
                item.setData(Qt.UserRole, mag)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if mag in saved else Qt.Unchecked)
                self.mag_list.addItem(item)
            self.mag_list.blockSignals(False)
            self._on_mag_changed()

        self._run_in_background(_scan, _done)

    def _run_in_background(self, work_fn, done_fn):
        """Run work_fn in a thread, call done_fn(*result) on the main thread.

        work_fn returns a single value or tuple; done_fn receives it unpacked.
        """
        def _worker():
            try:
                result = work_fn()
            except Exception:
                result = None
            # Schedule callback on main thread via timer
            if result is None:
                return
            if isinstance(result, tuple):
                QTimer.singleShot(0, lambda r=result: done_fn(*r))
            else:
                QTimer.singleShot(0, lambda r=result: done_fn(r))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
