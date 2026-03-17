import os
import re
import glob
import threading
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal


def discover_plates(root_dir, depth=0):
    """Find candidate plate directories under root_dir.

    Lists only immediate subdirectories (single os.listdir call — no stat,
    no is_dir check, no file scanning). Instant even over SMB.

    Returns all direct children as candidates. The user unchecks non-plates.
    Use the 'Search deeper' button to recurse one level if the root contains
    experiment folders rather than plate folders.
    """
    if not root_dir:
        return []

    try:
        entries = sorted(os.listdir(root_dir))
    except (PermissionError, FileNotFoundError, OSError):
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
    _bg_result = Signal(str, object)  # (task_name, result)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._bg_callbacks = {}
        self._bg_result.connect(self._on_bg_result)
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
        mag_row = QHBoxLayout()
        mag_row.addWidget(QLabel('Magnifications:'))
        self.detect_mag_btn = QPushButton('Detect from files')
        self.detect_mag_btn.setToolTip('Scan a few filenames in the first checked plate to detect magnification suffixes')
        mag_row.addWidget(self.detect_mag_btn)
        mag_row.addStretch()
        self.mag_status = QLabel('')
        self.mag_status.setStyleSheet('color: gray; font-size: 11px;')
        mag_row.addWidget(self.mag_status)
        layout.addLayout(mag_row)

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
        self.detect_mag_btn.clicked.connect(self._on_detect_mag_clicked)
        self.mag_list.itemChanged.connect(self._on_mag_changed)
        self.notes_edit.textChanged.connect(
            lambda: self.state.set('notes', self.notes_edit.toPlainText())
        )

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Root Directory')
        if path:
            self.root_edit.setText(path)
            self.state.set('rootDir', path)

            # Try to load config in background (os.path.exists is slow over SMB)
            def _try_load_config():
                config_path = os.path.join(path, 'experiment_config.json')
                try:
                    with open(config_path, 'r') as f:
                        import json
                        return json.load(f)
                except (FileNotFoundError, OSError):
                    return None

            def _config_loaded(config_data):
                if config_data:
                    self.state.from_dict(config_data)
                    self.root_edit.setText(self.state.get('rootDir', path))
                    self.outdir_edit.setText(self.state.get('outputDir', ''))

            self._run_in_background('config', _try_load_config, _config_loaded)
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

        def _done(result):
            self.refresh_btn.setEnabled(True)
            self.deeper_btn.setEnabled(True)
            self.refresh_btn.setText('Refresh')
            if isinstance(result, tuple):
                plates, root_used = result
            else:
                plates, root_used = (result or []), root
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
                item.setCheckState(Qt.Unchecked)
                self.plate_list.addItem(item)
            self.plate_list.blockSignals(False)
            self._update_selected_plates()

        self._run_in_background('plates', _scan, _done)

    def _refresh_plates_deeper(self):
        self._refresh_plates(search_depth=1)

    def _update_selected_plates(self):
        selected = []
        for i in range(self.plate_list.count()):
            item = self.plate_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        self.state.set('plates', selected)

    def _on_detect_mag_clicked(self):
        selected = self.state.get('plates', [])
        if not selected:
            self.mag_status.setText('No plates selected')
            return
        self.detect_mag_btn.setEnabled(False)
        self.mag_status.setText('Scanning...')
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
        """Detect magnifications (background thread).

        Strategy (fast to slow, stops at first success):
        1. Read protocol.csv in plate dir or one level deeper (tiny file)
        2. Parse directory/parent names for magnification patterns (no I/O)
        3. Sample first 30 filenames via os.listdir (one SMB call on plate dir)
        """
        def _scan():
            import csv
            all_mags = set()
            if not plates:
                return all_mags, 'no plates'

            # Collect dirs to check: each plate + its children
            dirs_to_check = []
            for plate_path in plates[:3]:
                dirs_to_check.append(plate_path)
                try:
                    for name in os.listdir(plate_path):
                        if name.startswith('.') or '.' in name:
                            continue
                        dirs_to_check.append(os.path.join(plate_path, name))
                except (PermissionError, OSError):
                    pass

            # Method 1: protocol.csv
            for d in dirs_to_check:
                try:
                    with open(os.path.join(d, 'protocol.csv'), 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if (row.get('action') == 'Imaging Read'
                                    and row.get('channel') == 'Bright Field'
                                    and 'step' in row):
                                mag = row.get('magnification', '')
                                step = row.get('step', '')
                                label = str(mag) if mag else f'_{step}'
                                all_mags.add(label)
                    if all_mags:
                        return all_mags, 'from protocol.csv'
                except (FileNotFoundError, OSError):
                    continue

            # Method 2: parse parent/plate directory names for mag patterns
            # e.g. "4x_10x_20x_40x" in the experiment folder name
            for plate_path in plates[:3]:
                for part in [plate_path, os.path.dirname(plate_path)]:
                    dirname = os.path.basename(part)
                    mag_matches = re.findall(r'(\d+)x', dirname, re.IGNORECASE)
                    if len(mag_matches) >= 2:
                        # Map known magnifications to suffixes
                        # Convention: _01=4x, _02=4x, _03=10x, _04=20x, _05=40x
                        # But we can't know the exact mapping without filenames.
                        # Just report the magnification values found.
                        for m in mag_matches:
                            all_mags.add(f'{m}x')
                        if all_mags:
                            return all_mags, 'from directory name'

            # Method 3: sample filenames from the deepest plate dir
            # Find the actual plate dir (the one with tifs, not an experiment folder)
            for d in dirs_to_check:
                try:
                    names = os.listdir(d)
                except (PermissionError, OSError):
                    continue
                count = 0
                for name in names:
                    if count >= 30:
                        break
                    if not name.endswith('.tif'):
                        continue
                    count += 1
                    m = re.match(r'^[A-P]\d+(_\d+)_', name)
                    if m:
                        all_mags.add(m.group(1))
                if all_mags:
                    return all_mags, f'from filenames in {os.path.basename(d)}'

            return all_mags, 'no magnifications found'

        def _done(result):
            self.detect_mag_btn.setEnabled(True)
            if result is None:
                self.mag_status.setText('Error during scan')
                return

            all_mags, source = result
            self.mag_status.setText(source)

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

        self._run_in_background('magnifications', _scan, _done)

    def _on_bg_result(self, task_name, result):
        """Main-thread handler for background task results."""
        cb = self._bg_callbacks.pop(task_name, None)
        if cb:
            cb(result)

    def _run_in_background(self, task_name, work_fn, done_fn):
        """Run work_fn in a daemon thread, deliver result via Qt signal.

        task_name: unique string key (used to route result to done_fn)
        work_fn: callable, returns result (run in thread)
        done_fn: callable, receives result (run on main thread)
        """
        self._bg_callbacks[task_name] = done_fn

        def _worker():
            try:
                result = work_fn()
            except Exception as e:
                print(f'Background task [{task_name}] error: {e}')
                result = None
            self._bg_result.emit(task_name, result)

        threading.Thread(target=_worker, daemon=True).start()
