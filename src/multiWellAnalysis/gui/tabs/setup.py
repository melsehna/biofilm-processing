import os
import re
import glob
import threading
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal


_WELL_TIF_RE = re.compile(r'^[A-P]\d{1,2}[_.]', re.IGNORECASE)

_SKIP_DIRS = {
    'processedimages', 'processed_images_py', 'numerical_data_py',
    'numericaldata', 'plots', '__pycache__', '.git', 'checkpoints',
}

_FILE_EXTS = {
    'tif', 'tiff', 'csv', 'json', 'xlsx', 'xls', 'pdf', 'png',
    'jpg', 'mp4', 'npz', 'npy', 'log', 'txt', 'py', 'r', 'md',
}


def _has_well_tifs(entries):
    """Check if any entry looks like a well TIF file (by name only)."""
    for name in entries:
        if name.lower().endswith(('.tif', '.tiff')) and _WELL_TIF_RE.match(name):
            return True
    return False


def discover_plates(root_dir, max_depth=4):
    """Find plate directories by walking the tree looking for well-named TIFs.

    A plate directory is one that contains TIF files matching well naming
    patterns (e.g. A1_02_Bright Field_1.tif).  Uses only os.listdir
    (no stat calls) for speed over SMB.
    """
    if not root_dir:
        return []

    plates = []
    queue = [(root_dir, 0)]

    while queue:
        path, depth = queue.pop(0)
        try:
            entries = os.listdir(path)
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f'[discover_plates] cannot list {path}: {e}')
            continue

        if _has_well_tifs(entries):
            plates.append(path)
            continue

        if depth >= max_depth:
            continue

        for name in sorted(entries):
            if name.startswith('.') or name.startswith('~$'):
                continue
            if name.lower() in _SKIP_DIRS:
                continue
            # Only skip entries that look like files (have a short known extension)
            # Directory names with dots (e.g. "v1.2_data") should NOT be skipped
            if '.' in name:
                ext = name.rsplit('.', 1)[1].lower()
                if len(ext) <= 5 and ext in _FILE_EXTS:
                    continue
            queue.append((os.path.join(path, name), depth + 1))

    return sorted(plates)


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
        self.root_edit.editingFinished.connect(self._on_root_edited)
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

    def _on_root_edited(self):
        root = self.root_edit.text().strip()
        if root and root != self.state.get('rootDir', ''):
            self.state.set('rootDir', root)
            self._refresh_plates()

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

    def _refresh_plates(self, max_depth=4):
        root = self.root_edit.text().strip()
        self.plate_list.blockSignals(True)
        self.plate_list.clear()
        self.plate_list.blockSignals(False)
        self.refresh_btn.setEnabled(False)
        self.deeper_btn.setEnabled(False)
        self.refresh_btn.setText('Scanning...')

        def _scan():
            return discover_plates(root, max_depth=max_depth), root

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
            if not plates:
                self.mag_status.setText(
                    f'No plates found (no well-named TIF files in {root})'
                )
            self._update_selected_plates()

        self._run_in_background('plates', _scan, _done)

    def _refresh_plates_deeper(self):
        self._refresh_plates(max_depth=6)

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

    # suffix → human-readable magnification
    MAG_SUFFIXES = {'_02': '4x', '_03': '10x', '_04': '20x', '_05': '40x'}

    def _scan_magnifications_async(self, plates):
        """Detect magnifications (background thread).

        One os.listdir call on the first checked plate.  Scan filenames for
        known suffixes (_02=4x, _03=10x, _04=20x, _05=40x).
        """
        def _scan():
            all_mags = set()
            if not plates:
                return all_mags, 'no plates'

            for plate_path in plates[:3]:
                try:
                    names = os.listdir(plate_path)
                except (PermissionError, OSError):
                    continue
                for name in names:
                    if not name.endswith('.tif'):
                        continue
                    m = re.match(r'^[A-P]\d+(_\d+)_', name)
                    if m:
                        suffix = m.group(1)
                        if suffix in self.MAG_SUFFIXES:
                            all_mags.add(suffix)
                    if len(all_mags) == len(self.MAG_SUFFIXES):
                        break  # found all possible magnifications
                if all_mags:
                    return all_mags, f'from filenames in {os.path.basename(plate_path)}'

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
                mag_label = self.MAG_SUFFIXES.get(mag, mag)
                item = QListWidgetItem(f'{mag_label} ({mag})')
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
