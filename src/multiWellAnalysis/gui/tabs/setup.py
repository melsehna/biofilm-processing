import os
import re
import threading
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
    QInputDialog,
)
from PySide6.QtCore import Qt, Signal


# Directories that contain pipeline outputs, never raw images.
_OUTPUT_DIR_NAMES = {
    'processedimages', 'processed_images', 'processed_images_py',
    'numerical_data', 'numerical_data_py',
    'results_images', 'results_data',
}


def _detect_mag_suffixes_from_tifs(root, suffixes, max_depth=2):
    """Check which magnification suffixes appear in TIF filenames under *root*.

    For each suffix in *suffixes* (e.g. {'_02', '_03', '_04', '_05'}), scans
    directories up to *max_depth* levels deep and stops checking a suffix as
    soon as one matching file is found.  Skips output directories.
    Uses os.listdir to avoid incomplete results from macOS SMB caching.
    Returns the set of suffixes found.
    """
    remaining = set(suffixes)
    found = set()

    def _scan_dir(directory, depth):
        nonlocal remaining
        try:
            names = os.listdir(directory)
        except (PermissionError, OSError):
            return
        # Check files at this level
        for name in names:
            if not remaining:
                return
            if name.lower().endswith('.tif'):
                for suffix in list(remaining):
                    if suffix in name:
                        found.add(suffix)
                        remaining.discard(suffix)
                        if not remaining:
                            return
        # Recurse into subdirectories (skip output dirs)
        if depth < max_depth:
            for name in names:
                if not remaining:
                    return
                child = os.path.join(directory, name)
                if not name.startswith('.') \
                        and name.lower() not in _OUTPUT_DIR_NAMES \
                        and os.path.isdir(child):
                    _scan_dir(child, depth + 1)

    _scan_dir(root, 0)
    return found


class SetupTab(QWidget):
    _bg_result = Signal(str, object)  # (task_name, result)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._bg_callbacks = {}
        self._bg_result.connect(self._on_bg_result)
        self.state = state
        self._build_ui()
        self._connect_signals()

        # restore plates from state
        for p in self.state.get('plates', []):
            self._add_plate_item(p)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # plate folders — user selects directly (like Julia GUI)
        plates_label = QLabel('Plate folders (containing images):')
        layout.addWidget(plates_label)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton('Add plate folders...')
        btn_row.addWidget(self.add_btn)
        self.add_parent_btn = QPushButton('Add from parent folder...')
        self.add_parent_btn.setToolTip(
            'Select a root directory and add all its sub-folders as plates\n'
            '(e.g. select the experiment folder containing multiple drawer directories)'
        )
        btn_row.addWidget(self.add_parent_btn)
        self.paste_btn = QPushButton('Paste path...')
        self.paste_btn.setToolTip('Type or paste a path to a plate folder (useful for network mounts)')
        btn_row.addWidget(self.paste_btn)
        self.remove_btn = QPushButton('Remove selected')
        self.remove_btn.setEnabled(False)
        btn_row.addWidget(self.remove_btn)
        self.clear_btn = QPushButton('Clear all')
        self.clear_btn.setEnabled(False)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.plate_list = QListWidget()
        self.plate_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.plate_list, stretch=1)

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
        self.add_btn.clicked.connect(self._add_plates)
        self.add_parent_btn.clicked.connect(self._add_from_parent)
        self.paste_btn.clicked.connect(self._paste_path)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear_all)
        self.plate_list.itemSelectionChanged.connect(self._update_remove_btn)
        self.outdir_browse_btn.clicked.connect(self._browse_outdir)
        self.outdir_edit.editingFinished.connect(
            lambda: self.state.set('outputDir', self.outdir_edit.text().strip())
        )
        self.detect_mag_btn.clicked.connect(self._on_detect_mag_clicked)
        self.mag_list.itemChanged.connect(self._on_mag_changed)
        self.notes_edit.textChanged.connect(
            lambda: self.state.set('notes', self.notes_edit.toPlainText())
        )

    def _add_plates(self):
        """Add plate folders using the native dialog.

        The native macOS dialog only supports single-folder selection,
        so we loop until the user cancels, allowing them to add plates
        one at a time.  This preserves access to SMB/network mounts
        that the non-native Qt dialog cannot see.
        """
        existing = {
            self.plate_list.item(i).data(Qt.UserRole)
            for i in range(self.plate_list.count())
        }
        added_any = False
        while True:
            start = self.state.get('lastBrowseDir', '')
            path = QFileDialog.getExistingDirectory(
                self, 'Select a plate folder (Cancel when done)', start,
            )
            if not path:
                break  # user cancelled — done adding
            self.state.set('lastBrowseDir', os.path.dirname(path))
            if path not in existing:
                self._add_plate_item(path)
                existing.add(path)
                added_any = True
        if added_any:
            self._sync_state()

    def _add_from_parent(self):
        """Select a parent directory and add all its sub-folders as plates.

        Useful for experiments stored as:
            root/
              drawer1/
                plate1/   ← images here
              drawer2/
                plate1/   ← images here
        The user picks 'root' and every immediate child directory is added.
        """
        start = self.state.get('lastBrowseDir', '')
        parent = QFileDialog.getExistingDirectory(
            self, 'Select parent folder containing plate directories', start,
        )
        if not parent:
            return
        self.state.set('lastBrowseDir', parent)
        existing = {
            self.plate_list.item(i).data(Qt.UserRole)
            for i in range(self.plate_list.count())
        }
        added_any = False
        try:
            children = sorted(os.listdir(parent))
        except (PermissionError, OSError):
            return
        for name in children:
            path = os.path.join(parent, name)
            if os.path.isdir(path) and not name.startswith('.') and path not in existing:
                self._add_plate_item(path)
                existing.add(path)
                added_any = True
        if added_any:
            self._sync_state()

    def _paste_path(self):
        """Add a plate folder by typing or pasting a path directly."""
        text, ok = QInputDialog.getText(
            self, 'Paste plate folder path',
            'Enter full path to plate folder:',
        )
        if not ok or not text.strip():
            return
        path = os.path.expanduser(text.strip())
        if not os.path.isdir(path):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Not found', f'Directory not found:\n{path}')
            return
        existing = {
            self.plate_list.item(i).data(Qt.UserRole)
            for i in range(self.plate_list.count())
        }
        if path not in existing:
            self._add_plate_item(path)
            self._sync_state()

    def _add_plate_item(self, path):
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.plate_list.addItem(item)
        self.clear_btn.setEnabled(True)

    def _remove_selected(self):
        for item in self.plate_list.selectedItems():
            self.plate_list.takeItem(self.plate_list.row(item))
        self._sync_state()
        self._update_remove_btn()

    def _clear_all(self):
        self.plate_list.clear()
        self.clear_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self._sync_state()

    def _update_remove_btn(self):
        self.remove_btn.setEnabled(len(self.plate_list.selectedItems()) > 0)

    def _sync_state(self):
        plates = [
            self.plate_list.item(i).data(Qt.UserRole)
            for i in range(self.plate_list.count())
        ]
        self.state.set('plates', plates)
        self.clear_btn.setEnabled(len(plates) > 0)

    def _browse_outdir(self):
        start = self.state.get('outputDir', '') or self.state.get('lastBrowseDir', '')
        path = QFileDialog.getExistingDirectory(self, 'Select Output Directory', start)
        if path:
            self.outdir_edit.setText(path)
            self.state.set('outputDir', path)

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

    # suffix → human-readable magnification (and reverse)
    MAG_SUFFIXES = {'_02': '4x', '_03': '10x', '_04': '20x', '_05': '40x'}
    _MAG_LABEL_TO_SUFFIX = {v: k for k, v in MAG_SUFFIXES.items()}  # '4x' → '_02'

    def _scan_magnifications_async(self, plates):
        """Detect magnifications (background thread).

        First tries to parse magnifications from directory names (instant,
        no I/O into subdirs).  Falls back to scanning a sample of TIF
        filenames if directory names don't contain magnification labels.
        """
        def _scan():
            all_mags = set()
            if not plates:
                return all_mags, 'no plates'

            # --- Fast path: parse directory names, filenames, and subdir names ---
            for plate_path in plates[:3]:
                dirname = os.path.basename(plate_path)
                for label, suffix in self._MAG_LABEL_TO_SUFFIX.items():
                    if re.search(rf'(?<![a-zA-Z0-9]){re.escape(label)}(?![a-zA-Z0-9])', dirname):
                        all_mags.add(suffix)
                # also check names of files/subdirs inside (CSV names, plate subdir names)
                try:
                    for name in os.listdir(plate_path):
                        for label, suffix in self._MAG_LABEL_TO_SUFFIX.items():
                            if re.search(rf'(?<![a-zA-Z0-9]){re.escape(label)}(?![a-zA-Z0-9])', name):
                                all_mags.add(suffix)
                except (PermissionError, OSError):
                    pass
            if all_mags:
                return all_mags, 'from directory names'

            # --- Slow path: check first TIF filename per suffix ---
            for plate_path in plates[:1]:  # only scan first plate over SMB
                found = _detect_mag_suffixes_from_tifs(
                    plate_path, set(self.MAG_SUFFIXES), max_depth=2,
                )
                if found:
                    return found, f'from filenames in {os.path.basename(plate_path)}'

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

    def refreshFromState(self):
        """Sync all widgets to current state (call after loading a config)."""
        # Plate list
        self.plate_list.blockSignals(True)
        self.plate_list.clear()
        for p in self.state.get('plates', []):
            self._add_plate_item(p)
        self.plate_list.blockSignals(False)
        self.clear_btn.setEnabled(self.plate_list.count() > 0)

        # Output dir
        self.outdir_edit.blockSignals(True)
        self.outdir_edit.setText(self.state.get('outputDir', ''))
        self.outdir_edit.blockSignals(False)

        # Notes
        self.notes_edit.blockSignals(True)
        self.notes_edit.setPlainText(self.state.get('notes', ''))
        self.notes_edit.blockSignals(False)

        # Mag list: populate from saved magnification + magParams keys
        mag_setting = self.state.get('magnification', 'all')
        if isinstance(mag_setting, list):
            selected_mags = set(mag_setting)
        elif isinstance(mag_setting, str) and mag_setting != 'all':
            selected_mags = {mag_setting}
        else:
            selected_mags = set()

        all_mags = set(selected_mags)
        for m in self.state.get('magParams', {}):
            all_mags.add(m)

        if all_mags:
            self.mag_list.blockSignals(True)
            self.mag_list.clear()
            for mag in sorted(all_mags):
                mag_label = self.MAG_SUFFIXES.get(mag, mag)
                item = QListWidgetItem(f'{mag_label} ({mag})')
                item.setData(Qt.UserRole, mag)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                is_checked = mag in selected_mags or mag_setting == 'all'
                item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
                self.mag_list.addItem(item)
            self.mag_list.blockSignals(False)
            self.mag_status.setText('loaded from config')

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
