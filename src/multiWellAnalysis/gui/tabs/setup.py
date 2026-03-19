import os
import re
import threading
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal


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
        dlg = QFileDialog(self, 'Select plate folder(s)')
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        # Enable multi-select via list view
        from PySide6.QtWidgets import QListView, QTreeView, QAbstractItemView
        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if dlg.exec():
            existing = {
                self.plate_list.item(i).data(Qt.UserRole)
                for i in range(self.plate_list.count())
            }
            for path in dlg.selectedFiles():
                if path not in existing:
                    self._add_plate_item(path)
                    existing.add(path)
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
        path = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
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
