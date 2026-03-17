import os
import re
import glob
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt


def _has_well_tifs(path):
    """Fast check: does this directory contain at least one well-like .tif?

    Looks for files matching A1*.tif through P24*.tif (well ID prefix).
    Returns on the first match instead of globbing all files.
    """
    try:
        for f in os.scandir(path):
            if not f.is_file() or not f.name.endswith('.tif'):
                continue
            if re.match(r'^[A-P]\d{1,2}[_.]', f.name):
                return True
    except PermissionError:
        pass
    return False


def _is_plate_dir(path):
    """Check if a directory looks like a plate with raw well images."""
    return _has_well_tifs(path)


def discover_plates(root_dir, max_depth=3):
    """Find plate subdirectories under root_dir, searching up to max_depth levels.

    Handles structures like:
        root/plate/images.tif              (depth 1)
        root/experiment/plate/images.tif   (depth 2)
        root/group/experiment/plate/...    (depth 3)

    Also treats root_dir itself as a plate if it directly contains tifs.
    """
    if not root_dir or not os.path.isdir(root_dir):
        return []

    if _is_plate_dir(root_dir):
        return [root_dir]

    plates = []
    _discover_recursive(root_dir, plates, depth=0, max_depth=max_depth)
    return sorted(plates)


def _discover_recursive(path, plates, depth, max_depth):
    if depth >= max_depth:
        return
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name)
    except PermissionError:
        return
    for entry in entries:
        if not entry.is_dir():
            continue
        if _is_plate_dir(entry.path):
            plates.append(entry.path)
        else:
            _discover_recursive(entry.path, plates, depth + 1, max_depth)


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

    def _refresh_plates(self):
        root = self.root_edit.text().strip()
        self.plate_list.blockSignals(True)
        self.plate_list.clear()

        plates = discover_plates(root)
        for p in plates:
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.UserRole, p)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.plate_list.addItem(item)

        self.plate_list.blockSignals(False)
        self._update_selected_plates()

    def _update_selected_plates(self):
        selected = []
        for i in range(self.plate_list.count()):
            item = self.plate_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        self.state.set('plates', selected)
        self._scan_magnifications(selected)

    def _on_mag_changed(self, item=None):
        """Update state with currently checked magnifications."""
        selected = []
        for i in range(self.mag_list.count()):
            it = self.mag_list.item(i)
            if it.checkState() == Qt.Checked:
                selected.append(it.data(Qt.UserRole))
        # 'all' if nothing checked or all checked
        total = self.mag_list.count()
        if not selected or len(selected) == total:
            self.state.set('magnification', 'all')
        elif len(selected) == 1:
            self.state.set('magnification', selected[0])
        else:
            self.state.set('magnification', selected)

    def _scan_magnifications(self, plates):
        """Scan selected plates for available magnifications."""
        all_mags = set()
        for plate_path in plates[:3]:
            tif_files = sorted(glob.glob(os.path.join(plate_path, '*.tif')))
            bf_files = [f for f in tif_files
                        if 'Bright Field' in f or 'Bright_Field' in f]
            if not bf_files:
                continue
            for f in bf_files:
                m = re.match(r'^[A-H]\d+(_\d+)_', os.path.basename(f))
                if m:
                    all_mags.add(m.group(1))

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
