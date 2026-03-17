import os
import re
import glob
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt


def discover_plates(root_dir):
    """Find candidate plate directories under root_dir.

    Strategy: list immediate subdirectories only (one scandir call).
    If any child has no subdirectories of its own, it's likely a plate.
    If all children have subdirectories, they're experiment folders —
    list THEIR children as plate candidates instead (two scandir calls).

    Never scans file contents. Fast over SMB/NFS.
    If root_dir itself has no subdirectories, treat it as a single plate.
    """
    if not root_dir or not os.path.isdir(root_dir):
        return []

    children = _list_subdirs(root_dir)

    # No subdirectories — root itself is likely a plate
    if not children:
        return [root_dir]

    # Check if children are plates (no subdirs) or experiment folders (have subdirs)
    # Only check the first child to decide — avoids scanning all of them
    first_child_subdirs = _list_subdirs(children[0])

    if not first_child_subdirs:
        # Children are leaf dirs (plates). Example:
        #   root/Plate1/  root/Plate2/
        return sorted(children)

    # Children are experiment folders. Go one level deeper. Example:
    #   root/Experiment_A/Plate1/  root/Experiment_A/Plate2/
    plates = []
    for child in children:
        plates.extend(_list_subdirs(child))
    return sorted(plates)


def _list_subdirs(path):
    """List immediate subdirectories of path. Single scandir call."""
    try:
        return sorted(
            e.path for e in os.scandir(path)
            if e.is_dir() and not e.name.startswith('.')
        )
    except PermissionError:
        return []


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
            # Show relative path from root so nested plates are distinguishable
            try:
                display = os.path.relpath(p, root)
            except ValueError:
                display = os.path.basename(p)
            item = QListWidgetItem(display)
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
        """Scan selected plates for available magnifications.

        Uses os.scandir and stops after finding enough samples — avoids
        listing thousands of files over SMB.
        """
        all_mags = set()
        for plate_path in plates[:1]:  # only scan first plate
            try:
                count = 0
                for entry in os.scandir(plate_path):
                    if count > 200:  # sample first 200 entries max
                        break
                    if not entry.is_file() or not entry.name.endswith('.tif'):
                        count += 1
                        continue
                    if 'Bright Field' in entry.name or 'Bright_Field' in entry.name:
                        m = re.match(r'^[A-H]\d+(_\d+)_', entry.name)
                        if m:
                            all_mags.add(m.group(1))
                    count += 1
            except PermissionError:
                continue

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
