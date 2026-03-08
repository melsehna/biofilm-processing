import os
import glob
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QLabel, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt


def discover_plates(root_dir):
    """Find plate subdirectories under root_dir."""
    if not root_dir or not os.path.isdir(root_dir):
        return []
    plates = []
    for entry in sorted(os.listdir(root_dir)):
        plate_path = os.path.join(root_dir, entry)
        if not os.path.isdir(plate_path):
            continue
        # check for TIF files (Bright Field pattern) or metadata
        has_tifs = len(glob.glob(os.path.join(plate_path, '*.tif'))) > 0
        has_meta = os.path.exists(os.path.join(plate_path, 'protocol.csv'))
        has_processed = os.path.isdir(os.path.join(plate_path, 'processedImages'))
        has_nested_tifs = len(glob.glob(os.path.join(plate_path, '*', '*.tif'))) > 0
        if has_tifs or has_meta or has_processed or has_nested_tifs:
            plates.append(plate_path)
    return plates


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
