import sys
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QMessageBox, QFileDialog,
)

from .state import AppState
from .tabs.setup import SetupTab
from .tabs.parameters import ParametersTab
from .tabs.preview import PreviewTab
from .tabs.conditions import ConditionsTab
from .tabs.runGUI import RunTab


CONFIG_FILENAME = 'experiment_config.json'


class PhenotyprApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Phenotypr')
        self.resize(1000, 750)

        self.state = AppState()

        # tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(SetupTab(self.state), 'Setup')
        self.tabs.addTab(ParametersTab(self.state), 'Parameters')
        self.tabs.addTab(PreviewTab(self.state), 'Preview')
        self.tabs.addTab(ConditionsTab(self.state), 'Conditions')
        self.tabs.addTab(RunTab(self.state), 'Run')

        # layout
        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)

        btn_row = QHBoxLayout()
        save_btn = QPushButton('Save configuration')
        save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(save_btn)

        load_btn = QPushButton('Load configuration')
        load_btn.clicked.connect(self._load_config)
        btn_row.addWidget(load_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        container.setLayout(layout)
        self.setCentralWidget(container)

    def _config_path(self):
        """Config saves to rootDir if set, otherwise CWD."""
        root = self.state.get('rootDir', '')
        if root and os.path.isdir(root):
            return os.path.join(root, CONFIG_FILENAME)
        return CONFIG_FILENAME

    def _save_config(self):
        path = self._config_path()
        try:
            self.state.save(path)
            QMessageBox.information(self, 'Saved', f'Configuration saved to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def _load_config(self):
        # try rootDir first, then let user browse
        path = self._config_path()
        if not os.path.exists(path):
            path, _ = QFileDialog.getOpenFileName(
                self, 'Load Configuration', '',
                'JSON files (*.json);;All files (*)'
            )
            if not path:
                return
        try:
            self.state.load(path)
            QMessageBox.information(self, 'Loaded', f'Configuration loaded from {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


def run():
    app = QApplication(sys.argv)
    win = PhenotyprApp()
    win.show()
    sys.exit(app.exec())
