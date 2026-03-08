import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QMessageBox,
)

from .state import AppState
from .tabs.setup import SetupTab
from .tabs.parameters import ParametersTab
from .tabs.preview import PreviewTab
from .tabs.conditions import ConditionsTab
from .tabs.run import RunTab


CONFIG_PATH = Path('experiment_config.json')


class PhenotyprApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Phenotypr')
        self.resize(1000, 750)

        self.state = AppState()

        # load existing config if present
        if CONFIG_PATH.exists():
            try:
                self.state.load(str(CONFIG_PATH))
            except Exception:
                pass

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

    def _save_config(self):
        try:
            self.state.save(str(CONFIG_PATH))
            QMessageBox.information(self, 'Saved', f'Configuration saved to {CONFIG_PATH}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def _load_config(self):
        try:
            self.state.load(str(CONFIG_PATH))
            QMessageBox.information(self, 'Loaded', f'Configuration loaded from {CONFIG_PATH}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


def run():
    app = QApplication(sys.argv)
    win = PhenotyprApp()
    win.show()
    sys.exit(app.exec())
