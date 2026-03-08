import sys
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QTabWidget, QPushButton,
    QMessageBox
)

from .tabs.config import ConfigTab
from .tabs.threshPrev import PreviewTab
from .tabs.conditions import ConditionsTab
from .tabs.run import RunTab


configPath = Path('experiment_config.json')


class PhenotyprApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Phenotypr')
        self.resize(900, 700)

        self.state = {}

        self.tabs = QTabWidget()

        self.configTab = ConfigTab(self.state)
        self.previewTab = PreviewTab(self.state)
        self.conditionsTab = ConditionsTab(self.state)
        self.runTab = RunTab(self.state)

        self.tabs.addTab(self.configTab, 'Configuration')
        self.tabs.addTab(self.previewTab, 'Threshold preview')
        self.tabs.addTab(self.conditionsTab, 'Conditions')
        self.tabs.addTab(self.runTab, 'Run')

        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)

        saveBtn = QPushButton('Save configuration')
        saveBtn.clicked.connect(self.saveConfig)
        layout.addWidget(saveBtn)

        container.setLayout(layout)
        self.setCentralWidget(container)

    def saveConfig(self):
        try:
            with open(configPath, 'w') as f:
                json.dump(self.state, f, indent=4)
            QMessageBox.information(self, 'Saved', 'Configuration saved.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


def run():
    app = QApplication(sys.argv)
    win = PhenotyprApp()
    win.show()
    sys.exit(app.exec())