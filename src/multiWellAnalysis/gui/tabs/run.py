from PySide6.QtCore import QThread, QObject, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit

from multiWellAnalysis.processing.analysis_main import main as runAnalysis


class Worker(QObject):
    finished = Signal()
    error = Signal(str)

    def run(self):
        try:
            runAnalysis('experiment_config.json')
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class RunTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        layout = QVBoxLayout()

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.runBtn = QPushButton('Run analysis')
        self.runBtn.clicked.connect(self.runPipeline)
        layout.addWidget(self.runBtn)

        self.setLayout(layout)

    def runPipeline(self):
        self.runBtn.setEnabled(False)
        self.log.append('Running analysis...')

        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.onFinished)
        self.worker.error.connect(self.onError)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def onFinished(self):
        self.log.append('Done.')
        self.runBtn.setEnabled(True)

    def onError(self, msg):
        self.log.append(f'Error: {msg}')
        self.runBtn.setEnabled(True)