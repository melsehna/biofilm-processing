from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QPushButton, QListWidget,
    QFileDialog, QSpinBox,
    QDoubleSpinBox, QCheckBox,
    QTextEdit
)


class ConfigTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        layout = QVBoxLayout()

        layout.addWidget(QLabel('Image directories'))
        self.dirList = QListWidget()
        layout.addWidget(self.dirList)

        addDirBtn = QPushButton('Add directory')
        addDirBtn.clicked.connect(self.addDirectory)
        layout.addWidget(addDirBtn)

        layout.addWidget(QLabel('Threshold'))
        self.threshSpin = QDoubleSpinBox()
        self.threshSpin.setRange(0.0, 1.0)
        self.threshSpin.setDecimals(4)
        self.threshSpin.setValue(0.04)
        self.threshSpin.valueChanged.connect(self.updateState)
        layout.addWidget(self.threshSpin)

        layout.addWidget(QLabel('Block diameter'))
        self.blockSpin = QSpinBox()
        self.blockSpin.setRange(11, 501)
        self.blockSpin.setValue(101)
        self.blockSpin.valueChanged.connect(self.updateState)
        layout.addWidget(self.blockSpin)

        self.dustCheck = QCheckBox('Dust correction')
        self.dustCheck.stateChanged.connect(self.updateState)
        layout.addWidget(self.dustCheck)

        layout.addWidget(QLabel('Notes'))
        self.notesEdit = QTextEdit()
        self.notesEdit.textChanged.connect(self.updateState)
        layout.addWidget(self.notesEdit)

        self.setLayout(layout)
        self.updateState()

    def addDirectory(self):
        path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if path:
            self.dirList.addItem(path)
            self.updateState()

    def updateState(self):
        self.state['imagesDirectory'] = [
            self.dirList.item(i).text()
            for i in range(self.dirList.count())
        ]
        self.state['fixedThresh'] = self.threshSpin.value()
        self.state['blockDiam'] = self.blockSpin.value()
        self.state['dustCorrection'] = self.dustCheck.isChecked()
        self.state['notes'] = self.notesEdit.toPlainText()