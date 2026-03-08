from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout,
    QPushButton, QLineEdit, QLabel
)


class ConditionsTab(QWidget):
    def __init__(self, state, rows=8, cols=12):
        super().__init__()
        self.state = state

        layout = QVBoxLayout()

        layout.addWidget(QLabel('Condition name'))
        self.nameEdit = QLineEdit()
        layout.addWidget(self.nameEdit)

        grid = QGridLayout()
        self.buttons = {}

        for r in range(rows):
            for c in range(cols):
                well = f'{chr(65 + r)}{c + 1}'
                btn = QPushButton(well)
                btn.setCheckable(True)
                grid.addWidget(btn, r, c)
                self.buttons[well] = btn

        layout.addLayout(grid)

        saveBtn = QPushButton('Save condition')
        saveBtn.clicked.connect(self.saveCondition)
        layout.addWidget(saveBtn)

        self.setLayout(layout)

    def saveCondition(self):
        name = self.nameEdit.text().strip()
        if not name:
            return

        selected = [
            well for well, btn in self.buttons.items()
            if btn.isChecked()
        ]

        self.state.setdefault('conditions', {})
        self.state['conditions'][name] = selected

        self.nameEdit.clear()
        for btn in self.buttons.values():
            btn.setChecked(False)