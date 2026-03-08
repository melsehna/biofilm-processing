from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLineEdit, QLabel, QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt


class ConditionsTab(QWidget):
    def __init__(self, state, rows=8, cols=12, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)

        # condition name + save
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel('Condition name:'))
        self.nameEdit = QLineEdit()
        self.nameEdit.setPlaceholderText('e.g. WT, hapR, vpsL...')
        name_row.addWidget(self.nameEdit, stretch=1)
        save_btn = QPushButton('Save condition')
        save_btn.clicked.connect(self._save_condition)
        name_row.addWidget(save_btn)
        layout.addLayout(name_row)

        # 96-well grid
        grid = QGridLayout()
        self.buttons = {}
        for r in range(rows):
            for c in range(cols):
                well = f'{chr(65 + r)}{c + 1}'
                btn = QPushButton(well)
                btn.setCheckable(True)
                btn.setFixedSize(48, 32)
                grid.addWidget(btn, r, c)
                self.buttons[well] = btn
        layout.addLayout(grid)

        # saved conditions list
        layout.addWidget(QLabel('Saved conditions:'))
        self.conditions_list = QListWidget()
        self.conditions_list.setMaximumHeight(120)
        layout.addWidget(self.conditions_list)

        # delete button
        del_btn = QPushButton('Delete selected condition')
        del_btn.clicked.connect(self._delete_condition)
        layout.addWidget(del_btn)

        # populate from state
        self._refresh_list()

    def _save_condition(self):
        name = self.nameEdit.text().strip()
        if not name:
            return

        selected = [
            well for well, btn in self.buttons.items()
            if btn.isChecked()
        ]
        if not selected:
            return

        conditions = self.state.get('conditions', {})
        conditions[name] = selected
        self.state.set('conditions', conditions)

        self.nameEdit.clear()
        for btn in self.buttons.values():
            btn.setChecked(False)

        self._refresh_list()

    def _delete_condition(self):
        item = self.conditions_list.currentItem()
        if item is None:
            return
        name = item.text().split(':')[0].strip()
        conditions = self.state.get('conditions', {})
        conditions.pop(name, None)
        self.state.set('conditions', conditions)
        self._refresh_list()

    def _refresh_list(self):
        self.conditions_list.clear()
        conditions = self.state.get('conditions', {})
        for name, wells in conditions.items():
            self.conditions_list.addItem(f'{name}: {", ".join(sorted(wells))}')
