from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLineEdit, QLabel, QListWidget,
)
from PySide6.QtCore import Qt, QEvent


class WellButton(QPushButton):
    """Well button that supports click-drag painting."""
    pass


class ConditionsTab(QWidget):
    def __init__(self, state, rows=8, cols=12, parent=None):
        super().__init__(parent)
        self.state = state
        self.rows = rows
        self.cols = cols
        self._painting = False
        self._paint_state = True

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

        # select all / clear row
        action_row = QHBoxLayout()
        select_all_btn = QPushButton('Select all')
        select_all_btn.clicked.connect(self._select_all)
        action_row.addWidget(select_all_btn)
        clear_btn = QPushButton('Clear selection')
        clear_btn.clicked.connect(self._clear_selection)
        action_row.addWidget(clear_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        # 96-well grid with row/column headers
        grid = QGridLayout()
        grid.setSpacing(2)

        # column headers (1-12)
        self.col_btns = {}
        for c in range(cols):
            btn = QPushButton(str(c + 1))
            btn.setFixedSize(48, 24)
            btn.setStyleSheet('font-weight: bold; font-size: 11px;')
            btn.clicked.connect(lambda checked, col=c: self._toggle_column(col))
            grid.addWidget(btn, 0, c + 1)
            self.col_btns[c] = btn

        # row headers (A-H) + well buttons
        self.row_btns = {}
        self.buttons = {}
        for r in range(rows):
            row_letter = chr(65 + r)
            rbtn = QPushButton(row_letter)
            rbtn.setFixedSize(28, 32)
            rbtn.setStyleSheet('font-weight: bold; font-size: 11px;')
            rbtn.clicked.connect(lambda checked, row=r: self._toggle_row(row))
            grid.addWidget(rbtn, r + 1, 0)
            self.row_btns[r] = rbtn

            for c in range(cols):
                well = f'{row_letter}{c + 1}'
                btn = WellButton(well)
                btn.setCheckable(True)
                btn.setFixedSize(48, 32)
                btn.installEventFilter(self)
                grid.addWidget(btn, r + 1, c + 1)
                self.buttons[well] = btn

        layout.addLayout(grid)

        # hint
        hint = QLabel('Click wells to select. Click row/column headers to select entire rows/columns. Click and drag to paint.')
        hint.setStyleSheet('color: gray; font-size: 11px;')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # saved conditions list
        layout.addWidget(QLabel('Saved conditions:'))
        self.conditions_list = QListWidget()
        self.conditions_list.setMaximumHeight(120)
        self.conditions_list.itemClicked.connect(self._highlight_condition)
        layout.addWidget(self.conditions_list)

        # delete button
        del_btn = QPushButton('Delete selected condition')
        del_btn.clicked.connect(self._delete_condition)
        layout.addWidget(del_btn)

        # populate from state
        self._refresh_list()

    def eventFilter(self, obj, event):
        """Enable click-drag painting across well buttons."""
        if isinstance(obj, WellButton):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._painting = True
                self._paint_state = not obj.isChecked()
                obj.setChecked(self._paint_state)
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._painting = False
                return True
            elif event.type() == QEvent.Enter and self._painting:
                obj.setChecked(self._paint_state)
                return True
        return super().eventFilter(obj, event)

    def _toggle_row(self, row):
        row_letter = chr(65 + row)
        wells = [f'{row_letter}{c + 1}' for c in range(self.cols)]
        any_unchecked = any(not self.buttons[w].isChecked() for w in wells)
        for w in wells:
            self.buttons[w].setChecked(any_unchecked)

    def _toggle_column(self, col):
        wells = [f'{chr(65 + r)}{col + 1}' for r in range(self.rows)]
        any_unchecked = any(not self.buttons[w].isChecked() for w in wells)
        for w in wells:
            self.buttons[w].setChecked(any_unchecked)

    def _select_all(self):
        for btn in self.buttons.values():
            btn.setChecked(True)

    def _clear_selection(self):
        for btn in self.buttons.values():
            btn.setChecked(False)

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

    def _highlight_condition(self, item):
        """When clicking a saved condition, highlight its wells on the grid."""
        name = item.text().split(':')[0].strip()
        conditions = self.state.get('conditions', {})
        wells = conditions.get(name, [])
        for well, btn in self.buttons.items():
            btn.setChecked(well in wells)

    def _refresh_list(self):
        self.conditions_list.clear()
        conditions = self.state.get('conditions', {})
        for name, wells in conditions.items():
            self.conditions_list.addItem(f'{name}: {", ".join(sorted(wells))}')
