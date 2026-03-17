import json
from PySide6.QtCore import QObject, Signal


DEFAULTS = {
    'rootDir':          '',
    'plates':           [],
    'blockDiam':        101,
    'fixedThresh':      0.04,
    'shiftThresh':      50,
    'fftStride':        6,
    'downsample':       4,
    'dustCorrection':   True,
    'saveRegistered':   True,
    'saveProcessed':    True,
    'saveMasks':        True,
    'saveOverlays':     True,
    'wholeImageFeats':  False,
    'colonyTracking':   False,
    'colonyFeats':      False,
    'minColonyAreaPx':  200,
    'propRadiusPx':     25,
    'conditions':       {},
    'notes':            '',
    'magnification':    'all',
    'workers':          4,
}


class AppState(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = dict(DEFAULTS)

    def set(self, key, value):
        self._data[key] = value
        self.changed.emit()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def to_dict(self):
        return dict(self._data)

    def from_dict(self, d):
        self._data.update(d)
        self.changed.emit()

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            self.from_dict(json.load(f))
