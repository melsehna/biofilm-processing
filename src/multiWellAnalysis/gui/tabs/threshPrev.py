from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import imageio.v3 as iio
import numpy as np
from scipy.ndimage import gaussian_filter

from multiWellAnalysis.processing.preprocessing import normalizeLocalContrast
class PreviewTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        layout = QVBoxLayout()

        self.canvas = FigureCanvasQTAgg(Figure())
        self.ax = self.canvas.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        btn = QPushButton('Load test image')
        btn.clicked.connect(self.loadImage)
        layout.addWidget(btn)

        self.setLayout(layout)

    def loadImage(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select image')
        if not path:
            return

        img = iio.imread(path).astype(np.float64)
        img /= img.max()

        block = self.state.get('blockDiam', 101)
        thresh = self.state.get('fixedThresh', 0.04)

        norm = normalize_local_contrast(img, block)
        blurred = gaussian_filter(norm, sigma=2.0)
        mask = blurred > thresh

        overlay = np.stack([img, img, img], axis=-1)
        overlay[mask] = [0, 1, 1]

        self.ax.clear()
        self.ax.imshow(overlay)
        self.canvas.draw()