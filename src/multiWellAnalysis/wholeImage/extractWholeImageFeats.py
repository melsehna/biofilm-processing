#!/usr/bin/env python3
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

import mahotas


def fractalDimension(z, threshold=0.9):

    z = (z < threshold * z.max()).astype(np.uint8)

    p = min(z.shape)
    n = 2 ** int(np.floor(np.log2(p)))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)

    def boxCount(arr, k):
        s = np.add.reduceat(
            np.add.reduceat(arr, np.arange(0, arr.shape[0], k), axis=0),
            np.arange(0, arr.shape[1], k), axis=1
        )
        return np.sum((s > 0) & (s < k * k))

    counts = [boxCount(z, size) for size in sizes]
    return -np.polyfit(np.log(sizes), np.log(counts), 1)[0]


def extractFrameFeats(img):

    if img.ndim == 3:
        img = rgb2gray(img)

    img = img_as_ubyte(img)
    imgFloat = img.astype(float)

    h, w = img.shape
    if h < 32 or w < 32:
        raise ValueError(f'image too small: {h}x{w}')

    feats = {}

    feats['meanIntensity'] = imgFloat.mean()
    feats['stdIntensity'] = imgFloat.std()
    feats['medianIntensity'] = np.median(imgFloat)
    feats['madIntensity'] = np.median(np.abs(imgFloat - feats['medianIntensity']))
    feats['iqrIntensity'] = np.percentile(imgFloat, 75) - np.percentile(imgFloat, 25)

    flat = imgFloat.ravel()
    feats['skew'] = skew(flat)
    feats['kurtosis'] = kurtosis(flat)

    for p in [1, 5, 25, 50, 75, 95, 99]:
        feats[f'p{p}'] = np.percentile(imgFloat, p)

    feats['entropy'] = shannon_entropy(img)

    haralick = mahotas.features.haralick(img, return_mean=True)
    for i, value in enumerate(haralick):
        feats[f'haralick_{i}'] = float(value)

    return feats