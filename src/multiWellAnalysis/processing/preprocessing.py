# multiWellAnalysis/preprocessing.py

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.exposure import rescale_intensity

# from analysis_pipeline.preprocessing import mean_filter

def round_odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1

def _safe_uniform_mean(img, size):
    # uniform_filter preserves dtype; add epsilon to avoid division by 0
    mean_local = uniform_filter(img, size=size, mode="reflect")
    return mean_local

from skimage.filters import gaussian

def normalize_local_contrast(img, block_diameter):
    img = img.astype(np.float64, copy=False)

    img_inv = 1.0 - img
    blurred = gaussian(img_inv, sigma=block_diameter, preserve_range=True)
    out = img_inv - blurred

    return out


def preprocess_stack(stack, block_diameter, sigma, gaussian_func):
    """
    Convenience wrapper (if you need it): ratio-normalize then Gaussian blur.
    """
    h, w, t = stack.shape
    out = np.empty_like(stack, dtype=np.float64)
    for i in range(t):
        r = normalize_local_contrast(stack[..., i], block_diameter)
        out[..., i] = gaussian_func(r, sigma=sigma)
    return out


import numpy as np

# def mean_filter(X, length_scale):
#     H, W = X.shape
#     I = np.zeros((H+1, W+1), dtype=np.float64)
#     I[1:, 1:] = np.cumsum(np.cumsum(X, axis=0), axis=1)

#     out = np.zeros_like(X)

#     for i in range(H):
#         x0 = max(i - length_scale, 0)
#         x1 = min(i + length_scale + 1, H)

#         for j in range(W):
#             y0 = max(j - length_scale, 0)
#             y1 = min(j + length_scale + 1, W)

#             area = (x1 - x0) * (y1 - y0)
#             out[i, j] = (
#                 I[x1, y1]
#                 - I[x0, y1]
#                 - I[x1, y0]
#                 + I[x0, y0]
#             ) / area

#     return out



def mean_filter(X, length_scale):
    size = 2 * length_scale + 1
    return uniform_filter(X, size=size, mode='reflect')



def normalize_local_contrast_output(images, block_diameter, fpMean):
    length_scale = (block_diameter - 1) // 2
    out = np.empty_like(images)

    if images.ndim == 3:
        for t in range(images.shape[2]):
            img = images[..., t]
            mean_img = mean_filter(img, length_scale)
            tmp = img - mean_img + fpMean
            out[..., t] = np.clip(tmp, 0.0, 1.0)
    else:
        mean_img = mean_filter(images, length_scale)
        out = np.clip(images - mean_img + fpMean, 0.0, 1.0)

    return out
