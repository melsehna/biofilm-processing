# preprocessing.py — OpenCV-accelerated version
# Changes from original:
#   - skimage.filters.gaussian  -> cv2.GaussianBlur
#   - scipy.ndimage.uniform_filter -> cv2.blur
#   - float32 throughout (halves memory bandwidth)

import numpy as np
import cv2


def round_odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def normalize_local_contrast(img, block_diameter):
    img = img.astype(np.float32, copy=False)

    img_inv = 1.0 - img

    # cv2.GaussianBlur requires odd kernel size; use 0 to auto-compute from sigma
    blurred = cv2.GaussianBlur(
        img_inv,
        ksize=(0, 0),
        sigmaX=block_diameter,
        borderType=cv2.BORDER_REFLECT
    )
    out = img_inv - blurred

    return out


def preprocess_stack(stack, block_diameter, sigma, gaussian_func):
    """
    Convenience wrapper: ratio-normalize then Gaussian blur.
    """
    h, w, t = stack.shape
    out = np.empty_like(stack, dtype=np.float32)
    for i in range(t):
        r = normalize_local_contrast(stack[..., i], block_diameter)
        out[..., i] = gaussian_func(r, sigma=sigma)
    return out


def mean_filter(X, length_scale):
    size = 2 * length_scale + 1
    # cv2.blur is a box/mean filter, equivalent to uniform_filter
    return cv2.blur(
        X.astype(np.float32, copy=False),
        ksize=(size, size),
        borderType=cv2.BORDER_REFLECT
    )


def normalize_local_contrast_output(images, block_diameter, fpMean):
    length_scale = (block_diameter - 1) // 2
    out = np.empty_like(images, dtype=np.float32)

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
