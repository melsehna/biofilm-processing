import numpy as np
import cv2
import imageio.v3 as iio
import os
from concurrent.futures import ThreadPoolExecutor


def _read_one(args):
    arr, t, path = args
    # cv2.imread can fail on paths with special characters; use imdecode as fallback
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    arr[..., t] = img.astype(np.float32)


def read_images_inplace(ntimepoints, arr, files):
    # Threaded I/O — TIFF decoding releases the GIL
    with ThreadPoolExecutor() as pool:
        pool.map(_read_one, [(arr, t, files[t]) for t in range(ntimepoints)])
    return arr


def save_stack(stack, outdir, filename):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{filename}.tif")
    iio.imwrite(path, stack.astype(np.float32, copy=False))
