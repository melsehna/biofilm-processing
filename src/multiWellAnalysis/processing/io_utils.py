import numpy as np
import imageio.v3 as iio
import os
from concurrent.futures import ThreadPoolExecutor


def _read_one(args):
    arr, t, path = args
    arr[..., t] = iio.imread(path).astype(np.float64)


def read_images_inplace(ntimepoints, arr, files):
    # Threaded I/O — TIFF decoding releases the GIL
    with ThreadPoolExecutor() as pool:
        pool.map(_read_one, [(arr, t, files[t]) for t in range(ntimepoints)])
    return arr


def save_stack(stack, outdir, filename):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{filename}.tif")
    import tifffile
    # Save as (T, H, W) multi-page TIF — tifffile expects pages first
    data = stack.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[2] < data.shape[0]:
        # (H, W, T) → (T, H, W)
        data = np.transpose(data, (2, 0, 1))
    tifffile.imwrite(path, data)
