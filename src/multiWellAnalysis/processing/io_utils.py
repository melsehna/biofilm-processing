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
    iio.imwrite(path, stack.astype(np.float32, copy=False))
