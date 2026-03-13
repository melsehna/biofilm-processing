import numpy as np
import imageio.v3 as iio
import os

def read_images_inplace(ntimepoints, arr, files):
    for t in range(ntimepoints):
        arr[..., t] = iio.imread(files[t]).astype(np.float64)
    return arr

def save_stack(stack, outdir, filename):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{filename}.tif")
    iio.imwrite(path, stack)
