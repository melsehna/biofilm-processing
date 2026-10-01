import numpy as np
import imageio.v3 as iio
import os
from concurrent.futures import ThreadPoolExecutor


def _readOne(args):
    arr, t, path = args
    arr[..., t] = iio.imread(path).astype(np.float64)


def readWellStack(wellFiles):
    """Read one well's raw frames into an (H, W, T) stack in the source dtype.

    `wellFiles` is either one multi-page TIFF path or a list of per-frame TIFF
    paths in frame order. Shared by the pipeline (run.py:_processOneWell) and
    the registration rebuild (registration.rebuildRegisteredRaw) so both see
    byte-identical input.
    """
    import tifffile
    if isinstance(wellFiles, str):
        raw = tifffile.imread(wellFiles)
        stack = raw[np.newaxis] if raw.ndim == 2 else raw
        del raw
    else:
        first = tifffile.imread(wellFiles[0])
        h, w = first.shape[:2]
        stack = np.empty((len(wellFiles), h, w), dtype=first.dtype)
        stack[0] = first
        del first
        for fi in range(1, len(wellFiles)):
            stack[fi] = tifffile.imread(wellFiles[fi])

    if stack.ndim == 3 and stack.shape[0] < stack.shape[2]:
        stack = np.transpose(stack, (1, 2, 0))
    return stack


def readImagesInplace(ntimepoints, arr, files):
    with ThreadPoolExecutor() as pool:
        pool.map(_readOne, [(arr, t, files[t]) for t in range(ntimepoints)])
    return arr


def _trimSpeculativePrealloc(path):
    """Release post-EOF blocks XFS speculatively preallocated during the write.

    tifffile writes a multi-page TIFF as many incremental appends, which XFS
    reads as a growing stream and answers with aggressive speculative
    preallocation beyond EOF. That extra space is NOT trimmed on close and does
    not age out: measured on /mnt/data (XFS, no extsize hint), a 372.6 MiB
    25-page stack occupied 884.5 MiB — 2.37x — and still did an hour later,
    while the same byte count written by a single `dd` occupied exactly 1.00x.
    Left alone it inflates a 96-well plate from 35 GiB to 83 GiB.

    An ftruncate to the file's own length frees the post-EOF extents and is a
    metadata-only operation: content is byte-identical (verified by md5) and no
    data is rewritten. Best-effort — other filesystems have no post-EOF
    preallocation to release, and Windows has no st_blocks at all, so any
    failure here is not worth failing a well over.
    """
    try:
        st = os.stat(path)
        allocated = getattr(st, 'st_blocks', 0) * 512
        if allocated > st.st_size:
            with open(path, 'r+b') as fh:
                fh.truncate(st.st_size)
    except OSError:
        pass


def saveStack(stack, outDir, filename):
    """Save (H, W, T) stack as multi-page TIF in (T, H, W) layout."""
    import tifffile
    os.makedirs(outDir, exist_ok=True)
    path = os.path.join(outDir, f"{filename}.tif")
    data = np.transpose(stack, (2, 0, 1)).astype(np.float32, copy=False)
    tifffile.imwrite(path, data)
    _trimSpeculativePrealloc(path)
