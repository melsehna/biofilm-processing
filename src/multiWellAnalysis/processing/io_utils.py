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


# `_processed.tif` storage. 'float32' (default): uncompressed float32 in [0, 1].
# 'uint16': round(v * 65535) as uint16, deflate + horizontal predictor (~2.45x
# smaller on 4x brightfield; readable by ImageJ/Fiji). 65535 = the raw camera
# full scale (inputs are bit-depth scaled by 65535), so the step is one raw grey
# level and the max error is half of one (7.6e-6). Steps of 1/65535 never fall
# on an 8-bit bin edge ((n + 0.5) / 255), so img_as_ubyte - and every
# whole-image feature - is bit-identical to the float32 file. Always read with
# loadProcessedStack(), which returns float32 [0, 1] for either storage.
PROCESSED_DTYPES = ('float32', 'uint16')
UINT16_SCALE = 65535.0


def saveStack(stack, outDir, filename, dtype='float32'):
    """Save (H, W, T) stack as multi-page TIF in (T, H, W) layout.

    dtype='uint16' stores a [0, 1] stack as round(v * 65535), deflate-compressed
    (see PROCESSED_DTYPES); values outside [0, 1] are clipped.
    """
    import tifffile
    if dtype not in PROCESSED_DTYPES:
        raise ValueError(f'dtype must be one of {PROCESSED_DTYPES}, got {dtype!r}')
    os.makedirs(outDir, exist_ok=True)
    path = os.path.join(outDir, f"{filename}.tif")
    data = np.transpose(stack, (2, 0, 1))
    if dtype == 'uint16':
        data = np.round(np.clip(data, 0.0, 1.0) * UINT16_SCALE).astype(np.uint16)
        # processedScale in the (shaped JSON) ImageDescription marks the file as
        # a scaled processed stack, so readers can tell it from raw uint16 images.
        tifffile.imwrite(path, data, compression='zlib', predictor=2,
                         metadata={'processedScale': int(UINT16_SCALE)})
    else:
        tifffile.imwrite(path, data.astype(np.float32, copy=False))
    _trimSpeculativePrealloc(path)


def processedScale(path):
    """The `processedScale` marker of a uint16 `_processed.tif`, or None."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        md = tf.shaped_metadata[0] if tf.shaped_metadata else {}
    return md.get('processedScale') if isinstance(md, dict) else None


def toUnitFloat(arr, scale=UINT16_SCALE):
    """float32 [0, 1] view of a processed array stored as float32 or uint16."""
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / np.float32(scale)
    return arr


def loadProcessedStack(path):
    """Read a `_processed.tif` (as written, usually (T, H, W)) as float32 in [0, 1].

    Works for both storages (float32, or uint16 = round(v * 65535)). Uncompressed
    float32 files come back as a read-only memmap (no copy); compressed or uint16
    files are decoded into memory.
    """
    import tifffile
    try:
        arr = tifffile.memmap(path, mode='r')
    except Exception:
        arr = tifffile.imread(path)
    if arr.dtype == np.uint16:
        scale = processedScale(path)
        if not scale:
            raise ValueError(f'{path} is uint16 without a processedScale marker: '
                             f'not a processed stack written by saveStack (raw image?)')
        return toUnitFloat(arr, float(scale))
    return toUnitFloat(arr)
