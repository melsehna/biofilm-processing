# multiWellAnalysis/registration.py — two-pass in-place registration
import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor


def phaseOffset(fixed, moving):
    """Compute sub-pixel translational shift via OpenCV phase correlation."""
    fixed = fixed.astype(np.float64, copy=False)
    moving = moving.astype(np.float64, copy=False)

    # cv2.phaseCorrelate returns (dx, dy), response
    # Our convention: shifts are (row_shift, col_shift) = (dy, dx)
    (dx, dy), _response = cv2.phaseCorrelate(fixed, moving)

    return np.array([dy, dx], dtype=np.float64)


def _apply_shift(image, shift):
    """Translate image by (row_shift, col_shift) using cv2.warpAffine."""
    dy, dx = shift
    h, w = image.shape[:2]

    # 2x3 affine translation matrix
    M = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy]
    ], dtype=np.float64)

    # BORDER_CONSTANT with NaN — NOT BORDER_REFLECT. The strip vacated by the
    # shift must be marked invalid, not filled with mirrored interior pixels:
    # reflection fabricates biofilm-like texture at the edge AND defeats
    # cropStack (which trims NaN borders and otherwise no-ops). The stack is
    # float here, so NaN is representable; cropStack then trims the region
    # invalid in any frame.
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float('nan'),
    )


def _compute_shifts(normBlurStack, shiftThresh, fftStride, downsample):
    """Pass 1: compute per-frame shifts (read-only on normBlurStack)."""
    t = normBlurStack.shape[2]

    cumShift = np.array([0.0, 0.0], dtype=np.float64)
    shifts = [cumShift.copy()]

    lastKeyframe = 0
    lastKeyShift = cumShift.copy()

    for i in range(1, t):
        if i % fftStride == 0:
            fixedSmall = normBlurStack[..., lastKeyframe][::downsample, ::downsample]
            movingSmall = normBlurStack[..., i][::downsample, ::downsample]

            proposed = -phaseOffset(fixedSmall, movingSmall) * downsample

            if np.linalg.norm(proposed) < float(shiftThresh):
                cumShift += proposed

            lastKeyframe = i
            lastKeyShift = cumShift.copy()
            shifts.append(cumShift.copy())
        else:
            shifts.append(lastKeyShift.copy())

    return shifts


def _apply_shifts_inplace(stack, shifts, workers=4):
    """Pass 2: apply precomputed shifts in-place (threaded)."""
    def _do(i):
        s = shifts[i]
        if s[0] == 0.0 and s[1] == 0.0:
            return
        stack[..., i] = _apply_shift(stack[..., i], s)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pool.map(_do, range(1, stack.shape[2]))


def registerStackNormblur(
    normBlurStack,
    rawStack,
    shiftThresh,
    fftStride=1,
    downsample=2,
    workers=4,
):
    """Two-pass registration: compute shifts, then apply in-place.

    Both normBlurStack and rawStack are modified in-place to avoid
    allocating two additional full-size copies (~940 MB for 1992x1992x31).
    """
    shifts = _compute_shifts(normBlurStack, shiftThresh, fftStride, downsample)
    _apply_shifts_inplace(normBlurStack, shifts, workers=workers)
    _apply_shifts_inplace(rawStack, shifts, workers=workers)
    return normBlurStack, rawStack, shifts


# ---------------------------------------------------------------------------
# Registration sidecar: the registration of a well is just per-frame (dy, dx)
# translations plus the NaN-border crop box, ~500 bytes. Saving it lets the
# registered raw stack be rebuilt bit-exactly from the original Cytation TIFFs
# on demand, instead of storing a full `_registered_raw.tif` (~490 MB/well).
# ---------------------------------------------------------------------------

REGISTRATION_VERSION = 1


def registrationPath(procDir, wellId):
    return os.path.join(procDir, f'{wellId}_registration.npz')


def saveRegistration(path, shifts, cropIndices, frameShape, shiftThresh,
                     fftStride, downsample, sourceFiles=None):
    """Write the per-well registration sidecar (`<well>_registration.npz`)."""
    if sourceFiles is None:
        files = []
    elif isinstance(sourceFiles, str):
        files = [os.path.abspath(sourceFiles)]
    else:
        files = [os.path.abspath(f) for f in sourceFiles]
    np.savez(
        path,
        version=np.int64(REGISTRATION_VERSION),
        shifts=np.asarray(shifts, dtype=np.float64).reshape(-1, 2),
        cropIndices=np.asarray(cropIndices, dtype=np.int64),
        frameShape=np.asarray(frameShape[:2], dtype=np.int64),
        shiftThresh=np.float64(shiftThresh),
        fftStride=np.int64(fftStride),
        downsample=np.int64(downsample),
        sourceFiles=np.asarray(files, dtype=str),
    )


def loadRegistration(path):
    """Load a registration sidecar as a plain dict (no pickle)."""
    with np.load(path, allow_pickle=False) as d:
        return {
            'version': int(d['version']),
            'shifts': d['shifts'],
            'cropIndices': tuple(int(v) for v in d['cropIndices']),
            'frameShape': tuple(int(v) for v in d['frameShape']),
            'shiftThresh': float(d['shiftThresh']),
            'fftStride': int(d['fftStride']),
            'downsample': int(d['downsample']),
            'sourceFiles': [str(f) for f in d['sourceFiles']],
        }


def rebuildRegisteredRaw(path, sourceFiles=None):
    """Rebuild a well's registered raw stack (H, W, T float32) from its sidecar.

    Re-reads the raw frames (from `sourceFiles`, or the paths recorded in the
    sidecar), then applies the exact pipeline steps: bit-depth scaling, the
    stored shifts via `_apply_shifts_inplace`, and the stored crop. The result
    equals what `_registered_raw.tif` would have held, given the same library
    versions (pin via environment.yml). Pass `sourceFiles` if the raw data has
    moved since processing.
    """
    from .analysis_main import _toBitDepthScaled
    from .io_utils import readWellStack

    reg = loadRegistration(path)
    files = sourceFiles if sourceFiles is not None else reg['sourceFiles']
    if not files:
        raise ValueError(f'{path}: no source files recorded; pass sourceFiles')
    if not isinstance(files, str):
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} raw frame(s) missing, e.g. {missing[0]} '
                '(pass sourceFiles if the raw data moved)')

    images = _toBitDepthScaled(readWellStack(files))
    if tuple(images.shape[:2]) != reg['frameShape'] or images.shape[2] != len(reg['shifts']):
        raise ValueError(
            f'{path}: raw stack {images.shape} does not match registration '
            f'(frames {reg["frameShape"]} x {len(reg["shifts"])})')
    _apply_shifts_inplace(images, reg['shifts'], workers=1)
    r0, r1, c0, c1 = reg['cropIndices']
    return images[r0:r1, c0:c1, :]


def loadRegisteredRaw(procDir, wellId):
    """Registered raw stack (H, W, T) for a well: from `_registered_raw.tif` if
    it was saved, else rebuilt from `_registration.npz`. None if neither exists."""
    import tifffile
    tifPath = os.path.join(procDir, f'{wellId}_registered_raw.tif')
    if os.path.exists(tifPath):
        raw = tifffile.imread(tifPath)
        if raw.ndim == 3 and raw.shape[0] < raw.shape[1]:
            raw = np.transpose(raw, (1, 2, 0))
        return raw
    regPath = registrationPath(procDir, wellId)
    if os.path.exists(regPath):
        return rebuildRegisteredRaw(regPath)
    return None
