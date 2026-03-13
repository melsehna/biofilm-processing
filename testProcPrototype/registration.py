# registration.py — OpenCV-accelerated version
# Changes from original:
#   - Manual FFT phase correlation -> cv2.phaseCorrelate (built-in, optimized)
#   - scipy.ndimage.shift -> cv2.warpAffine with translation matrix
#   - float32 throughout

import numpy as np
import cv2


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

    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def registerStackNormblur(
    normBlurStack,
    rawStack,
    shiftThresh,
    fftStride=3,
    downsample=2
):
    h, w, t = normBlurStack.shape

    registeredNorm = np.empty_like(normBlurStack)
    registeredRaw = np.empty_like(rawStack, dtype=np.float64)

    registeredNorm[..., 0] = normBlurStack[..., 0]
    registeredRaw[..., 0] = rawStack[..., 0]

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
            shiftToApply = cumShift
        else:
            shiftToApply = lastKeyShift

        shifts.append(shiftToApply.copy())

        registeredNorm[..., i] = _apply_shift(
            normBlurStack[..., i], shiftToApply
        )

        registeredRaw[..., i] = _apply_shift(
            rawStack[..., i], shiftToApply
        )

    return registeredNorm, registeredRaw, shifts
