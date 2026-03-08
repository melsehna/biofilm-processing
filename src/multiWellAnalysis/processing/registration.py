# multiWellAnalysis/registration.py
import numpy as np
from scipy.ndimage import shift as apply_shift
from scipy.fft import fftn, ifftn


def phaseOffset(fixed, moving):
    fixed = fixed.astype(np.float64, copy=False)
    moving = moving.astype(np.float64, copy=False)

    F = fftn(fixed, workers=-1)
    M = fftn(moving, workers=-1)

    R = F * np.conj(M)
    r = np.abs(ifftn(R, workers=-1))

    maxIdx = np.unravel_index(np.argmax(r), r.shape)
    shifts = np.array(maxIdx, dtype=np.float64)

    for d in range(len(shifts)):
        if shifts[d] > r.shape[d] // 2:
            shifts[d] -= r.shape[d]

    return shifts


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

        registeredNorm[..., i] = apply_shift(
            normBlurStack[..., i],
            shift=shiftToApply,
            order=1,
            mode='reflect'
        )

        registeredRaw[..., i] = apply_shift(
            rawStack[..., i],
            shift=shiftToApply,
            order=1,
            mode='reflect'
        )

    return registeredNorm, registeredRaw, shifts

