# analysis_main.py

import os
import re
import numpy as np
import cv2

from .io_utils import saveStack
from .preprocessing import normalizeLocalContrast, normalizeLocalContrastOutput
from .segmentation import computeMaskInplace, dustCorrectInplace
from .registration import registerStackNormblur
from .overlay import writeOverlayVideo, writeProcessedVideo


from typing import Optional

def _toBitDepthScaled(arr):
    """Cast to float32 in [0, 1] using bit-depth-aware scaling.

    Integer dtype → divide by the dtype's full-scale.
    Float dtype with max > 1.5 → infer bit depth (255 vs 65535) and divide.
    Float dtype already in [0, 1] → return as float32 unchanged.
    Empty / None → returned as-is.
    """
    if arr is None:
        return None
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    arr = arr.astype(np.float32, copy=False)
    amax = float(arr.max()) if arr.size else 0.0
    if amax > 1.5:
        bitDepthScale = 255.0 if amax <= 255.0 else 65535.0
        return arr / bitDepthScale
    return arr


def cropStack(imgStack):
    h, w = imgStack.shape[:2]
    if not (np.isnan(imgStack[0, 0, :]).any() or
            np.isnan(imgStack[-1, -1, :]).any() or
            np.isnan(imgStack[0, -1, :]).any() or
            np.isnan(imgStack[-1, 0, :]).any()):
        return imgStack, (0, h, 0, w)

    mask = ~np.any(np.isnan(imgStack), axis=2)
    maskI = np.any(mask, axis=1)
    maskJ = np.any(mask, axis=0)
    i1, i2 = np.where(maskI)[0][[0, -1]]
    j1, j2 = np.where(maskJ)[0][[0, -1]]
    cropped = imgStack[i1:i2 + 1, j1:j2 + 1, :]
    return cropped, (i1, i2 + 1, j1, j2 + 1)


def frameIndexFromFilename(path):
    m = re.search(r'_(\d+)\.tif$', os.path.basename(path))
    if m is None:
        raise ValueError(f'Cannot extract frame index from {path}')
    return int(m.group(1))


def timelapseProcessing(
    images,
    blockDiameter,
    ntimepoints,
    shiftThresh,
    fixedThresh,
    dustCorrection,
    outdir,
    filename,
    imageRecords,
    Imin: Optional[np.ndarray] = None,
    Imax: Optional[np.ndarray] = None,
    fftStride=1,
    downsample=2,
    skipOverlay=False,
    saveProcessedVideo=False,
    saveFpHalf=True,
    label=None,
    workers=4,
    progressFn=None,
):
    processedDir = os.path.join(outdir, 'processedImages')
    os.makedirs(processedDir, exist_ok=True)

    if ntimepoints != images.shape[2]:
        raise ValueError(
            f'ntimepoints ({ntimepoints}) does not match images shape ({images.shape})'
        )

    def _registerImage(kind, path):
        if imageRecords is not None:
            imageRecords.append({
                'Well': filename,
                'Type': kind,
                'Path': os.path.abspath(path)
            })

    def _progress(msg):
        if progressFn is not None:
            progressFn(msg)

    # Bit-depth-aware scaling: every input (images, Imin, Imax) goes onto a
    # shared [0, 1] photometric axis based on the originating bit depth, not
    # each stack's own max. This is what makes cross-well intensity features
    # and OD math comparable across wells/plates/sessions.
    images = _toBitDepthScaled(images)
    Imin = _toBitDepthScaled(Imin)
    Imax = _toBitDepthScaled(Imax)

    sigma = 2.0
    normBlur = np.empty(images.shape, dtype=np.float32)

    for t in range(ntimepoints):
        _progress(f'Normalizing frame {t+1}/{ntimepoints}')
        r = normalizeLocalContrast(images[..., t], blockDiameter)
        normBlur[..., t] = cv2.GaussianBlur(
            r, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )

    _progress('Registering stack...')

    registeredNorm, registeredRaw, shiftsArray = registerStackNormblur(
        normBlur,
        images,
        shiftThresh,
        fftStride=fftStride,
        downsample=downsample,
        workers=workers,
    )

    _progress('Cropping + computing masks...')

    processedStack, cropIndices = cropStack(registeredNorm)

    rowMin, rowMax, colMin, colMax = cropIndices
    rawCropped = registeredRaw[rowMin:rowMax, colMin:colMax, :]
    if Imin is not None:
        Imin = Imin[rowMin:rowMax, colMin:colMax]
    if Imax is not None:
        Imax = Imax[rowMin:rowMax, colMin:colMax]

    masks = np.zeros(processedStack.shape, dtype=bool)
    computeMaskInplace(processedStack, masks, fixedThresh)

    if dustCorrection:
        dustCorrectInplace(masks)

    biomass = np.zeros(ntimepoints, dtype=np.float64)
    odMean = None

    if Imin is not None:
        if Imax is not None:
            denom = Imax[..., np.newaxis] - Imin[..., np.newaxis] + 1e-12
        else:
            denom = rawCropped[..., 0:1] - Imin[..., np.newaxis] + 1e-12

        OD = -np.log10((rawCropped - Imin[..., np.newaxis]) / denom + 1e-12)
        biomass = np.nanmean(OD * masks, axis=(0, 1))
        odMean = biomass.copy()
    else:
        # No Imin/Imax flat-field references: use the well's own early frames as a
        # per-pixel blank. Biofilm never appears before ~frame 8, so frames 0-4 are
        # biofilm-free. OD = -log10(I_t / blank) is a ratio against the same well's
        # own blank, so it cancels exposure/gain and illumination/vignetting -> a
        # batch-robust, level-preserving absorbance, with no calibration images.
        # Biomass is therefore reported as mean OD. See ISSUES.md Phase 5.
        nBlank = min(5, rawCropped.shape[2])
        blank = np.maximum(np.nanmean(rawCropped[..., :nBlank], axis=2, keepdims=True), 1e-6)
        OD = -np.log10(np.clip(rawCropped / blank, 1e-6, None))
        biomass = np.nanmean(OD * masks, axis=(0, 1))
        odMean = biomass.copy()

    _progress('Saving outputs...')

    # Display rendering: FIXED fpMean = 0.5 — the sole render. The per-stack
    # adaptive fpMean was retired: it drifts well-to-well / batch-to-batch and is
    # unsafe for cross-batch feature/embedding comparison. `_processed.tif` IS this
    # fixed render (the separate adaptive render and the `_fpHalf` suffix are gone;
    # the `saveFpHalf` arg is now vestigial). The local-contrast subtraction carries
    # all the biological signal; fpMean only sets the uniform background gray. Use
    # the version stamp (run_params.json) to tell fixed (new) from adaptive (old)
    # `_processed.tif`. See ISSUES.md "fpMean policy".
    displayStack = np.clip(
        normalizeLocalContrastOutput(rawCropped, blockDiameter, 0.5),
        0.0, 1.0,
    )
    saveStack(displayStack, processedDir, f"{filename}_processed")

    saveStack(rawCropped, processedDir, f"{filename}_registered_raw")

    npzPath = os.path.join(processedDir, f'{filename}_masks.npz')
    np.savez_compressed(npzPath, masks=masks)
    _registerImage('masks', npzPath)

    if not skipOverlay:
        overlayPath = os.path.join(processedDir, f'{filename}_overlay.mp4')
        writeOverlayVideo(displayStack, masks, overlayPath, label=label)
        _registerImage('overlay_mp4', overlayPath)

    if saveProcessedVideo:
        processedVideoPath = os.path.join(processedDir, f'{filename}_processed.mp4')
        writeProcessedVideo(displayStack, processedVideoPath, label=label)
        _registerImage('processed_mp4', processedVideoPath)

    return masks, biomass, odMean
