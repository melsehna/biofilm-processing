# analysis_main.py

import os
import re
import numpy as np
import cv2

from .io_utils import saveStack
from .preprocessing import normalizeLocalContrast, normalizeLocalContrastOutput
from .segmentation import computeMaskInplace, dustCorrectInplace
from .registration import registerStackNormblur
from .overlay import writeOverlayVideo


from typing import Optional

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
    fftStride=3,
    downsample=2,
    skipOverlay=False,
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

    images = images.astype(np.float32, copy=False)
    imax = images.max()
    if imax > 0:
        images /= imax

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
        biomass = np.nanmean((1.0 - rawCropped) * masks, axis=(0, 1))

    _progress('Saving outputs...')

    # Invert processed stack before saving: biofilm pixels are bright after
    # normalizeLocalContrast; inverting gives the expected dark-biofilm appearance.
    processedToSave = np.clip(1.0 - processedStack, 0.0, 1.0)
    saveStack(processedToSave, processedDir, f"{filename}_processed")

    saveStack(rawCropped, processedDir, f"{filename}_registered_raw")

    npzPath = os.path.join(processedDir, f'{filename}_masks.npz')
    np.savez_compressed(npzPath, masks=masks)
    _registerImage('masks', npzPath)

    if not skipOverlay:
        overlayPath = os.path.join(processedDir, f'{filename}_overlay.mp4')
        fpMean = 0.5 * (np.nanmax(rawCropped) + np.nanmin(rawCropped))
        overlayDisplay = np.clip(
            normalizeLocalContrastOutput(rawCropped, blockDiameter, fpMean),
            0.0, 1.0,
        )
        writeOverlayVideo(overlayDisplay, masks, overlayPath, label=label)
        _registerImage('overlay_mp4', overlayPath)

    return masks, biomass, odMean
