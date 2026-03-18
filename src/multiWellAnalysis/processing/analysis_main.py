# analysis_main.py

import os
import re
import numpy as np
import cv2

from .io_utils import save_stack
from .preprocessing import normalize_local_contrast, normalize_local_contrast_output
from .segmentation import compute_mask_inplace, dust_correct_inplace
from .registration import registerStackNormblur
from .overlay import write_overlay_video



# Helper functions
def crop_stack(img_stack):
    h, w = img_stack.shape[:2]
    # Fast path: check a small sample for NaN before scanning the full stack
    if not (np.isnan(img_stack[0, 0, :]).any() or
            np.isnan(img_stack[-1, -1, :]).any() or
            np.isnan(img_stack[0, -1, :]).any() or
            np.isnan(img_stack[-1, 0, :]).any()):
        return img_stack, (0, h, 0, w)

    mask = ~np.any(np.isnan(img_stack), axis=2)
    mask_i = np.any(mask, axis=1)
    mask_j = np.any(mask, axis=0)
    i1, i2 = np.where(mask_i)[0][[0, -1]]
    j1, j2 = np.where(mask_j)[0][[0, -1]]
    cropped = img_stack[i1:i2 + 1, j1:j2 + 1, :]
    return cropped, (i1, i2 + 1, j1, j2 + 1)


def frame_index_from_filename(path):
    m = re.search(r'_(\d+)\.tif$', os.path.basename(path))
    if m is None:
        raise ValueError(f'Cannot extract frame index from {path}')
    return int(m.group(1))

from typing import Optional

def timelapse_processing(
    images,
    block_diameter,
    ntimepoints,
    shift_thresh,
    fixed_thresh,
    dust_correction,
    outdir,
    filename,
    image_records,
    Imin: Optional[np.ndarray] = None,
    Imax: Optional[np.ndarray] = None,
    fftStride=3,
    downsample=2,
    skip_overlay=False,
    label=None,
    workers=4,
    progress_fn=None,
):

    processed_dir = os.path.join(outdir, 'processedImages')
    os.makedirs(processed_dir, exist_ok=True)
    
    if ntimepoints != images.shape[2]:
        raise ValueError(
            f'ntimepoints ({ntimepoints}) does not match images shape ({images.shape})'
        )
    
    def register_image(kind, path):
        if image_records is not None:
            image_records.append({
                'Well': filename,
                'Type': kind,
                'Path': os.path.abspath(path)
            })
        

    def _progress(msg):
        if progress_fn is not None:
            progress_fn(msg)

    # scale raw to [0,1] like Julia floatloader
    images = images.astype(np.float32, copy=False)
    imax = images.max()
    if imax > 0:
        images /= imax


    # Julia hardcodes sig = 2 for the post-normalization blur
    sigma = 2.0

    norm_blur = np.empty(images.shape, dtype=np.float32)

    for t in range(ntimepoints):
        _progress(f'Normalizing frame {t+1}/{ntimepoints}')
        r = normalize_local_contrast(images[..., t], block_diameter)
        norm_blur[..., t] = cv2.GaussianBlur(
            r, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )

    _progress('Registering stack...')

    # 2) register normalized for masking; register raw for OD/biomass
    registered_norm, registered_raw, shifts_array = registerStackNormblur(
        norm_blur,
        images,
        shift_thresh,
        fftStride=fftStride,
        downsample=downsample,
        workers=workers,
    )

    _progress('Cropping + computing masks...')

    # 3) crop away NaN borders (if any appeared)

    processed_stack, crop_indices = crop_stack(registered_norm)

    row_min, row_max, col_min, col_max = crop_indices
    raw_cropped = registered_raw[row_min:row_max, col_min:col_max, :]
    if Imin is not None:
        Imin = Imin[row_min:row_max, col_min:col_max]
    if Imax is not None:
        Imax = Imax[row_min:row_max, col_min:col_max]

    # 4) mask computation - binary mask on processed (normalized) stack
    masks = np.zeros(processed_stack.shape, dtype=bool)
    compute_mask_inplace(processed_stack, masks, fixed_thresh)


    if dust_correction:
        dust_correct_inplace(masks)

    # 5) biomass curve (OD if references provided; else 1 - normalized)
    biomass = np.zeros(ntimepoints, dtype=np.float64)
    odMean = None

    if Imin is not None:
        # Vectorized OD computation across all frames at once
        if Imax is not None:
            denom = Imax[..., np.newaxis] - Imin[..., np.newaxis] + 1e-12
        else:
            denom = raw_cropped[..., 0:1] - Imin[..., np.newaxis] + 1e-12

        OD = -np.log10((raw_cropped - Imin[..., np.newaxis]) / denom + 1e-12)

        biomass = np.nanmean(OD * masks, axis=(0, 1))
        odMean = biomass.copy()

    else:
        biomass = np.nanmean((1.0 - raw_cropped) * masks, axis=(0, 1))

    _progress('Saving outputs...')

    # 6) save stacks 
    save_stack(processed_stack, processed_dir, f"{filename}_processed") # normalized+blurred, registered, cropped
    
    save_stack(
        raw_cropped,
        processed_dir,
        f"{filename}_registered_raw"
    )

    # save_stack(processed_stack * masks, processed_dir, f"{filename}_processed_masked")

    # 7) save masks
    npz_path = os.path.join(processed_dir, f'{filename}_masks.npz')
    np.savez_compressed(npz_path, masks=masks)
    register_image('masks', npz_path)

    # 8) overlay video (optional)
    if not skip_overlay:
        fpMax = np.nanmax(raw_cropped)
        fpMin = np.nanmin(raw_cropped)
        fpMean = 0.5 * (fpMax + fpMin)

        display_stack = normalize_local_contrast_output(
            raw_cropped, block_diameter, fpMean
        )
        procVis = np.clip(display_stack, 0.0, 1.0)

        overlay_mp4_path = os.path.join(
            processed_dir, f'{filename}_overlay.mp4'
        )

        write_overlay_video(procVis, masks, overlay_mp4_path, label=label)
        register_image('overlay_mp4', overlay_mp4_path)

    return masks, biomass, odMean

