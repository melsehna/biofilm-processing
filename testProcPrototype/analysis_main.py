# analysis_main.py — OpenCV-accelerated version
# Changes from original:
#   - scipy.ndimage.gaussian_filter -> cv2.GaussianBlur
#   - Downsample blur uses cv2.pyrDown (anti-aliased) + cv2.resize (upsample)
#   - float32 where possible for lower memory bandwidth
#   - skimage.measure.label / label2rgb removed from overlay (not needed)

import os
import json
import numpy as np
import pandas as pd
import imageio.v3 as iio
import imageio
import cv2
from glob import glob
from tqdm import tqdm

from helpers import round_odd
from io_utils import read_images_inplace, save_stack
from preprocessing import (
    normalize_local_contrast,
    mean_filter,
    preprocess_stack,
    normalize_local_contrast_output
)
from segmentation import compute_mask_inplace, dust_correct_inplace
from registration import registerStackNormblur
from plotting_tools import save_biomass_curve, save_peak_panel

import matplotlib.pyplot as plt
import re
from typing import Optional


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


    # scale raw to [0,1]
    images = images.astype(np.float32, copy=False)
    imax = images.max()
    if imax > 0:
        images /= imax


    # G_sigma(G_bd(img) - img) = G_sigma_c(img) - G_sigma(img)
    # where sigma_c = sqrt(bd^2 + sigma^2)
    sigma = 2.0 * (block_diameter / 31.0)
    sigma_combined = np.sqrt(block_diameter ** 2 + sigma ** 2)

    norm_blur = np.empty(images.shape, dtype=np.float32)

    for t in range(ntimepoints):
        img_t = images[..., t]
        big = cv2.GaussianBlur(
            img_t, ksize=(0, 0), sigmaX=sigma_combined,
            borderType=cv2.BORDER_REFLECT
        )
        small = cv2.GaussianBlur(
            img_t, ksize=(0, 0), sigmaX=sigma,
            borderType=cv2.BORDER_REFLECT
        )
        norm_blur[..., t] = big - small


    # 2) register normalized for masking; register raw for OD/biomass
    registered_norm, registered_raw, shifts_array = registerStackNormblur(
        norm_blur,
        images,
        shift_thresh,
        fftStride=fftStride,
        downsample=downsample
    )

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

        # Mean over all pixels (masked pixels contribute OD, unmasked contribute 0)
        biomass = np.nanmean(OD * masks, axis=(0, 1))
        odMean = biomass.copy()

    else:
        # Mean over all pixels (masked pixels contribute 1-raw, unmasked contribute 0)
        biomass = np.nanmean((1.0 - raw_cropped) * masks, axis=(0, 1))

    # 6) save stacks
    save_stack(processed_stack, processed_dir, f"{filename}_processed")

    save_stack(
        raw_cropped,
        processed_dir,
        f"{filename}_registered_raw"
    )

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

        procVis = np.clip(display_stack, 0.0, 1.0)  # (H, W, T) float32

        h_out, w_out = procVis.shape[:2]
        # Ensure even dimensions for H.264 yuv420p
        h_out_even = h_out & ~1
        w_out_even = w_out & ~1

        overlay_mp4_path = os.path.join(
            processed_dir, f'{filename}_overlay.mp4'
        )

        alpha = np.float32(0.6)
        cyan = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # BGR cyan

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(
            overlay_mp4_path, fourcc, 2, (w_out_even, h_out_even)
        )

        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                overlay_mp4_path, fourcc, 2, (w_out_even, h_out_even)
            )

        for t in range(ntimepoints):
            gray = procVis[:h_out_even, :w_out_even, t]
            frame = cv2.merge([gray, gray, gray])  # (H, W, 3) float32 BGR
            mask_t = masks[:h_out_even, :w_out_even, t]
            frame[mask_t] = (1.0 - alpha) * frame[mask_t] + alpha * cyan
            frame_u8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
            writer.write(frame_u8)

        writer.release()
        register_image('overlay_mp4', overlay_mp4_path)

    return masks, biomass, odMean


# File helpers and batch execution

def extract_base_and_ext(filename, batch):
    base, ext = os.path.splitext(os.path.basename(filename))
    if batch:
        match = re.match(r"^(.*\D)\d+(\.[^\.]+)$", filename)
        if match:
            base, ext = match.groups()
    return base, ext


def filter_same_well(target, paths):
    well = os.path.basename(target).split('_')[0]
    return [p for p in paths if os.path.basename(p).split('_')[0] == well]


# Main entry: analysis_main
def main(config_path="experiment_config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    image_dirs = config["images_directory"]
    dust_correction = config["dust_correction"] == "True"
    batch = config["batch_processing"] == "True"
    fixed_thresh = float(config["fixed_thresh"])
    Imin_path = config.get("Imin_path", "")
    Imax_path = config.get("Imax_path", "")
    block_diameter = round_odd(config["blockDiam"])
    shift_thresh = 50

    Imin = iio.imread(Imin_path).astype(np.float64) if Imin_path else None
    Imax = iio.imread(Imax_path).astype(np.float64) if Imax_path else None




    for dir_path in image_dirs:
        print(f"\nProcessing directory: {dir_path}")
        image_records = []
        timeseriesDfs = []

        processed_dir = os.path.join(dir_path, "processedImages")

        numeric_dir = os.path.join(dir_path, "numericalData")
        for d in [processed_dir, numeric_dir]:
            if os.path.isdir(d):
                import shutil; shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        protocol_path = os.path.join(dir_path, "protocol.csv")
        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
        protocol = pd.read_csv(protocol_path)

        bf_reads = protocol[
            (protocol["action"] == "Imaging Read") &
            (protocol["channel"] == "Bright Field")
        ]

        tif_files = [f for f in glob(os.path.join(dir_path, "*.tif")) if os.path.isfile(f)]

        for _, bf_row in tqdm(bf_reads.iterrows(), total=len(bf_reads), desc="Processing wells"):
            step = bf_row["step"]
            pattern = f"_0{step}_"
            bf_files = [f for f in tif_files if pattern in f and "Bright_Field" in f]

            for file in bf_files:
                test_img = iio.imread(file)
                shape = test_img.shape
                target_base, _ = extract_base_and_ext(file, batch)

                if len(shape) == 3:
                    h, w, ntimepoints = shape
                    images = test_img.astype(np.float64)
                    masks, biomass, odMean = timelapse_processing(
                        images,
                        block_diameter,
                        ntimepoints,
                        shift_thresh,
                        fixed_thresh,
                        dust_correction,
                        dir_path,
                        target_base,
                        image_records,
                        Imin,
                        Imax
                    )

                    df = pd.DataFrame({
                        'Well': target_base,
                        'Frame': np.arange(ntimepoints),
                        'Biomass': biomass,
                        'OD_mean': odMean if odMean is not None else np.full(ntimepoints, np.nan)
                    })

                    timeseriesDfs.append(df)

                elif len(shape) == 2:
                    pattern = f"{target_base}_*.tif"
                    timelapse_files = sorted(
                        glob(os.path.join(dir_path, pattern)),
                        key=frame_index_from_filename
                    )

                    if len(timelapse_files) > 1:
                        ntimepoints = len(timelapse_files)
                        height, width = test_img.shape
                        stack = np.empty((height, width, ntimepoints), dtype=np.float64)

                        frame_indices = [frame_index_from_filename(f) for f in timelapse_files]
                        assert frame_indices == sorted(frame_indices), (
                            f'Frame order error in {target_base}: {frame_indices}'
                        )


                        read_images_inplace(ntimepoints, stack, timelapse_files)
                        masks, biomass, odMean = timelapse_processing(
                            stack,
                            block_diameter,
                            ntimepoints,
                            shift_thresh,
                            fixed_thresh,
                            dust_correction,
                            dir_path,
                            target_base,
                            image_records,
                            Imin,
                            Imax
                        )

                        df = pd.DataFrame({
                            'Well': target_base,
                            'Frame': np.arange(ntimepoints),
                            'Biomass': biomass,
                            'OD_mean': odMean if odMean is not None else np.full(ntimepoints, np.nan)
                        })

                        timeseriesDfs.append(df)

                    else:
                        raise RuntimeError(
                            f"Single-frame processing not supported in Python pipeline yet: {file}"
                        )
                else:
                    raise ValueError(f"Unexpected image shape: {shape}")


        if timeseriesDfs:
            final_df = pd.concat(timeseriesDfs, ignore_index=True)
            out_csv = os.path.join(numeric_dir, 'BF_timeseries.csv')
            final_df.to_csv(out_csv, index=False)
            print(f'[X] Wrote: {out_csv}')


        if image_records:
            df_records = pd.DataFrame(image_records)
            records_path = os.path.join(numeric_dir, "image_records.csv")
            df_records.to_csv(records_path, index=False)
            print(f' [X] Wrote image manifest: {records_path}')




if __name__ == "__main__":
    raise RuntimeError(
        "analysis_main.py is deprecated. "
        "Use 12-18-runPlates.py for all plate processing."
    )
