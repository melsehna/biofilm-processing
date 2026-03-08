# analysis_main.py

import os
import json
import numpy as np
import pandas as pd
import imageio.v3 as iio
import imageio
from glob import glob
from tqdm import tqdm

# local package imports 
# from multiWellAnalysis.helpers import round_odd
# from multiWellAnalysis.io_utils import read_images_inplace, save_stack
# from multiWellAnalysis.preprocessing import normalize_local_contrast, mean_filter, preprocess_stack, normalize_local_contrast_output
# from multiWellAnalysis.segmentation import compute_mask_inplace, dust_correct_inplace
# from multiWellAnalysis.registration import phaseOffset, registerStackNormblur


from .helpers import round_odd
from .io_utils import read_images_inplace, save_stack
from .preprocessing import (
    normalize_local_contrast,
    mean_filter,
    preprocess_stack,
    normalize_local_contrast_output
)
from .segmentation import compute_mask_inplace, dust_correct_inplace
from .registration import phaseOffset, registerStackNormblur
from .plotting_tools import save_biomass_curve, save_peak_panel


import matplotlib.pyplot as plt
import re
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from skimage.color import label2rgb



# Helper functions
def crop_stack(img_stack):
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
        

    # scale raw to [0,1] like Julia floatloader
    images = images.astype(np.float64, copy=False)
    imax = images.max()
    if imax > 0:
        images /= imax

        
    # scale sigma to block diam
    # Julia uses σ in pixel units, same scale as block_diameter
    sigma = 2.0 * (block_diameter / 31.0)
    
    norm_blur = np.empty_like(images)

    for t in range(ntimepoints):
        r = normalize_local_contrast(images[..., t], block_diameter)

        r_small = r[::2, ::2]
        blur_small = gaussian_filter(
            r_small,
            sigma=sigma / 2.0,
            mode='reflect'
        )

        norm_blur[..., t] = (
            blur_small
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )[:images.shape[0], :images.shape[1]]


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

    
    fpMax = np.nanmax(raw_cropped)
    fpMin = np.nanmin(raw_cropped)
    fpMean = 0.5 * (fpMax + fpMin)
    
    display_stack = normalize_local_contrast_output(
        raw_cropped,block_diameter,fpMean
    )
    
    # 4) mask computation - binary mask on processed (normalized) stack
    masks = np.zeros(processed_stack.shape, dtype=bool)
    compute_mask_inplace(processed_stack, masks, fixed_thresh)


    if dust_correction:
        dust_correct_inplace(masks)

    # 5) biomass curve (OD if references provided; else 1 - normalized)
    biomass = np.zeros(ntimepoints, dtype=np.float64)
    odMean = None
    
    if Imin is not None:
        OD = np.empty_like(raw_cropped, dtype=np.float64)
        for t in range(masks.shape[2]):
            if Imax is not None:
                OD[..., t] = -np.log10((raw_cropped[..., t] - Imin) / (Imax - Imin + 1e-12) + 1e-12)
            else:
                OD[..., t] = -np.log10((raw_cropped[..., t] - Imin) / (raw_cropped[..., 0] - Imin + 1e-12) + 1e-12)
            biomass[t] = np.nanmean(OD[..., t] * masks[..., t])
            
        odMean = np.array([
            np.nanmean(OD[..., t] * masks[..., t])
            for t in range(masks.shape[2])
        ])

    else:
        for t in range(masks.shape[2]):
            biomass[t] = np.nanmean((1.0 - raw_cropped[..., t]) * masks[..., t])

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
    
    # 8) save overlay stack (RGB TIFF)
    # procVis = processed_stack.copy()
    # pmin = np.nanmin(procVis)
    # pmax = np.nanmax(procVis)
    # if pmax > pmin:
    #     procVis = (procVis - pmin) / (pmax - pmin)
    
    procVis = np.clip(display_stack, 0.0, 1.0)


    overlays = []
    for t in range(ntimepoints):
        gray = procVis[..., t]
        rgb = np.stack([gray, gray, gray], axis=-1)
        
        alpha = 0.6  # transparency of mask overlay (cyan)

        mask_t = masks[..., t]
        cyan = np.array([0.0, 1.0, 1.0])

        rgb[mask_t] = (
            (1.0 - alpha) * rgb[mask_t]
            + alpha * cyan
        )


        overlays.append((rgb * 255).astype(np.uint8))


        # ov = label2rgb(
        #     label(masks[..., t]),
        #     image=procVis[..., t],
        #     bg_label=0,
        #     alpha=0.2
        # )
        # overlays.append((ov * 255).astype(np.uint8))

    # Save overlay stack as TIFF (optional, see redundancy section below)
    # overlay_tif_path = os.path.join(processed_dir, f'{filename}_overlay_stack.tif')
    # iio.imwrite(overlay_tif_path, np.stack(overlays, axis=0))
    # register_image('overlay_tif', overlay_tif_path)

    # Save overlay as GIF
    overlay_mp4_path = os.path.join(processed_dir, f'{filename}_overlay.mp4')

    try:
        with imageio.get_writer(
            overlay_mp4_path,
            fps=2,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        ) as writer:
            for frame in overlays:
                writer.append_data(frame)

        register_image('overlay_mp4', overlay_mp4_path)

    except Exception as e:
        # Fallback to GIF if ffmpeg is unavailable
        overlay_gif_path = os.path.join(processed_dir, f'{filename}_overlay.gif')
        iio.imwrite(overlay_gif_path, overlays, duration=0.5)
        register_image('overlay_gif', overlay_gif_path)

        # Save overlay as MP4 (much faster than GIF)


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
