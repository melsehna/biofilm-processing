"""Regenerate overlay mp4s from existing processed data for a few test wells."""
import os, sys
import numpy as np
import imageio.v3 as iio
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from multiWellAnalysis.processing.preprocessing import normalize_local_contrast_output

PLATE_DIR = '/mnt/data/transposonSet/241106_150118_Plate 1'
PROC_DIR = os.path.join(PLATE_DIR, 'processedImages')
WELLS = ['A1_03', 'B5_03', 'D7_03', 'F10_03', 'H12_03']

for well in WELLS:
    raw_path = os.path.join(PROC_DIR, f'{well}_registered_raw.tif')
    mask_path = os.path.join(PROC_DIR, f'{well}_masks.npz')
    out_path = os.path.join(PROC_DIR, f'{well}_overlay.mp4')

    if not os.path.exists(raw_path) or not os.path.exists(mask_path):
        print(f'{well}: missing files, skipping')
        continue

    raw = iio.imread(raw_path).astype(np.float32)
    # Normalize to [0,1]
    imax = raw.max()
    if imax > 0:
        raw /= imax

    masks = np.load(mask_path)['masks']
    nframes = raw.shape[2]

    fpMax = np.nanmax(raw)
    fpMin = np.nanmin(raw)
    fpMean = 0.5 * (fpMax + fpMin)

    display_stack = normalize_local_contrast_output(raw, 101, fpMean)
    procVis = np.clip(display_stack, 0.0, 1.0)

    h_out, w_out = procVis.shape[:2]
    h_even = h_out & ~1
    w_even = w_out & ~1

    alpha = np.float32(0.35)
    cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    with imageio.get_writer(
        out_path, fps=2, codec='libx264',
        quality=8, pixelformat='yuv420p',
        macro_block_size=16
    ) as writer:
        for t in range(nframes):
            gray = procVis[:h_even, :w_even, t]
            frame = np.stack([gray, gray, gray], axis=-1)
            mask_t = masks[:h_even, :w_even, t]
            frame[mask_t] = (1.0 - alpha) * frame[mask_t] + alpha * cyan
            writer.append_data(np.clip(frame * 255, 0, 255).astype(np.uint8))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'{well}: {nframes} frames, {size_mb:.1f} MB -> {out_path}')
