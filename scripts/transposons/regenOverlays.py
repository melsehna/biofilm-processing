import os
import sys
import numpy as np
import imageio.v3 as iio
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from multiWellAnalysis.processing.preprocessing import normalize_local_contrast_output

PLATE_DIR = '/mnt/data/transposonSet/241106_150118_Plate 1'
PROC_DIR = os.path.join(PLATE_DIR, 'processedImages')
WELLS = ['A1_03', 'B5_03', 'D7_03', 'F10_03', 'H12_03']

for well in WELLS:
    rawPath = os.path.join(PROC_DIR, f'{well}_registered_raw.tif')
    maskPath = os.path.join(PROC_DIR, f'{well}_masks.npz')
    outPath = os.path.join(PROC_DIR, f'{well}_overlay.mp4')

    if not os.path.exists(rawPath) or not os.path.exists(maskPath):
        print(f'{well}: missing files, skipping')
        continue

    raw = iio.imread(rawPath).astype(np.float32)
    imax = raw.max()
    if imax > 0:
        raw /= imax

    masks = np.load(maskPath)['masks']
    nframes = raw.shape[2]

    fpMean = 0.5 * (np.nanmax(raw) + np.nanmin(raw))
    displayStack = normalize_local_contrast_output(raw, 101, fpMean)
    procVis = np.clip(displayStack, 0.0, 1.0)

    hEven = procVis.shape[0] & ~1
    wEven = procVis.shape[1] & ~1

    alpha = np.float32(0.35)
    cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    with imageio.get_writer(
        outPath, fps=2, codec='libx264',
        quality=8, pixelformat='yuv420p',
        macro_block_size=16
    ) as writer:
        for t in range(nframes):
            gray = procVis[:hEven, :wEven, t]
            frame = np.stack([gray, gray, gray], axis=-1)
            maskT = masks[:hEven, :wEven, t]
            frame[maskT] = (1.0 - alpha) * frame[maskT] + alpha * cyan
            writer.append_data(np.clip(frame * 255, 0, 255).astype(np.uint8))

    sizeMb = os.path.getsize(outPath) / 1e6
    print(f'{well}: {nframes} frames, {sizeMb:.1f} MB -> {outPath}')
