#!/usr/bin/env python3

import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from multiWellAnalysis.colony.io_utils import loadRawStack

# configs

PLATE_ID = '241010_105227_Plate_1'
DATA_ROOT = '/mnt/data/trainingData'
OUTDIR = f'{DATA_ROOT}/{PLATE_ID}/processedImages/gifs'
WELLS = ['A4', 'A5', 'A9', 'A11']

LABEL_VERSION = 'trackingVec_v1'

FPS = 5
ALPHA = 0.45

os.makedirs(OUTDIR, exist_ok=True)


# helpers

def make_overlay(raw, labels, colors, alpha):
    overlay = np.zeros((*labels.shape, 4), dtype=float)

    for lab in range(1, labels.max() + 1):
        sel = labels == lab
        if not sel.any():
            continue
        overlay[sel, :3] = colors[lab]
        overlay[sel, 3] = alpha

    return overlay


# main

def process_well(well_id):
    print(f'Processing well {well_id}')

    raw_path = f'{DATA_ROOT}/{PLATE_ID}/processedImages/{well_id}_registered_raw.tif'
    label_path = (
        f'{DATA_ROOT}/{PLATE_ID}/processedImages/'
        f'{well_id}_trackedLabels_allFrames_{LABEL_VERSION}.npz'
    )

    if not os.path.exists(label_path):
        print(f'  missing labels for {well_id}, skipping')
        return

    raw_stack = loadRawStack(raw_path)
    label_npz = np.load(label_path)

    labels = label_npz['labels']
    frames = label_npz['frames']
    biomass_valid = label_npz['biomassValidFrames']

    H, W, n_frames = labels.shape

    max_label = labels.max()
    rng = np.random.default_rng(0)
    colors = rng.random((max_label + 1, 3))

    out_gif = f'{OUTDIR}/{well_id}_tracking.gif'
    writer = imageio.get_writer(out_gif, fps=FPS)

    for i, t in enumerate(frames):
        raw = raw_stack[:, :, t]
        lab = labels[:, :, i]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(raw, cmap='gray', interpolation='nearest')

        if lab.max() > 0:
            overlay = make_overlay(raw, lab, colors, ALPHA)
            ax.imshow(overlay, interpolation='nearest')

        if not biomass_valid[i]:
            ax.set_title('below biomass threshold', color='red', fontsize=8)

        ax.axis('off')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame)

        plt.close(fig)

    writer.close()
    print(f'  wrote {out_gif}')


def main():
    for well in WELLS:
        process_well(well)


if __name__ == '__main__':
    main()
