#!/usr/bin/env python3
"""Regenerate overlay MP4s from existing processed stacks and masks.

Usage:
    # Regenerate overlays for a single plate directory
    python scripts/regenOverlays.py /mnt/data/plates/241106_Plate1

    # Regenerate overlays for specific wells
    python scripts/regenOverlays.py /mnt/data/plates/241106_Plate1 --wells A1_03 B5_03

    # Use a timeseries CSV or index CSV for mutant labels
    python scripts/regenOverlays.py /mnt/data/plates/241106_Plate1 --index /mnt/data/indices/index.csv

    # Custom block diameter and fps
    python scripts/regenOverlays.py /mnt/data/plates/241106_Plate1 --block-diam 101 --fps 6
"""

import argparse
import os
import sys
import numpy as np
import imageio.v3 as iio
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from multiWellAnalysis.processing.preprocessing import normalize_local_contrast_output
from multiWellAnalysis.processing.overlay import write_overlay_video


def find_wells(proc_dir):
    """Discover wells from *_registered_raw.tif files in proc_dir."""
    wells = []
    for f in sorted(os.listdir(proc_dir)):
        if f.endswith('_registered_raw.tif'):
            well = f.replace('_registered_raw.tif', '')
            wells.append(well)
    return wells


def load_mutant_map(index_path):
    """Load mutant labels from an index CSV. Returns {wellId: mutant} dict."""
    if not index_path or not os.path.exists(index_path):
        return {}
    df = pd.read_csv(index_path)

    mutant_map = {}
    well_col = 'wellId' if 'wellId' in df.columns else 'well'
    if well_col not in df.columns:
        return {}

    for col in ['geneName', 'mutant']:
        if col in df.columns:
            for _, row in df.iterrows():
                w = str(row[well_col]).strip()
                m = str(row[col]).strip()
                if m and m != 'nan':
                    mutant_map[w] = m
            break

    return mutant_map


def regen_overlay(proc_dir, well, block_diam, fps, label=None):
    """Regenerate a single overlay MP4."""
    raw_path = os.path.join(proc_dir, f'{well}_registered_raw.tif')
    mask_path = os.path.join(proc_dir, f'{well}_masks.npz')
    out_path = os.path.join(proc_dir, f'{well}_overlay.mp4')

    if not os.path.exists(raw_path) or not os.path.exists(mask_path):
        print(f'{well}: missing files, skipping')
        return

    raw = iio.imread(raw_path).astype(np.float32)
    imax = raw.max()
    if imax > 0:
        raw /= imax

    masks = np.load(mask_path)['masks']

    fpMean = 0.5 * (np.nanmax(raw) + np.nanmin(raw))
    display_stack = normalize_local_contrast_output(raw, block_diam, fpMean)
    proc_vis = np.clip(display_stack, 0.0, 1.0)

    write_overlay_video(proc_vis, masks, out_path, fps=fps, label=label)

    size_mb = os.path.getsize(out_path) / 1e6
    nframes = raw.shape[2]
    print(f'{well}: {nframes} frames, {size_mb:.1f} MB -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Regenerate overlay MP4s from processed stacks and masks')
    parser.add_argument('plate_dir', help='Plate directory containing processedImages/ or Processed_images_py/')
    parser.add_argument('--wells', nargs='+', help='Specific wells to regenerate (default: all)')
    parser.add_argument('--index', help='Index CSV with mutant labels (columns: wellId/well, geneName/mutant)')
    parser.add_argument('--block-diam', type=int, default=101, help='Block diameter for normalization (default: 101)')
    parser.add_argument('--fps', type=int, default=2, help='Output video FPS (default: 2)')
    args = parser.parse_args()

    proc_dir = None
    for subdir in ['processedImages', 'Processed_images_py']:
        candidate = os.path.join(args.plate_dir, subdir)
        if os.path.isdir(candidate):
            proc_dir = candidate
            break

    if proc_dir is None:
        print(f'No processedImages/ or Processed_images_py/ found in {args.plate_dir}')
        sys.exit(1)

    wells = args.wells if args.wells else find_wells(proc_dir)
    if not wells:
        print(f'No wells found in {proc_dir}')
        sys.exit(1)

    mutant_map = load_mutant_map(args.index)
    plate_name = os.path.basename(os.path.normpath(args.plate_dir))

    print(f'Regenerating overlays for {len(wells)} well(s) in {proc_dir}')

    for well in wells:
        well_base = well.split('_')[0]
        mutant = mutant_map.get(well_base) or mutant_map.get(well)
        if mutant:
            label = f'{mutant}  {plate_name}-{well_base}'
        else:
            label = f'{plate_name}-{well}'

        regen_overlay(proc_dir, well, args.block_diam, args.fps, label=label)

    print('Done.')


if __name__ == '__main__':
    main()
