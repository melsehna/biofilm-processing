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
from multiWellAnalysis.processing.preprocessing import normalizeLocalContrastOutput
from multiWellAnalysis.processing.overlay import writeOverlayVideo


def findWells(procDir):
    """Discover wells from *_registered_raw.tif files in procDir."""
    wells = []
    for f in sorted(os.listdir(procDir)):
        if f.endswith('_registered_raw.tif'):
            well = f.replace('_registered_raw.tif', '')
            wells.append(well)
    return wells


def loadMutantMap(indexPath):
    """Load mutant labels from an index CSV. Returns {wellId: mutant} dict."""
    if not indexPath or not os.path.exists(indexPath):
        return {}
    df = pd.read_csv(indexPath)

    mutantMap = {}
    wellCol = 'wellId' if 'wellId' in df.columns else 'well'
    if wellCol not in df.columns:
        return {}

    for col in ['geneName', 'mutant']:
        if col in df.columns:
            for _, row in df.iterrows():
                w = str(row[wellCol]).strip()
                m = str(row[col]).strip()
                if m and m != 'nan':
                    mutantMap[w] = m
            break

    return mutantMap


def regenOverlay(procDir, well, blockDiam, fps, label=None):
    """Regenerate a single overlay MP4."""
    rawPath = os.path.join(procDir, f'{well}_registered_raw.tif')
    maskPath = os.path.join(procDir, f'{well}_masks.npz')
    outPath = os.path.join(procDir, f'{well}_overlay.mp4')

    if not os.path.exists(rawPath) or not os.path.exists(maskPath):
        print(f'{well}: missing files, skipping')
        return

    raw = iio.imread(rawPath).astype(np.float32)
    imax = raw.max()
    if imax > 0:
        raw /= imax

    masks = np.load(maskPath)['masks']

    fpMean = 0.5 * (np.nanmax(raw) + np.nanmin(raw))
    displayStack = normalizeLocalContrastOutput(raw, blockDiam, fpMean)
    procVis = np.clip(displayStack, 0.0, 1.0)

    writeOverlayVideo(procVis, masks, outPath, fps=fps, label=label)

    sizeMb = os.path.getsize(outPath) / 1e6
    nframes = raw.shape[2]
    print(f'{well}: {nframes} frames, {sizeMb:.1f} MB -> {outPath}')


def main():
    parser = argparse.ArgumentParser(description='Regenerate overlay MP4s from processed stacks and masks')
    parser.add_argument('plate_dir', help='Plate directory containing processedImages/ or Processed_images_py/')
    parser.add_argument('--wells', nargs='+', help='Specific wells to regenerate (default: all)')
    parser.add_argument('--index', help='Index CSV with mutant labels (columns: wellId/well, geneName/mutant)')
    parser.add_argument('--block-diam', type=int, default=101, help='Block diameter for normalization (default: 101)')
    parser.add_argument('--fps', type=int, default=2, help='Output video FPS (default: 2)')
    args = parser.parse_args()

    procDir = None
    for subdir in ['processedImages', 'Processed_images_py']:
        candidate = os.path.join(args.plate_dir, subdir)
        if os.path.isdir(candidate):
            procDir = candidate
            break

    if procDir is None:
        print(f'No processedImages/ or Processed_images_py/ found in {args.plate_dir}')
        sys.exit(1)

    wells = args.wells if args.wells else findWells(procDir)
    if not wells:
        print(f'No wells found in {procDir}')
        sys.exit(1)

    mutantMap = loadMutantMap(args.index)
    plateName = os.path.basename(os.path.normpath(args.plate_dir))

    print(f'Regenerating overlays for {len(wells)} well(s) in {procDir}')

    for well in wells:
        wellBase = well.split('_')[0]
        mutant = mutantMap.get(wellBase) or mutantMap.get(well)
        if mutant:
            label = f'{mutant}  {plateName}-{wellBase}'
        else:
            label = f'{plateName}-{well}'

        regenOverlay(procDir, well, args.block_diam, args.fps, label=label)

    print('Done.')


if __name__ == '__main__':
    main()
