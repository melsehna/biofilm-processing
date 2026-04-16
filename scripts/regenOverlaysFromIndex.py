#!/usr/bin/env python3
"""Regenerate overlay MP4s for all wells in an index CSV.

Usage:
    python scripts/regenOverlaysFromIndex.py index.csv --outdir /mnt/phenotyper/vcReimaging

    # Custom settings
    python scripts/regenOverlaysFromIndex.py index.csv --outdir /out --block-diam 101 --fps 2 --workers 8

    # Dry run to check paths
    python scripts/regenOverlaysFromIndex.py index.csv --outdir /out --dry-run
"""

import argparse
import os
import sys
import time
import numpy as np
import tifffile
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from multiWellAnalysis.processing.preprocessing import normalizeLocalContrastOutput
from multiWellAnalysis.processing.overlay import writeOverlayVideo


GILLIUS_FONT = '/usr/share/fonts/truetype/adf/GilliusADF-Regular.otf'
BLOCK_DIAM = 101


def regenOne(args):
    """Regenerate a single overlay. Called by the worker pool."""
    rawPath, maskPath, outPath, label, fps, blockDiam = args
    try:
        if not os.path.exists(rawPath):
            return f'SKIP missing raw: {rawPath}'
        if not os.path.exists(maskPath):
            return f'SKIP missing mask: {maskPath}'

        raw = tifffile.imread(rawPath).astype(np.float32)
        # registeredRawTif is (T, H, W) — transpose to (H, W, T)
        if raw.ndim == 3 and raw.shape[0] < raw.shape[1]:
            raw = np.transpose(raw, (1, 2, 0))
        # scale to [0, 1]
        imax = raw.max()
        if imax > 0:
            raw /= imax

        masks = np.load(maskPath)['masks']  # already (H, W, T) bool

        fpMean = 0.5 * (np.nanmax(raw) + np.nanmin(raw))
        displayStack = normalizeLocalContrastOutput(raw, blockDiam, fpMean)
        displayStack = np.clip(displayStack, 0.0, 1.0)

        writeOverlayVideo(displayStack, masks, outPath, fps=fps, label=label,
                          fontPath=GILLIUS_FONT)

        sizeMb = os.path.getsize(outPath) / 1e6
        nframes = displayStack.shape[2]
        return f'{label}: {nframes} frames, {sizeMb:.1f} MB'
    except Exception as e:
        return f'ERROR {outPath}: {e}'


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate overlay MP4s from an index CSV with rawPath/maskPath columns')
    parser.add_argument('index', help='Index CSV with rawPath, maskPath, wellId, plateId, geneName columns')
    parser.add_argument('--outdir', required=True, help='Root output directory (e.g., /mnt/phenotyper/vcReimaging)')
    parser.add_argument('--src-root', default='/mnt/data/reimaging/processed',
                        help='Source root to strip from rawPath when building output paths (default: /mnt/data/reimaging/processed)')
    parser.add_argument('--block-diam', type=int, default=BLOCK_DIAM, help='Block diameter for normalization (default: 101)')
    parser.add_argument('--fps', type=int, default=2, help='Output video FPS (default: 2)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers (default: 8)')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without generating videos')
    parser.add_argument('--skip-existing', action='store_true', help='Skip wells that already have overlay MP4s')
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.index)

    required = ['rawPath', 'maskPath', 'wellId', 'plateId']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f'Missing required columns: {missing}')
        sys.exit(1)

    hasGene = 'geneName' in df.columns

    # build work list and collect output dirs
    outDirs = set()
    tasks = []
    for _, row in df.iterrows():
        rawPath = str(row['rawPath'])
        maskPath = str(row['maskPath'])
        wellId = str(row['wellId'])
        plateId = str(row['plateId'])
        geneName = str(row['geneName']).strip() if hasGene else ''

        # build output path: strip src-root, replace processedImages/ with overlays/
        relPath = os.path.relpath(rawPath, args.src_root)
        parts = relPath.split(os.sep)
        parts = ['overlays' if p == 'processedImages' else p for p in parts]
        outDir = os.path.join(args.outdir, os.sep.join(parts[:-1]))
        outPath = os.path.join(outDir, f'{wellId}_overlay.mp4')
        outDirs.add(outDir)

        if args.skip_existing and os.path.exists(outPath) and os.path.getsize(outPath) > 0:
            continue

        if geneName and geneName != 'nan':
            label = f'{geneName}  {plateId}-{wellId}'
        else:
            label = f'{plateId}-{wellId}'

        tasks.append((rawPath, maskPath, outPath, label, args.fps, args.block_diam))

    print(f'{len(tasks)} overlays to generate ({len(df)} total in index)')

    if args.dry_run:
        for t in tasks[:10]:
            print(f'  {t[3]} -> {t[2]}')
        if len(tasks) > 10:
            print(f'  ... and {len(tasks) - 10} more')
        return

    # create all output directories upfront
    for d in sorted(outDirs):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    done = 0
    errors = 0
    skipped = 0

    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(regenOne, tasks):
            done += 1
            if result.startswith('SKIP'):
                skipped += 1
            elif result.startswith('ERROR'):
                errors += 1
                print(f'[{done}/{len(tasks)}] {result}')
            elif done % 10 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f'[{done}/{len(tasks)}] {rate:.1f} wells/s, ETA {eta/60:.1f} min — {result}')

    elapsed = time.time() - t0
    print(f'\nDone: {done - errors - skipped}/{len(tasks)} overlays in {elapsed/60:.1f} min ({errors} errors)')


if __name__ == '__main__':
    main()
