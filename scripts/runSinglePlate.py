#!/usr/bin/env python3
"""Process a single plate directory with magnification filtering.

Usage:
    python scripts/runSinglePlate.py <plateDir> -o <outdir> -m <mag_suffix>

Example:
    python scripts/runSinglePlate.py '/mnt/data/plates/Plate1' -o /mnt/data/output -m _03
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import argparse
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from multiWellAnalysis.processing.batch_runner import run_plate, discover_mag_groups
from multiWellAnalysis.processing.helpers import round_odd
import glob as globmod


def main():
    parser = argparse.ArgumentParser(description='Process a single plate with magnification filtering')
    parser.add_argument('plateDir', help='Path to plate directory containing .tif images')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory (default: inside plateDir)')
    parser.add_argument('-m', '--mag', default=None,
                        help='Magnification suffix to process, e.g. _03 (default: all)')
    parser.add_argument('-r', '--replicates', default='/home/smellick/ImageLibrary/ReplicatePositions.csv',
                        help='Path to ReplicatePositions.csv')
    parser.add_argument('--skip-overlay', action='store_true',
                        help='Skip overlay video generation')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if results exist')
    parser.add_argument('--block-diam', type=int, default=101,
                        help='Block diameter (default: 101)')
    parser.add_argument('--fixed-thresh', type=float, default=0.014,
                        help='Fixed threshold (default: 0.014)')
    args = parser.parse_args()

    plate_dir = args.plateDir
    if not os.path.isdir(plate_dir):
        print(f'Error: plate directory not found: {plate_dir}')
        sys.exit(1)

    # If outdir specified, symlink-or-copy approach: process in outdir
    # For simplicity, run_plate outputs into plate_dir's subdirs,
    # so we set plate_dir to outdir if specified and copy relevant files
    effective_dir = plate_dir
    if args.outdir:
        plate_name = os.path.basename(os.path.normpath(plate_dir))
        effective_dir = os.path.join(args.outdir, plate_name)
        os.makedirs(effective_dir, exist_ok=True)

        # Symlink tifs from source if not already present
        for f in globmod.glob(os.path.join(plate_dir, '*.tif')):
            dst = os.path.join(effective_dir, os.path.basename(f))
            if not os.path.exists(dst):
                os.symlink(f, dst)

        # Copy protocol.csv and metadata.csv if present
        for meta_file in ['protocol.csv', 'metadata.csv']:
            src = os.path.join(plate_dir, meta_file)
            dst = os.path.join(effective_dir, meta_file)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    mutant_map = dict(
        pd.read_csv(args.replicates)
        .set_index('Header')['Replicate ID']
    )

    params = {
        'blockDiam': round_odd(args.block_diam),
        'fixed_thresh': args.fixed_thresh,
        'shift_thresh': 50,
        'dust_correction': True,
        'Imin': None,
        'Imax': None,
    }

    plate_name = os.path.basename(os.path.normpath(plate_dir))
    mag_str = args.mag or 'all'
    print(f'Plate:         {plate_name}')
    print(f'Source:        {plate_dir}')
    print(f'Output:        {effective_dir}')
    print(f'Magnification: {mag_str}')

    t0 = time.perf_counter()

    # If a specific magnification is requested, filter mag_groups
    # by temporarily monkey-patching discover_mag_groups
    if args.mag:
        _orig_discover = discover_mag_groups.__wrapped__ if hasattr(discover_mag_groups, '__wrapped__') else None

        import multiWellAnalysis.processing.batch_runner as br
        _orig_fn = br.discover_mag_groups

        def _filtered_discover(plate_dir, tif_files):
            groups = _orig_fn(plate_dir, tif_files)
            mag = args.mag
            # Match by exact key or by suffix
            filtered = {k: v for k, v in groups.items() if k == mag or k == mag.lstrip('_')}
            if not filtered:
                print(f'  Warning: mag={mag} not found in groups: {list(groups.keys())}')
                return groups
            return filtered

        br.discover_mag_groups = _filtered_discover

    result = run_plate(
        effective_dir, mutant_map, params,
        force=args.force, skip_overlay=args.skip_overlay
    )

    elapsed = time.perf_counter() - t0
    print(f'\nFinished in {elapsed / 60:.1f} minutes')

    if result is not None:
        print(f'Processed {len(result)} well-timepoints')


if __name__ == '__main__':
    main()
