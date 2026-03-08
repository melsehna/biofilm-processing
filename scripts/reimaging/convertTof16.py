#!/usr/bin/env python3

import numpy as np
import tifffile
from pathlib import Path
import os
import sys

ROOTS = [
    Path('/mnt/data/trainingData'),
    Path('/mnt/data/reimaging/processed'),
]

AXES_METADATA = {'axes': 'YXT'}


def bytes_human(n):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.2f}{unit}'
        n /= 1024
    return f'{n:.2f}PB'


def convert_one(path: Path):
    try:
        before_size = path.stat().st_size
        arr = tifffile.imread(path)

        if arr.dtype == np.float32:
            return before_size, before_size

        tmp = path.with_suffix(path.suffix + '.tmp')

        tifffile.imwrite(
            tmp,
            arr.astype(np.float32, copy=False),
            dtype=np.float32,
            metadata=AXES_METADATA,
            compression='zlib'
        )

        tmp.replace(path)

        after_size = path.stat().st_size

        print(
            f'[OK] {path}  '
            f'({arr.dtype} → float32, '
            f'{bytes_human(before_size)} → {bytes_human(after_size)})'
        )

        return before_size, after_size

    except Exception as e:
        print(f'[ERR] {path}: {e}', file=sys.stderr)
        return 0, 0


def process_root(root: Path):
    tifs = list(root.rglob('processedImages/*.tif'))
    print(f'\nScanning {root}')
    print(f'Found {len(tifs)} TIFFs')

    total_before = 0
    total_after = 0

    for tif in tifs:
        b, a = convert_one(tif)
        total_before += b
        total_after += a

    return total_before, total_after


def main():
    grand_before = 0
    grand_after = 0

    for root in ROOTS:
        b, a = process_root(root)
        grand_before += b
        grand_after += a

    print('\n================ SUMMARY ================')
    print(f'Before: {bytes_human(grand_before)}')
    print(f'After:  {bytes_human(grand_after)}')
    print(f'Saved:  {bytes_human(grand_before - grand_after)}')
    print('=========================================')


if __name__ == '__main__':
    main()
