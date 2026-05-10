#!/usr/bin/env python3
"""Run UMAP on a processed plate run's master_frame_features.csv.

Example:
    python scripts/runUmap.py /path/to/output_root --mag _03 --static --interactive

Outputs land under <outputRoot>/analysis/umap_<objective>X_*.
"""
import argparse
import json
import sys

from multiWellAnalysis.analysis.runner import runUmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputRoot',
                        help='Run output directory containing master_frame_features.csv')
    parser.add_argument('--mag', required=True,
                        help='Mag suffix to embed (e.g. _03 for 10x). One UMAP per mag.')
    parser.add_argument('--static', action='store_true',
                        help='Write static PNGs (canonical panel + 3x3 grid).')
    parser.add_argument('--interactive', action='store_true',
                        help='Write the interactive HTML viewer.')
    parser.add_argument('--layout-column', default=None,
                        help='Column in *layout*.csv to color by. Defaults to col index 1.')
    parser.add_argument('--plate-dir-map', default=None,
                        help='Path to JSON mapping plateId -> raw plate directory '
                             '(used to find per-plate *layout*.csv). Optional.')
    parser.add_argument('--frames', default=None,
                        help='Inclusive frame range as "min,max" (e.g. "0,30").')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    if not args.static and not args.interactive:
        print('Nothing to do — pass --static or --interactive (or both).',
              file=sys.stderr)
        return 1

    plateDirMap = None
    if args.plate_dir_map:
        with open(args.plate_dir_map) as f:
            plateDirMap = json.load(f)

    framesRange = None
    if args.frames:
        lo, hi = args.frames.split(',')
        framesRange = (int(lo), int(hi))

    out = runUmap(
        args.outputRoot,
        magnification=args.mag,
        doStatic=args.static,
        doInteractive=args.interactive,
        plateDirMap=plateDirMap,
        columnName=args.layout_column,
        framesRange=framesRange,
        randomState=args.random_state,
        logFn=print,
    )

    print()
    print('Wrote:')
    for k, v in out.items():
        print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
