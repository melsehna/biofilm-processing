#!/usr/bin/env python3
"""Headless single-well parameter verification (the GUI Test Well tab, no display).

Runs the real per-well pipeline — `_processOneWell` (+ optional `_trackOneWell`)
from `gui/tabs/run.py` — on ONE well with a given parameter set, and writes the
standard artifacts to a scratch dir so you can eyeball whether the parameters are
right before launching a full run:

    <output-dir>/<plate>/processedImages/
        <well>_processed.tif          display-normalized stack
        <well>_registered_raw.tif     phase-corrected raw
        <well>_masks.npz              binary masks
        <well>_overlay.mp4            mask overlay video
        <well>_biomass.csv            biomass curve
        <well>_trackedLabels_*.npz    (with --tracking)
        <well>_testMontage.png        N evenly-spaced frames, mask edges in red
        <well>_biomass.png            biomass curve plot

Parameters resolve exactly like a real run: GUI defaults < --config
experiment_config.json < per-mag overrides (magParams) < CLI flags. So whatever
looks right here is what the full run will use.

Usage:
    biofilm-processing-test-well /data/plateA --well B2 --mag _03 \
        --fixed-thresh 0.03 --block-diam 121 --tracking
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import argparse

import numpy as np

from ..gui.tabs.run import (
    discoverWells, _computeOutdir, _processOneWell, _trackOneWell,
)
from .run_pipeline import buildState


def _resolveWell(plate, well, mag):
    """Return (resolvedPlate, wellKey, wellFiles). Raises ValueError on miss."""
    resolved, wells = discoverWells(plate, mag or 'all')
    if not wells:
        raise ValueError(f'No wells found in {plate} (mag={mag or "all"}).')

    if well in wells:
        key = well
    elif mag and f'{well}{mag}' in wells:
        key = f'{well}{mag}'
    else:
        matches = [k for k in wells if k == well or k.startswith(well + '_')]
        if len(matches) == 1:
            key = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f'Well {well!r} is ambiguous across mags: {sorted(matches)}. '
                f'Pass --mag to disambiguate.')
        else:
            raise ValueError(
                f'Well {well!r} not found. Available: {sorted(wells)[:20]}'
                f'{" …" if len(wells) > 20 else ""}')

    return resolved, key, wells[key]


def _writeBiomassPlot(biomassCsv, outPng, seedFrame=None, peakFrame=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(biomassCsv)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(df['frame'], df['biomass'], '-o', ms=3)
    if seedFrame is not None:
        ax.axvline(seedFrame, color='g', ls='--', lw=1, label=f'seed {seedFrame}')
    if peakFrame is not None:
        ax.axvline(peakFrame, color='r', ls='--', lw=1, label=f'peak {peakFrame}')
    ax.set_xlabel('frame')
    ax.set_ylabel('biomass')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outPng, dpi=110)
    plt.close(fig)


def _writeMontage(processedTif, masksNpz, outPng, nFrames=6):
    """N evenly-spaced frames: grayscale processed render + red mask outline."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import tifffile

    proc = tifffile.imread(processedTif)
    if proc.ndim == 3 and proc.shape[0] < proc.shape[1]:
        proc = np.transpose(proc, (1, 2, 0))  # -> (H, W, T)

    md = np.load(masksNpz)
    masks = md['masks'] if 'masks' in md else md[list(md.keys())[0]]
    if masks.ndim == 3 and masks.shape[0] < masks.shape[1]:
        masks = np.transpose(masks, (1, 2, 0))

    T = min(proc.shape[2], masks.shape[2])
    idxs = np.unique(np.linspace(0, T - 1, min(nFrames, T)).astype(int))

    cols = min(len(idxs), 3)
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for n, t in enumerate(idxs):
        ax = axes[n // cols][n % cols]
        ax.imshow(proc[..., t], cmap='gray')
        ax.contour(masks[..., t].astype(float), levels=[0.5], colors='red', linewidths=0.6)
        ax.set_title(f'frame {t}', fontsize=9)
    fig.suptitle(os.path.basename(outPng).replace('_testMontage.png', ''), fontsize=10)
    fig.tight_layout()
    fig.savefig(outPng, dpi=110)
    plt.close(fig)


def buildParser():
    p = argparse.ArgumentParser(
        prog='biofilm-processing-test-well',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('plate', help='Plate (or drawer) directory.')
    p.add_argument('--well', required=True,
                   help="Well id, e.g. B2 or B2_03. Use --mag if a bare well is "
                        "ambiguous across magnifications.")
    p.add_argument('--mag', default=None,
                   help='Magnification suffix, e.g. _03.')
    p.add_argument('--config', default=None,
                   help='experiment_config.json to seed parameters (optional).')
    p.add_argument('--output-dir', dest='outputDir', default=None,
                   help='Scratch output root (default: ./testWell_out).')
    p.add_argument('--tracking', action='store_true',
                   help='Also run colony tracking and write tracked labels.')
    p.add_argument('--montage-frames', type=int, default=6,
                   help='Frames in the verification montage PNG (default 6).')
    p.add_argument('--no-montage', action='store_true',
                   help='Skip the montage / biomass PNGs (artifacts only).')

    # Parameter overrides (same names/keys as the full runner).
    p.add_argument('--block-diam', dest='blockDiam', type=int, default=None)
    p.add_argument('--fixed-thresh', dest='fixedThresh', type=float, default=None)
    p.add_argument('--shift-thresh', dest='shiftThresh', type=int, default=None)
    p.add_argument('--fft-stride', dest='fftStride', type=int, default=None)
    p.add_argument('--downsample', dest='downsample', type=int, default=None)
    p.add_argument('--dust-correction', dest='dustCorrection',
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--min-colony-area', dest='minColonyAreaPx', type=int, default=None)
    p.add_argument('--prop-radius', dest='propRadiusPx', type=int, default=None)
    p.add_argument('--overlays', dest='saveOverlays',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Write the overlay MP4 (default on).')
    return p


def main(argv=None):
    from multiWellAnalysis.processing.helpers import roundOdd

    args = buildParser().parse_args(argv)

    overrides = {
        'blockDiam':       roundOdd(args.blockDiam) if args.blockDiam else None,
        'fixedThresh':     args.fixedThresh,
        'shiftThresh':     args.shiftThresh,
        'fftStride':       args.fftStride,
        'downsample':      args.downsample,
        'dustCorrection':  args.dustCorrection,
        'minColonyAreaPx': args.minColonyAreaPx,
        'propRadiusPx':    args.propRadiusPx,
        'saveOverlays':    args.saveOverlays,
    }
    state = buildState(args.config, overrides)

    # Per-mag overrides apply on top, exactly like the GUI Test Well tab.
    if args.mag and args.mag in state.get('magParams', {}):
        state.update(state['magParams'][args.mag])

    try:
        resolved, wellKey, wellFiles = _resolveWell(args.plate, args.well, args.mag)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    outputRoot = (args.outputDir or os.path.join(os.getcwd(), 'testWell_out'))
    procDir = _computeOutdir(args.plate, resolved, outputRoot)
    os.makedirs(procDir, exist_ok=True)
    plateName = os.path.basename(resolved)

    print(f'Plate:   {resolved}')
    print(f'Well:    {wellKey}  ({len(wellFiles) if not isinstance(wellFiles, str) else 1} frame file(s))')
    print(f'Output:  {procDir}')
    print(f'Params:  blockDiam={state["blockDiam"]} fixedThresh={state["fixedThresh"]} '
          f'shiftThresh={state["shiftThresh"]} fftStride={state.get("fftStride")} '
          f'downsample={state.get("downsample")} dust={state["dustCorrection"]}')
    print('\nProcessing…', flush=True)

    row = _processOneWell(args.plate, procDir, wellKey, wellFiles, state)
    if row.get('status') != 'done':
        print(f'ERROR: processing failed: {row.get("error", row)}', file=sys.stderr)
        return 1
    print(f'  processed in {row["elapsed"]:.1f}s')

    import pandas as pd
    bdf = pd.read_csv(row['biomass'])
    biomass = bdf['biomass'].values
    peakFrame = int(np.argmax(biomass)) if len(biomass) else 0
    seedFrame = None
    try:
        from multiWellAnalysis.colony.runTrackingGUI import findSeedFrame
        seedFrame = findSeedFrame(biomass)
    except Exception:
        pass
    print(f'  frames={len(biomass)}  biomass min/max={biomass.min():.4f}/{biomass.max():.4f}  '
          f'peakFrame={peakFrame}  seedFrame={seedFrame}')

    if args.tracking:
        print('\nTracking…', flush=True)
        trk = _trackOneWell(plateName, row, state)
        if trk.get('status') == 'done':
            import numpy as _np
            td = _np.load(trk['tracked_labels'])
            labels = td['labels'] if 'labels' in td else None
            nColonies = int(labels.max()) if labels is not None else 0
            print(f'  tracked in {trk["elapsed"]:.1f}s  colonies(max label)={nColonies}')
            print(f'  tracked labels: {trk["tracked_labels"]}')
        else:
            print(f'  tracking {trk.get("status")}: {trk.get("reason", trk.get("error", ""))}')

    if not args.no_montage:
        print('\nVerification images…', flush=True)
        try:
            montagePng = os.path.join(procDir, f'{wellKey}_testMontage.png')
            _writeMontage(row['processed'], row['masks'], montagePng,
                          nFrames=args.montage_frames)
            biomassPng = os.path.join(procDir, f'{wellKey}_biomass.png')
            _writeBiomassPlot(row['biomass'], biomassPng, seedFrame, peakFrame)
            print(f'  montage: {montagePng}')
            print(f'  biomass: {biomassPng}')
        except Exception as e:
            print(f'  WARNING: could not write verification images: {e}')

    print('\nInspect these to verify parameters:')
    print(f'  overlay video: {row["processed"].replace("_processed.tif", "_overlay.mp4")}')
    print(f'  processed tif: {row["processed"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
