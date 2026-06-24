#!/usr/bin/env python3
"""Headless driver for the full processing pipeline (HPC / no-GUI servers).

Runs the *same* pipeline the GUI Run tab runs — `ProcessingWorker` in
`gui/tabs/run.py` — without a display or event loop. The worker takes a plain
state dict plus a `threading.Event`, and emits `log`/`overallProgress` Qt
signals; here we connect those to stdout and call `worker.run()` directly on
the main thread (the worker fans out per-well work via ProcessPoolExecutor, so
no QThread is needed). This deliberately reuses the canonical worker rather
than `processing/batch_runner.py`, which is a simpler, divergent path that does
not produce tracking / colony / whole-image features, master CSVs, or UMAPs.

Usage:
    # From a GUI-saved config, overriding output + workers for the cluster:
    biofilm-processing-run experiment_config.json \
        --output-dir /scratch/$USER/run1 --workers 40

    # Fully from flags, no config file:
    biofilm-processing-run \
        --plates /data/plateA /data/plateB \
        --output-dir /scratch/$USER/run1 \
        --mag _03 --workers 40 \
        --whole-image --colony-tracking --colony-feats

`config` is an experiment_config.json saved from the GUI (File > Save). Any
flag overrides the corresponding config value; flags left unset fall back to
the config (or to GUI defaults when no config is given). `--plates` and
`--output-dir` are required either via the config or on the command line.
"""

# Thread-limiting + headless Qt must be set before numpy / PySide6 import, so
# the per-process BLAS pools don't oversubscribe cores under ProcessPoolExecutor
# and Qt never tries to open a display.
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import json
import signal
import argparse
import threading

from ..gui.state import DEFAULTS
from ..gui.tabs.run import ProcessingWorker
from ..processing.helpers import roundOdd

# Keys carried from a saved config even though they are not in DEFAULTS.
# AppState.from_dict() drops these on load, so the GUI re-picks them each
# session; headless we must preserve them from the raw JSON instead.
_EXTRA_KEYS = ('outputDir', 'plates')


def buildState(configPath, overrides):
    """Build the worker state dict: DEFAULTS < config file < CLI overrides.

    overrides is a {stateKey: value} dict where None means "not specified".
    """
    state = dict(DEFAULTS)

    if configPath:
        with open(configPath, 'r') as f:
            raw = json.load(f)
        for k, v in raw.items():
            if k in DEFAULTS or k in _EXTRA_KEYS:
                state[k] = v

    for k, v in overrides.items():
        if v is not None:
            state[k] = v

    return state


def runHeadless(state, logFn=print, progressFn=None, stopEvent=None):
    """Run the pipeline to completion. Returns 0 on success, non-zero on error.

    Reuses ProcessingWorker verbatim: connect its signals, call run().
    """
    stopEvent = stopEvent or threading.Event()
    worker = ProcessingWorker(state, stopEvent)

    errBox = {'msg': None}

    worker.log.connect(logFn)
    if progressFn is not None:
        worker.overallProgress.connect(progressFn)
    worker.error.connect(lambda m: errBox.__setitem__('msg', m))

    # SIGINT (Ctrl-C) and SIGTERM (SLURM scancel) → cooperative cancel; the
    # worker checks stopEvent between wells and stages and exits cleanly.
    def _onSignal(signum, frame):
        logFn(f'\nReceived signal {signum} — finishing current well then stopping…')
        stopEvent.set()

    prevInt = signal.signal(signal.SIGINT, _onSignal)
    prevTerm = signal.signal(signal.SIGTERM, _onSignal)
    try:
        worker.run()
    finally:
        signal.signal(signal.SIGINT, prevInt)
        signal.signal(signal.SIGTERM, prevTerm)

    if errBox['msg']:
        logFn(f'ERROR: {errBox["msg"]}')
        return 1
    if stopEvent.is_set():
        return 130
    return 0


def _makeProgressPrinter():
    """Print overall progress, deduped, so it doesn't drown the log."""
    last = {'line': None}

    def _print(done, total, desc):
        line = f'  [progress] {done}/{total}  {desc}'
        if line != last['line']:
            print(line, flush=True)
            last['line'] = line

    return _print


def buildParser():
    p = argparse.ArgumentParser(
        prog='biofilm-processing-run',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('config', nargs='?', default=None,
                   help='experiment_config.json saved from the GUI (optional). '
                        'Flags below override its values.')

    # Inputs / outputs (required via config or flags)
    p.add_argument('--plates', nargs='+', default=None,
                   help='One or more plate directories (or drawer dirs).')
    p.add_argument('--output-dir', dest='outputDir', default=None,
                   help='Output root. Required (here or in config).')
    p.add_argument('--mag', dest='magnification', default=None,
                   help="Magnification suffix to process, e.g. _03, or 'all'.")
    p.add_argument('--workers', type=int, default=None,
                   help='Parallel workers (hard-capped at 75%% of cores).')

    # Stages (tri-state: unset = use config/default)
    p.add_argument('--whole-image', dest='wholeImageFeats',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Whole-image texture/intensity features.')
    p.add_argument('--colony-tracking', dest='colonyTracking',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Colony tracking (seeded label propagation).')
    p.add_argument('--colony-feats', dest='colonyFeats',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Per-colony features (implies tracking).')
    p.add_argument('--overlays', dest='saveOverlays',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Write mask-overlay MP4s.')
    p.add_argument('--processed-video', dest='saveProcessedVideo',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Write grayscale processed-stack MP4s (no mask overlay).')
    p.add_argument('--umap-static', dest='umapStatic',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Generate static UMAP PNGs (needs the .[umap] extra).')
    p.add_argument('--umap-interactive', dest='umapInteractive',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Generate the interactive UMAP HTML viewer.')
    p.add_argument('--umap-column', dest='umapColumnName', default=None,
                   help='Layout CSV column to color UMAP points by.')

    # Processing params
    p.add_argument('--block-diam', dest='blockDiam', type=int, default=None,
                   help='Local-contrast kernel (rounded to odd).')
    p.add_argument('--fixed-thresh', dest='fixedThresh', type=float, default=None,
                   help='Binary mask threshold on the normalized image.')
    p.add_argument('--shift-thresh', dest='shiftThresh', type=int, default=None,
                   help='Max registration shift (px).')
    p.add_argument('--fft-stride', dest='fftStride', type=int, default=None,
                   help='Registration keyframe step (1 = every frame).')
    p.add_argument('--downsample', dest='downsample', type=int, default=None,
                   help='FFT downsample factor.')
    p.add_argument('--dust-correction', dest='dustCorrection',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Remove persistent bright artifacts.')
    p.add_argument('--min-colony-area', dest='minColonyAreaPx', type=int, default=None,
                   help='Min connected-component area to label a colony (px).')
    p.add_argument('--prop-radius', dest='propRadiusPx', type=int, default=None,
                   help='Label-propagation effective radius (px).')

    # NAS mirror
    p.add_argument('--nas-mirror-dir', dest='nasMirrorDir', default=None,
                   help='Mirror each plate to this dir then delete local copy; '
                        'enables NAS mirror mode.')
    p.add_argument('--nas-lean', dest='nasMirrorLean',
                   action=argparse.BooleanOptionalAction, default=None,
                   help='Lean NAS mirror: skip large intermediates '
                        '(registered_raw.tif, masks.npz, trackedLabels.npz). '
                        'Roughly halves NAS footprint; NAS copy cannot re-run '
                        'tracking / colony features.')

    p.add_argument('--no-progress', action='store_true',
                   help='Suppress the deduped progress lines (log only).')
    return p


def main(argv=None):
    args = buildParser().parse_args(argv)

    overrides = {
        'plates':          args.plates,
        'outputDir':       args.outputDir,
        'magnification':   args.magnification,
        'workers':         args.workers,
        'wholeImageFeats': args.wholeImageFeats,
        'colonyTracking':  args.colonyTracking,
        'colonyFeats':     args.colonyFeats,
        'saveOverlays':    args.saveOverlays,
        'saveProcessedVideo': args.saveProcessedVideo,
        'umapStatic':      args.umapStatic,
        'umapInteractive': args.umapInteractive,
        'umapColumnName':  args.umapColumnName,
        'blockDiam':       roundOdd(args.blockDiam) if args.blockDiam else None,
        'fixedThresh':     args.fixedThresh,
        'shiftThresh':     args.shiftThresh,
        'fftStride':       args.fftStride,
        'downsample':      args.downsample,
        'dustCorrection':  args.dustCorrection,
        'minColonyAreaPx': args.minColonyAreaPx,
        'propRadiusPx':    args.propRadiusPx,
        'nasMirrorDir':    args.nasMirrorDir,
        'nasMirrorEnabled': True if args.nasMirrorDir else None,
        'nasMirrorLean':   args.nasMirrorLean,
    }

    state = buildState(args.config, overrides)

    plates = state.get('plates') or []
    if not plates:
        print('ERROR: no plates given. Pass --plates or a config that lists '
              'them.', file=sys.stderr)
        return 2
    if not (state.get('outputDir') or '').strip():
        print('ERROR: no output dir. Pass --output-dir or set it in the config.',
              file=sys.stderr)
        return 2

    print(f'Plates ({len(plates)}):')
    for pth in plates:
        print(f'   {pth}')
    print(f'Output:        {state["outputDir"]}')
    print(f'Magnification: {state.get("magnification", "all")}')
    print(f'Workers:       {state.get("workers")}')
    print()

    progressFn = None if args.no_progress else _makeProgressPrinter()
    rc = runHeadless(state, logFn=lambda m: print(m, flush=True),
                     progressFn=progressFn)
    print(f'\nExit code: {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
