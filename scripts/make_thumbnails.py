#!/usr/bin/env python3
"""Extract one small JPEG per well from `_processed.tif`, for embedding in plots.

Reads ONE frame per well, not the whole stack. A 25-frame 1992x1992 float32 stack
is ~374 MiB; a single page is ~15 MiB, so a 3,936-well run costs ~57 GiB of reads
instead of ~1.4 TiB. The frame is chosen per well as the peak-biomass frame, read
from the well's `_biomass.csv` (a few hundred bytes), so picking it is free.

Output: <out>/<drawerID>/<wellID>.jpg plus a manifest CSV keyed on
drawerID/plateID/wellID/mag, which joins directly to master_frame_features.csv
and to a DINOv2 cache's index.

Intensity mapping is a FIXED window, never per-image min/max. `_processed.tif` is
the fixed-fpMean=0.5 render precisely so that intensities are comparable across
wells and batches; stretching each thumbnail independently would throw that away
and make two wells with identical biology look different. The window used is
recorded in the manifest.

Disk is the scarce resource on a box whose stacks live on one spinning array, and
concurrent readers turn sequential reads into a seek storm. So this is
deliberately single-threaded, supports a throughput cap, and can yield to a
running GPU/DataLoader job.

Usage:
    python scripts/make_thumbnails.py <outputRoot> --out <thumbDir>
    python scripts/make_thumbnails.py <outputRoot> --out <thumbDir> \
        --throttle-mbps 20 --yield-to-gpu        # polite: share with an extraction
    python scripts/make_thumbnails.py <outputRoot> --out <thumbDir> --limit 20 --dry-run
"""

import os
import sys
import csv
import time
import glob
import argparse
import subprocess


def _isGpuJobRunning():
    """True while a DataLoader-backed GPU job is actively reading.

    `pt_data_worker` processes exist exactly while a torch DataLoader is feeding a
    model, which is when that job is hammering the disk. GPU *memory* is a poor
    signal: it stays allocated between plates, long after the reads stop.
    Deliberately does NOT use a pgrep pattern that could match this process's own
    command line.
    """
    try:
        out = subprocess.run(['pgrep', '-xc', 'pt_data_worker'],
                             capture_output=True, text=True)
        return out.stdout.strip().isdigit() and int(out.stdout.strip()) > 0
    except Exception:
        return False


def _identityFromProcDir(procDir, root):
    """(drawerID, plateID) for a processedImages dir, matching master_csv semantics.

    Layout is <root>/[<drawer>/]<plate>/processedImages. When a plate was reached
    through a drawer (the `Plate N` dirs of the Kleb screen), drawerID is that
    drawer; otherwise it falls back to the plate name, exactly as
    `assembleMasterCsvs` does via `drawerMap.get(plateName, plateName)`.
    """
    plateDir = os.path.dirname(procDir)
    plateID = os.path.basename(plateDir)
    parent = os.path.dirname(plateDir)
    drawerID = plateID if os.path.realpath(parent) == os.path.realpath(root) \
        else os.path.basename(parent)
    return drawerID, plateID


def _peakFrame(biomassPath, nFrames):
    """Frame index of maximum biomass, or the middle frame if unavailable."""
    try:
        import pandas as pd
        vals = pd.read_csv(biomassPath)['biomass'].values
        if len(vals):
            return int(vals.argmax()), float(vals.max())
    except Exception:
        pass
    return nFrames // 2, float('nan')


def renderThumb(page, size, window, invert):
    """float32 page -> uint8 square-ish thumbnail under a FIXED intensity window."""
    import cv2
    import numpy as np

    lo, hi = window
    img = np.clip((page.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    if invert:
        img = 1.0 - img
    h, w = img.shape[:2]
    scale = size / float(max(h, w))
    if scale < 1.0:
        # INTER_AREA is the correct filter for downsampling; it averages the
        # source pixels a target pixel covers instead of point-sampling, so fine
        # colony texture survives as tone rather than aliasing into speckle.
        img = cv2.resize(img, (max(int(round(w * scale)), 1),
                               max(int(round(h * scale)), 1)),
                         interpolation=cv2.INTER_AREA)
    return (img * 255.0).round().astype(np.uint8)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', help='pipeline output root (holds <plate>/processedImages)')
    p.add_argument('--out', required=True, help='directory to write thumbnails into')
    p.add_argument('--size', type=int, default=256, help='longest edge in px (default 256)')
    p.add_argument('--quality', type=int, default=85, help='JPEG quality (default 85)')
    p.add_argument('--window', default='0.40,0.60',
                   help='fixed intensity window "lo,hi" applied to every well '
                        '(default 0.40,0.60). NEVER per-image min/max -- see the '
                        'module docstring. Measured on this data: pixels occupy '
                        'p0.1=0.424 to p99.9=0.549, clustered around the fixed '
                        'fpMean of 0.5, so [0,1] wastes ~88%% of the 8-bit range '
                        'and every thumbnail comes out flat grey. [0.42,0.55] '
                        'maximises contrast; the default keeps margin for plates '
                        'that sit slightly wider.')
    p.add_argument('--invert', action='store_true',
                   help='invert tones for viewing, as the GUI Preview tab does. '
                        'Display only; the stored render is unchanged.')
    p.add_argument('--frame', default='peak',
                   help='"peak" (max biomass, default) or a fixed frame index')
    p.add_argument('--throttle-mbps', type=float, default=0.0,
                   help='cap read throughput, MB/s (0 = uncapped). Use when sharing '
                        'the disk with another job.')
    p.add_argument('--yield-to-gpu', action='store_true',
                   help='pause while a torch DataLoader job is running, so an '
                        'extraction keeps the disk to itself')
    p.add_argument('--workers', type=int, default=1,
                   help='parallel well readers (default 1). Raise this ONLY when the '
                        'disk is not already saturated -- check `iostat -x`. Reading '
                        'one page per well is latency-bound, not bandwidth-bound: '
                        'measured 12%% utilisation and queue depth 0.26 single- '
                        'threaded, so the disk idles between requests and more '
                        'readers fill it. That is the opposite of streaming whole '
                        'files, where extra readers only cause seek thrash.')
    p.add_argument('--limit', type=int, default=0, help='stop after N wells (smoke test)')
    p.add_argument('--sample', type=int, default=0,
                   help='render N wells spread evenly across the peak-biomass range '
                        'instead of the first N. Use this to eyeball output before '
                        'committing to a full pass: --limit takes whatever sorts '
                        'first, which on a screen like this is a run of near-empty '
                        'A-row wells and tells you nothing. Selection reads only the '
                        'per-well _biomass.csv files (a few hundred bytes each).')
    p.add_argument('--dry-run', action='store_true', help='list work, read nothing')
    args = p.parse_args(argv)

    try:
        lo, hi = (float(x) for x in args.window.split(','))
    except ValueError:
        print(f'ERROR: --window must be "lo,hi", got {args.window!r}', file=sys.stderr)
        return 2
    if hi <= lo:
        print('ERROR: --window hi must exceed lo', file=sys.stderr)
        return 2

    procDirs = sorted(glob.glob(os.path.join(args.root, '**', 'processedImages'),
                                recursive=True))
    if not procDirs:
        print(f'ERROR: no processedImages/ under {args.root}', file=sys.stderr)
        return 2

    work = []
    for procDir in procDirs:
        drawerID, plateID = _identityFromProcDir(procDir, args.root)
        for stack in sorted(glob.glob(os.path.join(procDir, '*_processed.tif'))):
            wellId = os.path.basename(stack)[:-len('_processed.tif')]
            mag = wellId.split('_')[1] if '_' in wellId else ''
            work.append((drawerID, plateID, wellId, f'_{mag}' if mag else '', stack))
    if args.sample:
        import pandas as pd
        # Peak biomass per well comes from the run-level master CSV when it
        # exists: ONE ~5 MB read instead of one tiny read per well. On a busy
        # spinning disk the difference is minutes vs seconds -- 3,936 small reads
        # are dominated by seek latency, not bytes. Falls back to the per-well
        # CSVs only if the master is absent (e.g. a cancelled run).
        peaks = {}
        masterPath = os.path.join(args.root, 'master_frame_features.csv')
        if os.path.isfile(masterPath):
            m = pd.read_csv(masterPath, usecols=['drawerID', 'wellID', 'biomass'])
            for (d, w), v in m.groupby(['drawerID', 'wellID'])['biomass'].max().items():
                peaks[(d, w)] = float(v)
        scored = []
        for row in work:
            v = peaks.get((row[0], row[2]))
            if v is None:
                bpath = os.path.join(os.path.dirname(row[4]), f'{row[2]}_biomass.csv')
                try:
                    v = float(pd.read_csv(bpath)['biomass'].max())
                except Exception:
                    continue
            scored.append((v, row))
        scored.sort(key=lambda t: t[0])
        if scored:
            n = min(args.sample, len(scored))
            idx = [round(i * (len(scored) - 1) / max(n - 1, 1)) for i in range(n)]
            work = [scored[i][1] for i in sorted(set(idx))]
            print(f'Sample:  {len(work)} wells spanning peak biomass '
                  f'{scored[0][0]:.5f} .. {scored[-1][0]:.5f}')
    elif args.limit:
        work = work[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    print(f'Root:    {args.root}', flush=True)
    print(f'Out:     {args.out}', flush=True)
    print(f'Wells:   {len(work)} across {len(procDirs)} plate(s)', flush=True)
    print(f'Frame:   {args.frame} | window [{lo}, {hi}] | {args.size}px | '
          f'invert={args.invert}', flush=True)
    if args.throttle_mbps:
        print(f'Throttle: {args.throttle_mbps} MB/s')
    if args.yield_to_gpu:
        print('Yield:   pauses while a torch DataLoader job is running')
    if args.dry_run:
        for row in work[:5]:
            print(f'  would render {row[0]}/{row[2]}')
        print(f'  ... ({len(work)} total) — dry run, nothing read')
        return 0

    import cv2
    import tifffile

    os.makedirs(args.out, exist_ok=True)
    manifestPath = os.path.join(args.out, 'thumbnails.csv')
    # Resume: an existing manifest is rewritten, but wells whose JPEG is already
    # on disk are not re-read. Re-running after an interruption is therefore cheap.
    rows, done, failed, readBytes = [], 0, 0, 0
    t0 = time.time()

    lock = __import__('threading').Lock()
    counter = {'n': 0}

    def _one(item):
        drawerID, plateID, wellId, mag, stack = item
        outDir = os.path.join(args.out, drawerID)
        os.makedirs(outDir, exist_ok=True)
        jpg = os.path.join(outDir, f'{wellId}.jpg')
        rel = os.path.relpath(jpg, args.out)
        if os.path.exists(jpg):
            return (drawerID, plateID, wellId, mag, '', '', rel), 0, None

        if args.yield_to_gpu:
            while _isGpuJobRunning():
                time.sleep(10)

        biomassPath = os.path.join(os.path.dirname(stack), f'{wellId}_biomass.csv')
        with tifffile.TiffFile(stack) as tf:
            series = tf.series[0]
            nFrames = series.shape[0]
            if args.frame == 'peak':
                idx, peakVal = _peakFrame(biomassPath, nFrames)
            else:
                idx, peakVal = int(args.frame), float('nan')
            idx = max(0, min(idx, nFrames - 1))
            page = series.asarray(key=idx)

        thumb = renderThumb(page, args.size, (lo, hi), args.invert)
        if not cv2.imwrite(jpg, thumb, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]):
            raise IOError('cv2.imwrite returned False')
        return ((drawerID, plateID, wellId, mag, idx,
                 '' if peakVal != peakVal else f'{peakVal:.6g}', rel),
                page.nbytes, None)

    def _record(res, item):
        row, nbytes, _ = res
        with lock:
            rows.append(row)
            counter['n'] += 1
            i = counter['n']
        return nbytes, i

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Threads, not processes: the heavy steps (TIFF read, cv2 resize, JPEG
        # encode) all release the GIL, and threads avoid re-importing tifffile/cv2
        # per worker.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_one, it): it for it in work}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    failed += 1
                    it = futs[fut]
                    print(f'  ERROR {it[0]}/{it[2]}: {e}', file=sys.stderr)
                    continue
                nbytes, i = _record(res, futs[fut])
                readBytes += nbytes
                if nbytes:
                    done += 1
                if i % 200 == 0 or i == len(work):
                    el = time.time() - t0
                    print(f'  {i}/{len(work)} wells · {readBytes/2**30:.1f} GiB · '
                          f'{readBytes/max(el,1e-9)/1e6:.0f} MB/s · {el/60:.1f} min',
                          flush=True)
    else:
        for i, item in enumerate(work, 1):
            try:
                res = _one(item)
            except Exception as e:
                failed += 1
                print(f'  ERROR {item[0]}/{item[2]}: {e}', file=sys.stderr)
                continue
            rows.append(res[0])
            readBytes += res[1]
            if res[1]:
                done += 1
            if args.throttle_mbps:
                target = readBytes / (args.throttle_mbps * 1e6)
                drift = target - (time.time() - t0)
                if drift > 0:
                    time.sleep(drift)
            if i % 100 == 0 or i == len(work):
                el = time.time() - t0
                print(f'  {i}/{len(work)} wells · {readBytes/2**30:.1f} GiB · '
                      f'{readBytes/max(el,1e-9)/1e6:.0f} MB/s · {el/60:.1f} min',
                      flush=True)

    with open(manifestPath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['drawerID', 'plateID', 'wellID', 'mag', 'frame',
                    'biomassAtFrame', 'thumb'])
        w.writerows(rows)

    el = time.time() - t0
    print(f'\nWrote {done} thumbnails ({failed} failed), manifest: {manifestPath}')
    print(f'Read {readBytes/2**30:.1f} GiB in {el/60:.1f} min')
    print(f'Window [{lo}, {hi}] invert={args.invert} — fixed for every well, so '
          f'thumbnails are comparable across plates.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
