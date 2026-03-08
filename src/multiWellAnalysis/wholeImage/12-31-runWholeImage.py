'''
LEGACY SCRIPT.

This file is intended to be run directly as a script and is NOT part of the
multiWellAnalysis Python package API.

It relies on local imports and execution context.
'''


#!/usr/bin/env python3
import os
import time
import warnings
warnings.filterwarnings("ignore")

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import tifffile

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from multiWellAnalysis.src.multiWellAnalysis.wholeImage.extractWholeImageFeats import extract_frame_features


# shared plate-level logging helpers

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def plate_log_path(outdir, plateID):
    return os.path.join(outdir, f'{plateID}.log')


def plate_log(outdir, plateID, msg):
    with open(plate_log_path(outdir, plateID), 'a') as f:
        f.write(f'[{now()}] {msg}\n')


# Well-level runner

def processWellWholeImage(
    plateID,
    wellID,
    processedPath,
    outdir,
    startFrame=6,
    featureVersion='mahotas_v1'
):
    timings = {}

    def tic(k):
        timings[k] = -time.perf_counter()

    def toc(k):
        timings[k] += time.perf_counter()

    plateOutdir = os.path.join(outdir, plateID)
    os.makedirs(plateOutdir, exist_ok=True)

    checkpointDir = os.path.join(outdir, 'checkpoints', plateID)
    os.makedirs(checkpointDir, exist_ok=True)

    paramHash = f'wholeImage_sf{startFrame}_{featureVersion}'

    outCSV = os.path.join(plateOutdir, f'{wellID}_wholeImage.csv')
    logPath = os.path.join(plateOutdir, f'{wellID}.wholeImage.log')
    checkpointPath = os.path.join(checkpointDir, f'{wellID}_{paramHash}.done')

    def log(msg):
        with open(logPath, 'a') as f:
            f.write(f'[{now()}] {msg}\n')

    log('status=in_progress')
    log(f'start well={wellID}')
    log(f'processed_path={processedPath}')
    log(f'params startFrame={startFrame} featureVersion={featureVersion}')

    if os.path.exists(checkpointPath) and os.path.exists(outCSV):
        log('checkpoint exists, skipping')
        return wellID, 'skipped'

    try:
        tic('load_stack')
        # stack = tifffile.imread(processedPath)
        stack = tifffile.memmap(processedPath)
        log(f'stack_shape={stack.shape}')

        toc('load_stack')
        if stack.ndim == 2:
            # single frame
            stack = stack[np.newaxis, :, :]

        elif stack.ndim == 3:
            # Ensure stack is (T, H, W)
            # Identify time axis as the one with the smallest dimension
            # (31 frames vs ~2000 px)
            if stack.shape[0] <= 64:
                pass  # already (T, H, W)
            elif stack.shape[2] <= 64:
                stack = np.moveaxis(stack, 2, 0)  # (H, W, T) → (T, H, W)
            elif stack.shape[1] <= 64:
                stack = np.moveaxis(stack, 1, 0)  # (H, T, W) → (T, H, W)
            else:
                raise ValueError(f'cannot infer time axis for stack {stack.shape}')

        else:
            raise ValueError(f'invalid stack shape {stack.shape}')

        tic('feature_extraction')
        rows = []
        for t in range(startFrame, stack.shape[0]):
            feats = extract_frame_features(stack[t])
            feats.update({
                'plateID': plateID,
                'wellID': wellID,
                'frame': t,
                'processed_path': processedPath
            })
            rows.append(feats)
        toc('feature_extraction')
        
        if not rows:
            raise RuntimeError('no frames processed (rows empty)')
        
        tic('write_csv')
        tmpCSV = outCSV + '.tmp'
        pd.DataFrame(rows).to_csv(tmpCSV, index=False)
        os.replace(tmpCSV, outCSV)
        toc('write_csv')

        elapsed = sum(v for v in timings.values() if v > 0)

        tmpChk = checkpointPath + '.tmp'
        with open(tmpChk, 'w') as f:
            f.write(f'well={wellID}\n')
            f.write(f'elapsedSeconds={elapsed:.3f}\n')
            f.write(f'finishedAt={now()}\n')
            f.write(f'startFrame={startFrame}\n')
            f.write(f'featureVersion={featureVersion}\n')
        os.replace(tmpChk, checkpointPath)

        for k, v in timings.items():
            log(f'timing.{k} = {v:.3f}s')

        log('status=done')
        return wellID, 'done'
    
    except Exception as e:
        log(f'error {e}')

        if rows:
            tmpCSV = outCSV + '.partial'
            pd.DataFrame(rows).to_csv(tmpCSV, index=False)
            log(f'wrote partial CSV with {len(rows)} frames')

        for k, v in timings.items():
            if v > 0:
                log(f'timing.{k} = {v:.3f}s (partial)')
        return wellID, 'error'


# plate-level runner

def runPlateWholeImage(indexDF, plateID, outdir, maxWorkers):
    rows = indexDF[indexDF['plateID'] == plateID]
    results = []

    with ProcessPoolExecutor(max_workers=maxWorkers) as pool:
        futures = {
            pool.submit(
                processWellWholeImage,
                plateID,
                row['wellID'],
                row['processed_path'],
                outdir
            ): row['wellID']
            for _, row in rows.iterrows()
        }

        for fut in as_completed(futures):
            wellID = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                plate_log(outdir, plateID,
                          f'WELL {wellID} FAILED with exception: {e}')
                results.append((wellID, 'error'))
                continue

            wellID, status = res
            plate_log(outdir, plateID,
                      f'WELL {wellID} status={status}')
            results.append(res)

    return results




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()

    idx = pd.read_csv(args.index)
    idx = idx[pd.to_numeric(idx['nFrames'], errors='coerce').notna()]

    os.makedirs(args.outdir, exist_ok=True)

    allResults = []

    for plateID in sorted(idx['plateID'].unique()):
        plate_log(args.outdir, plateID,
                  '=== START whole-image feature extraction ===')
        plate_log(args.outdir, plateID,
                  f'params: startFrame=6 featureVersion=mahotas_v1')

        print(f'Running whole-image features for {plateID}')
        t0 = time.perf_counter()

        plateResults = runPlateWholeImage(
            idx,
            plateID,
            args.outdir,
            args.workers
        )

        elapsed = time.perf_counter() - t0

        plate_log(args.outdir, plateID,
                  f'finished whole-image feature extraction in {elapsed/60:.2f} minutes')

        for wellID, status in plateResults:
            allResults.append({
                'plateID': plateID,
                'wellID': wellID,
                'status': status
            })

    pd.DataFrame(allResults).to_csv(
        os.path.join(args.outdir, 'run_summary.csv'),
        index=False
    )
