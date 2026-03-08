#!/usr/bin/env python3
import os
import time
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import tifffile

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from .extractWholeImageFeats import extractFrameFeats


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def plateLogPath(outDir, plateId):
    return os.path.join(outDir, f'{plateId}.log')


def plateLog(outDir, plateId, msg):
    with open(plateLogPath(outDir, plateId), 'a') as f:
        f.write(f'[{now()}] {msg}\n')


def processWellWholeImage(
    plateId,
    wellId,
    rawPath,
    processedPath,
    outDir,
    startFrame=0,
    featureVersion='mahotas_v2'
):

    timings = {}
    rows = []

    def tic(k):
        timings[k] = -time.perf_counter()

    def toc(k):
        timings[k] += time.perf_counter()

    plateOutDir = os.path.join(outDir, plateId)
    os.makedirs(plateOutDir, exist_ok=True)

    checkpointDir = os.path.join(outDir, 'checkpoints', plateId)
    os.makedirs(checkpointDir, exist_ok=True)

    paramHash = f'wholeImage_procOnly_sf{startFrame}_{featureVersion}'

    outCsv = os.path.join(
        plateOutDir,
        f'{wellId}_wholeImage_{featureVersion}.csv'
    )

    logPath = os.path.join(
        plateOutDir,
        f'{wellId}.wholeImage_{featureVersion}.log'
    )

    checkpointPath = os.path.join(
        checkpointDir,
        f'{wellId}_{paramHash}_{featureVersion}.done'
    )

    def log(msg):
        with open(logPath, 'a') as f:
            f.write(f'[{now()}] {msg}\n')

    log('status=in_progress')
    log(f'start well={wellId}')
    log(f'processedPath={processedPath}')
    log(f'params startFrame={startFrame} featureVersion={featureVersion}')

    if os.path.exists(checkpointPath) and os.path.exists(outCsv):
        log('checkpoint exists, skipping')
        return wellId, 'skipped'

    try:
        tic('loadStack')

        try:
            procStack = tifffile.memmap(processedPath)
        except Exception:
            procStack = tifffile.imread(processedPath)

        log(f'procStackShape={procStack.shape}')

        toc('loadStack')

        def ensureThw(stack):
            if stack.ndim == 2:
                stack = stack[np.newaxis, :, :]
            elif stack.ndim == 3:
                if stack.shape[0] <= 64:
                    pass
                elif stack.shape[2] <= 64:
                    stack = np.moveaxis(stack, 2, 0)
                elif stack.shape[1] <= 64:
                    stack = np.moveaxis(stack, 1, 0)
                else:
                    raise ValueError(f'cannot infer time axis for stack {stack.shape}')
            else:
                raise ValueError(f'invalid stack shape {stack.shape}')
            return stack

        procStack = ensureThw(procStack)

        tic('featureExtraction')

        for t in range(startFrame, procStack.shape[0]):

            procFeats = extractFrameFeats(procStack[t])
            procFeats = {f'proc_{k}': v for k, v in procFeats.items()}

            procFeats.update({
                'plateId': plateId,
                'wellId': wellId,
                'frame': t,
                'processedPath': processedPath
            })

            rows.append(procFeats)

        toc('featureExtraction')

        if not rows:
            raise RuntimeError('no frames processed')

        tic('writeCsv')
        tmpCsv = outCsv + '.tmp'
        pd.DataFrame(rows).to_csv(tmpCsv, index=False)
        os.replace(tmpCsv, outCsv)
        toc('writeCsv')

        elapsed = sum(v for v in timings.values() if v > 0)

        tmpChk = checkpointPath + '.tmp'
        with open(tmpChk, 'w') as f:
            f.write(f'wellId={wellId}\n')
            f.write(f'elapsedSeconds={elapsed:.3f}\n')
            f.write(f'finishedAt={now()}\n')
            f.write(f'startFrame={startFrame}\n')
            f.write(f'featureVersion={featureVersion}\n')
        os.replace(tmpChk, checkpointPath)

        for k, v in timings.items():
            log(f'timing.{k}={v:.3f}s')

        log('status=done')
        return wellId, 'done'

    except Exception as e:

        log(f'error {e}')

        if rows:
            tmpCsv = outCsv + '.partial'
            pd.DataFrame(rows).to_csv(tmpCsv, index=False)
            log(f'wrote partial CSV with {len(rows)} frames')

        for k, v in timings.items():
            if v > 0:
                log(f'timing.{k}={v:.3f}s (partial)')

        return wellId, 'error'

def runPlateWholeImage(indexDf, plateId, outDir, maxWorkers):

    plateRows = indexDf[indexDf['plateId'] == plateId]
    results = []

    with ProcessPoolExecutor(max_workers=maxWorkers) as pool:

        futures = {
            pool.submit(
                processWellWholeImage,
                plateId,
                row['wellId'],
                row['rawPath'],
                row['processedPath'],
                outDir
            ): row['wellId']
            for _, row in plateRows.iterrows()
        }

        for fut in as_completed(futures):
            wellId = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                plateLog(outDir, plateId, f'WELL {wellId} FAILED: {e}')
                results.append((wellId, 'error'))
                continue

            wellId, status = res
            plateLog(outDir, plateId, f'WELL {wellId} status={status}')
            results.append(res)

    return results


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    indexDf = pd.read_csv(args.index)

    indexDf = indexDf[
        indexDf['processedPath'].notna()
    ]

    if 'nFrames' in indexDf.columns:
        indexDf = indexDf[
            pd.to_numeric(indexDf['nFrames'], errors='coerce').notna()
        ]
    
    
    os.makedirs(args.outdir, exist_ok=True)

    allResults = []

    for plateId in sorted(indexDf['plateId'].unique()):

        plateLog(args.outdir, plateId, 'start whole-image feature extraction')
        plateLog(args.outdir, plateId, 'params startFrame=0 featureVersion=mahotas_v2')

        print(f'Running whole-image features for {plateId}')
        t0 = time.perf_counter()

        plateResults = runPlateWholeImage(
            indexDf,
            plateId,
            args.outdir,
            args.workers
        )

        elapsed = time.perf_counter() - t0

        plateLog(
            args.outdir,
            plateId,
            f'finished in {elapsed/60:.2f} minutes'
        )

        for wellId, status in plateResults:
            allResults.append({
                'plateId': plateId,
                'wellId': wellId,
                'status': status
            })

    pd.DataFrame(allResults).to_csv(
        os.path.join(args.outdir, 'run_summary.csv'),
        index=False
    )