#!/usr/bin/env python3

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import argparse

import numpy as np
import pandas as pd

from multiWellAnalysis.colony.io_utils import (
    loadRawStack,
    ensureDir,
    logPlate,
    logWell,
    checkpointExists,
    writeCheckpoint,
    timestamp
)

from multiWellAnalysis.colony.colonyFeatsMicrons import (
    extractColonyGeometry,
    addColonySpatialFeatures,
    addColonyNeighborFeatures,
    addColonyGraphFeatures,
    addColonyIntensityMassFeatures,
    extractBackgroundIntensityFeatures
)

from multiWellAnalysis.colony.wellAggMicrons import aggregateWellFeatures


backgroundDilateRadius = 5


def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def logWellAndPlate(plateId, wellId, msg, logSuffix):
    logWell(plateId, wellId, msg, suffix=logSuffix)
    logPlate(plateId, f'{wellId}: {msg}', suffix=logSuffix)


def extractTrackedColonyFeatures(
    rawStack,
    labelStack,
    frames,
    plateId,
    wellId,
    wasTracked,
    trackedLabelsPath,
    rawPath,
    pxToUm,
):
    rows = []
    nonEmpty = np.any(labelStack, axis=(0, 1))

    meta = {
        'plateID': plateId,
        'wellID': wellId,
        'trackedLabelsPath': trackedLabelsPath,
        'registeredRawPath': rawPath,
        'wasTracked': wasTracked,
    }

    for i in np.nonzero(nonEmpty)[0]:
        t = frames[i]
        labels = labelStack[:, :, i]
        rawImg = rawStack[:, :, t]

        colonyDf = extractColonyGeometry(labels, rawImg, pxToUm)
        if colonyDf.empty:
            continue

        for k, v in meta.items():
            colonyDf[k] = v

        colonyDf = addColonySpatialFeatures(colonyDf, pxToUm)
        colonyDf = addColonyNeighborFeatures(colonyDf, pxToUm)
        colonyDf = addColonyGraphFeatures(colonyDf, pxToUm)
        colonyDf = addColonyIntensityMassFeatures(colonyDf, labels, rawImg, pxToUm)

        bg = extractBackgroundIntensityFeatures(
            rawImg,
            labels,
            dilateRadius=backgroundDilateRadius
        )

        for k, v in bg.items():
            colonyDf[k] = v

        colonyDf['frame'] = int(t)
        rows.append(colonyDf)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, copy=False)


def processOneWell(argsTuple):
    row, outRoot, checkpointTag, logSuffix, featVersion = argsTuple

    plateId = row['plateId']
    wellId = row['wellId']
    trackedPath = row['trackedLabelsPath']
    rawPath = row['rawPath']
    pxToUmRaw = row.get('pxToUm')

    if checkpointExists(plateId, wellId, checkpointTag):
        logWellAndPlate(plateId, wellId, 'skipped (checkpoint exists)', logSuffix)
        return

    try:
        logWellAndPlate(plateId, wellId, 'status=in_progress', logSuffix)

        if pxToUmRaw in (None, '') or (isinstance(pxToUmRaw, float) and pxToUmRaw != pxToUmRaw):
            raise RuntimeError(f'missing pxToUm in index row for {wellId}')
        pxToUm = float(pxToUmRaw)

        if not isinstance(trackedPath, str) or not os.path.exists(trackedPath):
            raise RuntimeError(f'missing tracked labels: {trackedPath}')

        if not os.path.exists(rawPath):
            raise RuntimeError(f'missing raw stack: {rawPath}')

        rawStack = loadRawStack(rawPath)

        with np.load(trackedPath, allow_pickle=False) as npz:
            labelStack = npz['labels']
            frames = npz['frames']
            wasTracked = bool(npz['wasTracked']) if 'wasTracked' in npz else True

        colonyDf = extractTrackedColonyFeatures(
            rawStack,
            labelStack,
            frames,
            plateId,
            wellId,
            wasTracked,
            trackedPath,
            rawPath,
            pxToUm,
        )

        outdir = f'{outRoot}/{plateId}'
        ensureDir(outdir)

        colonyOutCsv = f'{outdir}/{wellId}_perColonyFeatures.csv'
        colonyDf.to_csv(colonyOutCsv, index=False)

        wellDf = aggregateWellFeatures(colonyDf, frames, plateId, wellId)
        wellOutCsv = f'{outdir}/{wellId}_wellColonyFeatures.csv'
        wellDf.to_csv(wellOutCsv, index=False)

        writeCheckpoint(
            plateId,
            wellId,
            checkpointTag,
            {
                'nColonyRows': int(len(colonyDf)),
                'nWellRows': int(len(wellDf)),
                'wasTracked': wasTracked,
                'timestamp': timestamp()
            }
        )

        logWellAndPlate(
            plateId,
            wellId,
            f'status=done nRows={len(colonyDf)}',
            logSuffix
        )

    except Exception:
        tb = traceback.format_exc()
        logWell(plateId, wellId, 'status=error', suffix=logSuffix)
        logWell(plateId, wellId, tb.replace('\n', ' | '), suffix=logSuffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--outRoot', required=True)
    parser.add_argument('--featVersion', default='colFeats_microns_v1')
    parser.add_argument('--nProc', type=int, default=16)

    args = parser.parse_args()
    

    # Make logs + checkpoints follow outRoot
    os.environ['MWA_LOG_ROOT'] = args.outRoot

    indexDf = pd.read_csv(args.index)

    required = {
        'plateId',
        'wellId',
        'rawPath',
        'trackedLabelsPath'
    }

    missing = required - set(indexDf.columns)
    if missing:
        raise ValueError(f'missing required columns: {missing}')

    checkpointTag = f'colonyFeats_{args.featVersion}'
    logSuffix = f'_{args.featVersion}'

    for plateId, plateDf in indexDf.groupby('plateId'):
        print(f'[{ts()}] START plate {plateId}', flush=True)
        logPlate(plateId, '==== START colony feature extraction ====', suffix=logSuffix)

        rows = plateDf.to_dict(orient='records')

        taskArgs = [
            (r, args.outRoot, checkpointTag, logSuffix, args.featVersion)
            for r in rows
        ]

        with ProcessPoolExecutor(max_workers=args.nProc) as ex:
            list(ex.map(processOneWell, taskArgs, chunksize=4))

        logPlate(plateId, '==== FINISHED colony feature extraction ====', suffix=logSuffix)
        print(f'[{ts()}] FINISHED plate {plateId}', flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()