#!/usr/bin/env python3

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from multiWellAnalysis.intensity.io_utils import (
    loadRawStack,
    loadProcessedStack,
    ensureDir,
    logPlate,
    logWell,
    checkpointExists,
    writeCheckpoint,
    timestamp
)

from multiWellAnalysis.intensity.intensityFeats import (
    addColonyIntensityFeatures,
    extractBackgroundIntensityFeatures
)

from multiWellAnalysis.colony.colonyFeats import extractColonyGeometry
from multiWellAnalysis.colony.wellAgg import aggregateWellFeatures



INDEX_CSV = '/mnt/data/trainingData/processed_index.csv'
OUT_ROOT = '/mnt/data/trainingData'

featVersion = 'colonyIntensity_v1'
checkpointTag = f'colonyFeats_{featVersion}'
logSuffix = f'_{featVersion}'

nProc = 16
backgroundDilateRadius = 5

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def logWellAndPlate(plateId, wellId, msg):
    logWell(plateId, wellId, msg, suffix=logSuffix)
    logPlate(plateId, f'{wellId}: {msg}', suffix=logSuffix)


def extractTrackedColonyFeatures(
    rawStack,
    procStack,
    labelStack,
    frames,
    plateId,
    wellId,
    wasTracked,
    trackedLabelsPath,
    registeredRawPath,
    registeredProcessedPath
):
    rows = []

    nonEmpty = np.any(labelStack, axis=(0, 1))

    meta = {
        'plateID': plateId,
        'wellID': wellId,
        'trackedLabelsPath': trackedLabelsPath,
        'registeredRawPath': registeredRawPath,
        'registeredProcessedPath': registeredProcessedPath,
        'wasTracked': wasTracked,
    }

    for i in np.nonzero(nonEmpty)[0]:
        t = int(frames[i])
        labels = labelStack[:, :, i]

        rawImg = rawStack[:, :, t]
        procImg = procStack[:, :, t]

        colonyDf = extractColonyGeometry(labels)
        if colonyDf.empty:
            continue

        for k, v in meta.items():
            colonyDf[k] = v

        colonyDf = addColonyIntensityFeatures(
            colonyDf, labels, rawImg, prefix='rawCol'
        )

        colonyDf = addColonyIntensityFeatures(
            colonyDf, labels, procImg, prefix='procCol'
        )

        bgRaw = extractBackgroundIntensityFeatures(
            rawImg, labels,
            dilateRadius=backgroundDilateRadius,
            prefix='bgRaw'
        )

        bgProc = extractBackgroundIntensityFeatures(
            procImg, labels,
            dilateRadius=backgroundDilateRadius,
            prefix='bgProc'
        )

        for k, v in {**bgRaw, **bgProc}.items():
            colonyDf[k] = v

        colonyDf['frame'] = t
        rows.append(colonyDf)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, copy=False)


def processOneWell(row):
    plateId = row['plateID']
    wellId = row['wellID']

    if checkpointExists(plateId, wellId, checkpointTag):
        logWellAndPlate(plateId, wellId, 'skipped (checkpoint exists)')
        return

    try:
        logWellAndPlate(plateId, wellId, 'status=in_progress')

        trackedPath = row['trackedLabelsPath']
        rawPath = row['registered_raw_path']
        procPath = row['registered_processed_path']

        if not os.path.exists(trackedPath):
            raise RuntimeError('missing trackedLabelsPath')

        rawStack = loadRawStack(rawPath)
        procStack = loadProcessedStack(procPath)

        with np.load(trackedPath, allow_pickle=False) as npz:
            labelStack = npz['labels']
            frames = npz['frames']
            wasTracked = bool(npz['wasTracked']) if 'wasTracked' in npz else True

        colonyDf = extractTrackedColonyFeatures(
            rawStack,
            procStack,
            labelStack,
            frames,
            plateId,
            wellId,
            wasTracked,
            trackedPath,
            rawPath,
            procPath
        )

        if colonyDf.empty:
            logWellAndPlate(plateId, wellId, 'status=no_colonies_detected')
            return

        outdir = f'{OUT_ROOT}/{plateId}'
        ensureDir(outdir)

        colonyOutCsv = f'{outdir}/{wellId}_colonyIntensity_{featVersion}.csv'
        colonyDf.to_csv(colonyOutCsv, index=False)

        wellDf = aggregateWellFeatures(colonyDf, frames, plateId, wellId)
        wellOutCsv = f'{outdir}/{wellId}_wellIntensity_{featVersion}.csv'
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
            f'status=done nRows={len(colonyDf)}'
        )

    except Exception:
        tb = traceback.format_exc()
        logWell(plateId, wellId, 'status=error', suffix=logSuffix)
        logWell(plateId, wellId, tb.replace('\n', ' | '), suffix=logSuffix)


def main():
    indexDf = pd.read_csv(INDEX_CSV)

    required = {
        'plateID',
        'wellID',
        'registered_raw_path',
        'registered_processed_path',
        'trackedLabelsPath',
    }

    missing = required - set(indexDf.columns)
    if missing:
        raise ValueError(f'missing required columns: {missing}')

    for plateId, plateDf in indexDf.groupby('plateID'):
        print(f'[{ts()}] START plate {plateId}', flush=True)

        logPlate(
            plateId,
            '==== START intensity feature extraction ====',
            suffix=logSuffix
        )

        rows = plateDf.to_dict(orient='records')

        with ProcessPoolExecutor(max_workers=nProc) as ex:
            list(ex.map(processOneWell, rows, chunksize=4))

        logPlate(
            plateId,
            '==== FINISHED intensity feature extraction ====',
            suffix=logSuffix
        )

        print(f'[{ts()}] FINISHED plate {plateId}', flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
