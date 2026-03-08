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

from multiWellAnalysis.colony.wellAgg import aggregateWellFeatures

# config

# INDEX_CSV = '/mnt/data/trainingData/processed_index.csv'
INDEX_CSV = '/mnt/data/reimaging/index/reimaging_processed_index_withTracked.csv'
OUT_ROOT = '/mnt/data/reimaging/processed'
# OUT_ROOT = '/mnt/data/trainingData'

featVersion = 'colFeats_microns_v1'
checkpointTag = f'colonyFeats_{featVersion}'
logSuffix = f'_{featVersion}'

nProc = 16
backgroundDilateRadius = 5


# utils

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def logWellAndPlate(plateId, wellId, msg):
    logWell(plateId, wellId, msg, suffix=logSuffix)
    logPlate(plateId, f'{wellId}: {msg}', suffix=logSuffix)


# core extraction

def extractTrackedColonyFeatures(
    rawStack,
    labelStack,
    frames,
    plateId,
    wellId,
    wasTracked,
    trackedLabelsPath,
    registeredRawPath
):
    rows = []

    nonEmpty = np.any(labelStack, axis=(0, 1))

    meta = {
        'plateID': plateId,
        'wellID': wellId,
        'trackedLabelsPath': trackedLabelsPath,
        'registeredRawPath': registeredRawPath,
        'wasTracked': wasTracked,
    }
    
    for i in np.nonzero(nonEmpty)[0]:
        t = frames[i]
        labels = labelStack[:, :, i]
        rawImg = rawStack[:, :, t]

        colonyDf = extractColonyGeometry(labels, rawImg)
        if colonyDf.empty:
            continue
        
        for k, v in meta.items():
            colonyDf[k] = v

        colonyDf = addColonySpatialFeatures(colonyDf)
        colonyDf = addColonyNeighborFeatures(colonyDf)
        colonyDf = addColonyGraphFeatures(colonyDf)
        colonyDf = addColonyIntensityMassFeatures(colonyDf, labels, rawImg)

        bg = extractBackgroundIntensityFeatures(rawImg, labels, dilateRadius=backgroundDilateRadius)
        for k, v in bg.items():
            colonyDf[k] = v
        
        colonyDf['frame'] = int(t)

        rows.append(colonyDf)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, copy=False)

# worker

def processOneWell(row):
    plateId = row['plateId']
    wellId = row['wellId']

    if checkpointExists(plateId, wellId, checkpointTag):
        logWellAndPlate(plateId, wellId, 'skipped (checkpoint exists)')
        return

    try:
        logWellAndPlate(plateId, wellId, 'status=in_progress')

        trackedPath = row['trackedLabelsPath']
        if not isinstance(trackedPath, str) or not os.path.exists(trackedPath):
            raise RuntimeError('missing tracked_labels_path')
        
        rawStack = loadRawStack(row['rawPath'])
            
        with np.load(trackedPath, allow_pickle=False) as npz:
            labelStack = npz['labels']
            frames = npz['frames']
            nFrames = len(frames)
            wasTracked = bool(npz['wasTracked']) if 'wasTracked' in npz else True

        colonyDf = extractTrackedColonyFeatures(
            rawStack,
            labelStack,
            frames,
            plateId,
            wellId,
            wasTracked,
            trackedPath,
            row['rawPath']
        )
        
            
        if colonyDf.empty:
            logWellAndPlate(
                plateId,
                wellId,
                'status=no_colonies_detected'
            )

        outdir = f'{OUT_ROOT}/{plateId}'
        ensureDir(outdir)

        colonyOutCsv = f'{outdir}/{wellId}_colonyFeatures_{featVersion}.csv'
        colonyDf.to_csv(colonyOutCsv, index=False)

        wellDf = aggregateWellFeatures(colonyDf, frames, plateId, wellId)
        wellOutCsv = f'{outdir}/{wellId}_wellColonyFeatures_{featVersion}.csv'
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



# main

def main():
    indexDf = pd.read_csv(INDEX_CSV)

    required = {
        'plateId',
        'wellId',
        'rawPath',
        'trackedLabelsPath'
    }

    missing = required - set(indexDf.columns)
    if missing:
        raise ValueError(f'missing required columns: {missing}')

    for plateId, plateDf in indexDf.groupby('plateId'):
        print(f'[{ts()}] START plate {plateId}', flush=True)
        logPlate(
            plateId,
            '==== START colony feature extraction (tracked) ====',
            suffix=logSuffix
        )


        rows = plateDf.to_dict(orient='records')

        with ProcessPoolExecutor(max_workers=nProc) as ex:
            list(ex.map(processOneWell, rows, chunksize=4))

        logPlate(
            plateId,
            '==== FINISHED colony feature extraction ====',
            suffix=logSuffix
        )
        print(f'[{ts()}] FINISHED plate {plateId}', flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
