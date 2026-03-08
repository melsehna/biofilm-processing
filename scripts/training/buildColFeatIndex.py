#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

indexCsv = '/mnt/data/trainingData/processed_index.csv'
outCsv = '/mnt/data/trainingData/processed_index.csv'
dataRoot = '/mnt/data/trainingData'

trackingFeatVersion = 'trackingVec_v2'

colonyFeatVersion = 'colFeats_' + trackingFeatVersion

indexDf = pd.read_csv(indexCsv)

requiredCols = {
    'plateID',
    'wellID',
    'registered_raw_path',
    'mask_path',
    'nFrames'
}

missingCols = requiredCols - set(indexDf.columns)
if missingCols:
    raise ValueError(f'missing required columns: {missingCols}')


def loadBiomassTimeseries(plateId, wellId):
    csvPath = os.path.join(dataRoot, plateId, f'{wellId}_timeseries.csv')
    if not os.path.exists(csvPath):
        return None
    return pd.read_csv(csvPath)


def trackedLabelsPath(plateId, wellId):
    return os.path.join(
        dataRoot,
        plateId,
        'processedImages',
        f'{wellId}_trackedLabels_allFrames_{trackingFeatVersion}.npz'
    )


def colonyFeaturesPath(plateId, wellId):
    return os.path.join(
        dataRoot,
        plateId,
        f'{wellId}_colonyFeatures_{colonyFeatVersion}.csv'
    )


def wellFeaturesPath(plateId, wellId):
    return os.path.join(
        dataRoot,
        plateId,
        f'{wellId}_wellColonyFeatures_{colonyFeatVersion}.csv'
    )


def inspectTracking(npzPath):
    if not os.path.exists(npzPath):
        return False, False, [], 0

    try:
        npz = np.load(npzPath)
    except Exception:
        return True, False, [], 0

    keys = set(npz.files)

    if {'labels', 'frames'} <= keys:
        frames = npz['frames'].astype(int).tolist()
        return True, True, frames, len(frames)

    frameKeys = [k for k in keys if k.startswith('frame_')]
    if frameKeys:
        frames = sorted(int(k.split('_')[1]) for k in frameKeys)
        return True, True, frames, len(frames)

    return True, False, [], 0

def inspectTrackingMeta(npzPath):
    if not os.path.exists(npzPath):
        return {}
    try:
        npz = np.load(npzPath)
    except Exception:
        return {}
    out = {}
    for k in ['seedFrame', 'peakFrame', 'wasTracked']:
        if k in npz:
            out[k] = int(npz[k]) if npz[k].dtype != bool else bool(npz[k])
    return out


rows = []

for _, row in indexDf.iterrows():
    plateId = row['plateID']
    wellId = row['wellID']
    nFrames = int(row['nFrames'])

    ts = loadBiomassTimeseries(plateId, wellId)

    if ts is None or 'biomass' not in ts.columns:
        seedFrame = np.nan
        seedStatus = 'missing_biomass_csv'
        peakFrame = np.nan
        peakBiomass = np.nan
        peakStatus = 'missing_biomass_csv'
    else:
        ts = ts.dropna(subset=['biomass'])

        if ts.empty:
            seedFrame = np.nan
            seedStatus = 'empty_biomass'
            peakFrame = np.nan
            peakBiomass = np.nan
            peakStatus = 'empty_biomass'
        else:
            seedFrame = np.nan
            seedStatus = 'no_seed_detected'

            for i in range(len(ts) - 1):
                if ts['biomass'].iloc[i] >= 0.005 and ts['biomass'].iloc[i + 1] >= 0.005:
                    seedFrame = int(ts['frame'].iloc[i])
                    seedStatus = 'ok'
                    break

            idx = ts['biomass'].idxmax()
            peakFrame = int(ts.loc[idx, 'frame'])
            peakBiomass = float(ts.loc[idx, 'biomass'])

            if peakFrame < 0 or peakFrame >= nFrames:
                peakFrame = np.nan
                peakBiomass = np.nan
                peakStatus = 'peak_frame_out_of_bounds'
            else:
                peakStatus = 'ok'

    segPath = trackedLabelsPath(plateId, wellId)
    hasTrackedLabels, wasTracked, frames, nTrackedFrames = inspectTracking(segPath)

    colFeatPath = colonyFeaturesPath(plateId, wellId)
    wellFeatPath = wellFeaturesPath(plateId, wellId)

    outRow = dict(row)
    outRow.update({
        'seedFrame': seedFrame,
        'seedStatus': seedStatus,
        'peakFrame': peakFrame,
        'peakBiomass': peakBiomass,
        'peakStatus': peakStatus,
        'trackedLabelsPath': segPath,
        'hasTrackedLabels': hasTrackedLabels,
        'wasTracked': wasTracked,
        'trackedFrames': ','.join(map(str, frames)),
        'nTrackedFrames': nTrackedFrames,
        'colonyFeaturesPath': colFeatPath,
        'hasColonyFeatures': os.path.exists(colFeatPath),
        'wellFeaturesPath': wellFeatPath,
        'hasWellFeatures': os.path.exists(wellFeatPath),
        'trackingVersion': trackingFeatVersion,
        'colonyFeatVersion': colonyFeatVersion,
    })
    trackingMeta = inspectTrackingMeta(segPath)
    outRow.update(trackingMeta)

    rows.append(outRow)


outDf = pd.DataFrame(rows)
outDf.to_csv(outCsv, index=False)

print(f'wrote {len(outDf)} rows to {outCsv}')
print('\nSeed status:')
print(outDf['seedStatus'].value_counts())
print('\nPeak status:')
print(outDf['peakStatus'].value_counts())
print('\nTracking:')
print(outDf['wasTracked'].value_counts())
print('\nColony features present:')
print(outDf['hasColonyFeatures'].value_counts())
print('\nWell features present:')
print(outDf['hasWellFeatures'].value_counts())
