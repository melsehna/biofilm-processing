#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd

# config
DATA_ROOT = '/mnt/data/trainingData'

INDEX_CSV = f'{DATA_ROOT}/processed_index.csv'
IN_GLOB = f'{DATA_ROOT}/*/processedColonyFeats/*_colonyFeatures_long.csv'

OUT_COLONY = f'{DATA_ROOT}/aggregated_colony_features.csv'
OUT_WELL = f'{DATA_ROOT}/aggregated_well_features.csv'

EPS = 1e-9

indexDf = pd.read_csv(INDEX_CSV)

longDfs = []
for csv in glob.glob(IN_GLOB):
    df = pd.read_csv(csv)
    if not df.empty:
        longDfs.append(df)

if not longDfs:
    raise RuntimeError('no colony feature CSVs found')

df = pd.concat(longDfs, ignore_index=True)


# merge peak frame info
df = df.merge(
    indexDf[['plateID', 'wellID', 'peakFrame']],
    on=['plateID', 'wellID'],
    how='left'
)


# temp features (per colony)
def addTemporalFeatures(colDf):
    colDf = colDf.sort_values('frame')

    frames = colDf['frame'].values
    areas = colDf['area_px'].values

    if len(colDf) < 2:
        colDf['dArea_dt'] = np.nan
        colDf['centroidDrift_px'] = np.nan
        colDf['shapeStability'] = np.nan
        return colDf

    dt = np.diff(frames)
    dA = np.diff(areas)

    dArea_dt = np.concatenate([[np.nan], dA / (dt + EPS)])
    colDf['dArea_dt'] = dArea_dt

    dx = np.diff(colDf['centroidX_px'].values)
    dy = np.diff(colDf['centroidY_px'].values)
    drift = np.concatenate([[0], np.sqrt(dx**2 + dy**2)])
    colDf['centroidDrift_px'] = drift

    shapeVar = colDf['circularity'].std()
    colDf['shapeStability'] = shapeVar

    return colDf


df = (
    df
    .groupby(['plateID', 'wellID', 'colonyId'], group_keys=False)
    .apply(addTemporalFeatures)
)


# peak frame filtering
peakDf = df[df['frame'] == df['peakFrame']].copy()


# per-colony aggregation
colonyAgg = (
    df
    .groupby(['plateID', 'wellID', 'colonyId'])
    .agg(
        wasTracked=('wasTracked', 'first'),
        nFrames=('frame', 'nunique'),

        area_mean_px=('area_px', 'mean'),
        area_max_px=('area_px', 'max'),
        area_std_px=('area_px', 'std'),

        meanGrowthRate=('dArea_dt', 'mean'),
        maxGrowthRate=('dArea_dt', 'max'),

        shapeStability=('shapeStability', 'first'),

        centroidDrift_mean_px=('centroidDrift_px', 'mean'),
        centroidDrift_max_px=('centroidDrift_px', 'max'),

        circularity_mean=('circularity', 'mean'),
        solidity_mean=('solidity', 'mean'),

        nnDist_mean_px=('nnDistance1_px', 'mean'),

        outerInnerMassBias=('outerMinusInnerMassFrac', 'mean')
    )
    .reset_index()
)


# per-well aggregation
wellAgg = (
    colonyAgg
    .groupby(['plateID', 'wellID'])
    .agg(
        nColonies=('colonyId', 'nunique'),
        wasTracked=('wasTracked', 'any'),

        meanColonyArea_px=('area_mean_px', 'mean'),
        maxColonyArea_px=('area_max_px', 'max'),

        meanGrowthRate=('meanGrowthRate', 'mean'),
        maxGrowthRate=('maxGrowthRate', 'max'),

        meanShapeStability=('shapeStability', 'mean'),

        meanCentroidDrift_px=('centroidDrift_mean_px', 'mean'),

        meanCircularity=('circularity_mean', 'mean'),
        meanSolidity=('solidity_mean', 'mean'),

        meanOuterInnerMassBias=('outerInnerMassBias', 'mean')
    )
    .reset_index()
)


# save
colonyAgg.to_csv(OUT_COLONY, index=False)
wellAgg.to_csv(OUT_WELL, index=False)

print(f'wrote {len(colonyAgg)} colony rows → {OUT_COLONY}')
print(f'wrote {len(wellAgg)} well rows → {OUT_WELL}')
