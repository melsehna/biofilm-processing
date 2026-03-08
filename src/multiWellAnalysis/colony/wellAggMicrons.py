#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import skew
from scipy.sparse.csgraph import minimum_spanning_tree

pxToUm = 0.697

requiredColonyColumns = {
    'plateID',
    'wellID',
    'frame',
    'area_um2',
    'centroidX_px',
    'centroidY_px',
    'circularity',
    'eccentricity',
    'aspectRatio',
    'solidity',
    'extent',
    'perimeterAreaRatio',
    'meanIntensity',
    'integratedIntensity',
    'maxMeanIntensityRatio',
    'centroidOffset_um',
    'centroidOffsetNorm',
    'bgMeanIntensity',
    'bgMedianIntensity',
    'bgStdIntensity',
    'bgP10Intensity',
    'bgP90Intensity',
    'bgCV',
    'nnDistance1_um',
    'nnDistanceMeanK_um',
    'nnDistanceVarK_um2',
    'mstDegree'
}


def validateColonySchema(colonyDf):
    missing = requiredColonyColumns - set(colonyDf.columns)
    if missing:
        raise ValueError(f'colonyDf missing required columns: {missing}')


def aggregateWellFeatures(colonyDf, allFrames, plateId, wellId):

    if colonyDf.empty:
        return pd.DataFrame({
            'plateID': plateId,
            'wellID': wellId,
            'frame': allFrames,
            'nColonies': 0
        })

    validateColonySchema(colonyDf)

    frames = np.asarray(allFrames)

    def safeMean(x):
        return x.mean() if len(x) > 0 else np.nan

    def safeVar(x):
        return x.var(ddof=1) if len(x) > 1 else np.nan

    def safeCv(x):
        m = np.nanmean(x)
        return np.nanstd(x, ddof=1) / m if m > 0 else np.nan

    rows = []

    for frame in frames:

        g = colonyDf[colonyDf['frame'] == frame]

        out = {
            'plateID': plateId,
            'wellID': wellId,
            'frame': int(frame),
            'nColonies': int(len(g))
        }

        if g.empty:
            rows.append(out)
            continue

        areas = g['area_um2'].values

        out['totalColonyArea_um2'] = areas.sum()
        out['meanColonyArea_um2'] = safeMean(areas)
        out['medianColonyArea_um2'] = np.median(areas)
        out['maxColonyArea_um2'] = np.max(areas)
        out['cvColonyArea'] = safeCv(areas)

        for col in [
            'circularity',
            'eccentricity',
            'aspectRatio',
            'solidity',
            'extent',
            'perimeterAreaRatio'
        ]:
            out[f'mean_{col}'] = safeMean(g[col])
            out[f'var_{col}'] = safeVar(g[col])

        if len(g) > 1:
            coords = g[['centroidX_px','centroidY_px']].values
            d_um = pdist(coords) * pxToUm

            out['meanPairwiseDistance_um'] = d_um.mean()
            out['varPairwiseDistance_um'] = d_um.var()
            out['skewPairwiseDistance'] = skew(d_um)

            mat = np.zeros((len(coords), len(coords)))
            iu = np.triu_indices(len(coords), 1)
            mat[iu] = d_um
            mat += mat.T

            mst = minimum_spanning_tree(mat).data
            out['mstMeanEdgeLength_um'] = mst.mean()
            out['mstMaxEdgeLength_um'] = mst.max()
            out['mstCvEdgeLength'] = safeCv(mst)
        else:
            out['meanPairwiseDistance_um'] = np.nan
            out['varPairwiseDistance_um'] = np.nan
            out['skewPairwiseDistance'] = np.nan
            out['mstMeanEdgeLength_um'] = np.nan
            out['mstMaxEdgeLength_um'] = np.nan
            out['mstCvEdgeLength'] = np.nan

        out['meanIntensity_mean'] = safeMean(g['meanIntensity'])
        out['meanIntensity_std'] = g['meanIntensity'].std()
        out['meanIntensity_cv'] = safeCv(g['meanIntensity'])
        out['integratedIntensity_sum'] = g['integratedIntensity'].sum()
        out['integratedIntensity_mean'] = safeMean(g['integratedIntensity'])
        out['maxMeanIntensityRatio_mean'] = safeMean(g['maxMeanIntensityRatio'])
        out['centroidOffset_um_mean'] = safeMean(g['centroidOffset_um'])
        out['centroidOffsetNorm_mean'] = safeMean(g['centroidOffsetNorm'])

        for k in [
            'bgMeanIntensity',
            'bgMedianIntensity',
            'bgStdIntensity',
            'bgP10Intensity',
            'bgP90Intensity',
            'bgCV'
        ]:
            out[k] = g[k].iloc[0]

        out['nnDistance1_mean'] = safeMean(g['nnDistance1_um'])
        out['nnDistance1_cv'] = safeCv(g['nnDistance1_um'])
        out['nnDistanceMeanK_mean'] = safeMean(g['nnDistanceMeanK_um'])
        out['nnDistanceVarK_mean'] = safeMean(g['nnDistanceVarK_um2'])

        out['mstDegree_mean'] = safeMean(g['mstDegree'])
        out['mstDegree_max'] = g['mstDegree'].max()

        rows.append(out)

    return pd.DataFrame(rows)