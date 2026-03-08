#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import variation, skew
from scipy.sparse.csgraph import minimum_spanning_tree

def aggregateWellFeatures(colonyDf, allFrames, plateId, wellId):

    if colonyDf.empty:
        rows = []
        for frame in allFrames:
            rows.append({
                'plateID': plateId,
                'wellID': wellId,
                'frame': int(frame),
                'nColonies': 0,
            })
        return pd.DataFrame(rows)

    plateId = colonyDf['plateID'].iloc[0]
    wellId = colonyDf['wellID'].iloc[0]

    frames = np.asarray(allFrames)

    nColonies = (
        colonyDf.groupby('frame').size().reindex(frames, fill_value=0)
    )

    def safeMean(x):
        return x.mean() if len(x) > 0 else np.nan

    def safeVar(x):
        return x.var(ddof=1) if len(x) > 1 else np.nan

    def safeCv(x):
        m = np.nanmean(x)
        return np.nanstd(x) / m if m > 0 else np.nan

    rows = []

    for frame in frames:
        out = {
            'plateID': plateId,
            'wellID': wellId,
            'frame': int(frame),
            'nColonies': int(nColonies.loc[frame]),
        }

        g = colonyDf[colonyDf['frame'] == frame]

        if g.empty:
            # geometry / shape
            out.update({
                'totalColonyArea_px': np.nan,
                'meanColonyArea_px': np.nan,
                'medianColonyArea_px': np.nan,
                'maxColonyArea_px': np.nan,
                'cvColonyArea': np.nan,
                'meanPairwiseDistance_px': np.nan,
                'varPairwiseDistance_px': np.nan,
                'skewPairwiseDistance': np.nan,
                'mstMeanEdgeLength_px': np.nan,
                'mstMaxEdgeLength_px': np.nan,
                'mstCvEdgeLength': np.nan,
            })

            for col in ['circularity', 'eccentricity', 'aspectRatio', 'solidity']:
                out[f'mean_{col}'] = np.nan
                out[f'var_{col}'] = np.nan

            # intensity
            for k in [
                'meanIntensity_mean','meanIntensity_std','meanIntensity_cv',
                'integratedIntensity_sum','integratedIntensity_mean',
                'skewIntensity_mean','kurtosisIntensity_mean',
                'p90MeanIntensityRatio_mean'
            ]:
                out[k] = np.nan

            # mass / radial
            for k in [
                'massFracInner_mean','massFracOuter_mean',
                'outerMinusInnerMassFrac_mean','centroidOffsetNorm_mean'
            ]:
                out[k] = np.nan

            # background
            for k in [
                'bgMeanIntensity','bgStdIntensity','bgCV',
                'bgP10Intensity','bgP90Intensity'
            ]:
                out[k] = np.nan

            # NN + MST topology
            for k in [
                'nnDistance1_mean','nnDistance1_cv',
                'nnDistanceMeanK_mean','nnDistanceVarK_mean',
                'mstDegree_mean','mstDegree_max'
            ]:
                out[k] = np.nan

        else:
            # area
            areas = g['area_px'].values
            out['totalColonyArea_px'] = areas.sum()
            out['meanColonyArea_px'] = safeMean(areas)
            out['medianColonyArea_px'] = np.median(areas)
            out['maxColonyArea_px'] = np.max(areas)
            out['cvColonyArea'] = safeCv(areas)

            # shape
            for col in ['circularity', 'eccentricity', 'aspectRatio', 'solidity']:
                out[f'mean_{col}'] = safeMean(g[col])
                out[f'var_{col}'] = safeVar(g[col])

            # pairwise + MST
            if len(g) > 1:
                coords = g[['centroidX_px','centroidY_px']].values
                d = pdist(coords)
                out['meanPairwiseDistance_px'] = d.mean()
                out['varPairwiseDistance_px'] = d.var()
                out['skewPairwiseDistance'] = skew(d)

                mat = np.zeros((len(coords), len(coords)))
                iu = np.triu_indices(len(coords), 1)
                mat[iu] = d
                mat += mat.T

                mst = minimum_spanning_tree(mat).data
                out['mstMeanEdgeLength_px'] = mst.mean()
                out['mstMaxEdgeLength_px'] = mst.max()
                out['mstCvEdgeLength'] = safeCv(mst)
            else:
                out.update({
                    'meanPairwiseDistance_px': np.nan,
                    'varPairwiseDistance_px': np.nan,
                    'skewPairwiseDistance': np.nan,
                    'mstMeanEdgeLength_px': np.nan,
                    'mstMaxEdgeLength_px': np.nan,
                    'mstCvEdgeLength': np.nan,
                })

            # intensity
            out['meanIntensity_mean'] = g['meanIntensity'].mean()
            out['meanIntensity_std'] = g['meanIntensity'].std()
            out['meanIntensity_cv'] = safeCv(g['meanIntensity'])
            out['integratedIntensity_sum'] = g['integratedIntensity'].sum()
            out['integratedIntensity_mean'] = g['integratedIntensity'].mean()
            out['skewIntensity_mean'] = g['skewIntensity'].mean()
            out['kurtosisIntensity_mean'] = g['kurtosisIntensity'].mean()
            out['p90MeanIntensityRatio_mean'] = g['p90MeanIntensityRatio'].mean()

            # radial
            for k in [
                'massFracInner','massFracOuter',
                'outerMinusInnerMassFrac','centroidOffsetNorm'
            ]:
                out[f'{k}_mean'] = g[k].mean()

            # background (frame-global)
            for k in [
                'bgMeanIntensity','bgStdIntensity',
                'bgCV','bgP10Intensity','bgP90Intensity'
            ]:
                out[k] = g[k].iloc[0]

            # NN
            out['nnDistance1_mean'] = g['nnDistance1_px'].mean()
            out['nnDistance1_cv'] = safeCv(g['nnDistance1_px'])
            out['nnDistanceMeanK_mean'] = g['nnDistanceMeanK_px'].mean()
            out['nnDistanceVarK_mean'] = g['nnDistanceVarK_px'].mean()

            # MST degree
            out['mstDegree_mean'] = g['mstDegree'].mean()
            out['mstDegree_max'] = g['mstDegree'].max()

        rows.append(out)

    wellDf = pd.DataFrame(rows)

    assert (
        wellDf.loc[wellDf['nColonies'] == 0]
        .drop(columns=['plateID','wellID','frame','nColonies'])
        .isna()
        .all()
        .all()
    )

    return wellDf

