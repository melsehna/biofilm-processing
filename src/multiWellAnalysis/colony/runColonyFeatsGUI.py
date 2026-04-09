#!/usr/bin/env python3
"""
GUI version of runColonyFeatsTrackedMP.py

Copy of the colony feature extraction pipeline adapted for the GUI:
- outdir is a required parameter (no hardcoded OUT_ROOT)
- No server-side logging (io_utils logWell/logPlate/checkpoints removed)
- No batch worker / multiprocessing main()
- Uses wellAggMicrons (micron-scaled features)
- Single entry point: extractAndSave()
"""

import os

import numpy as np
import pandas as pd

from multiWellAnalysis.colony.colonyFeatsMicrons import (
    extractColonyGeometry,
    addColonySpatialFeatures,
    addColonyNeighborFeatures,
    addColonyGraphFeatures,
    addColonyIntensityMassFeatures,
    extractBackgroundIntensityFeatures
)

from multiWellAnalysis.colony.wellAggMicrons import aggregateWellFeatures


# config

featVersion = 'colFeats_microns_v1'
backgroundDilateRadius = 5


# core extraction

def extractTrackedColonyFeatures(
    rawStack,
    labelStack,
    frames,
    plateId,
    wellId,
    wasTracked,
    trackedLabelsPath,
    registeredRawPath,
    pxToUm=0.697,
):
    """Extract per-colony features for all non-empty frames.

    Parameters
    ----------
    rawStack : ndarray (H, W, T)
    labelStack : ndarray (H, W, T) — label IDs per pixel
    frames : array-like of int — frame indices
    plateId, wellId : str
    wasTracked : bool
    trackedLabelsPath, registeredRawPath : str — stored as metadata columns

    Returns
    -------
    pd.DataFrame — one row per colony per frame
    """
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

        colonyDf = extractColonyGeometry(labels, rawImg, pxToUm)
        if colonyDf.empty:
            continue

        for k, v in meta.items():
            colonyDf[k] = v

        colonyDf = addColonySpatialFeatures(colonyDf, pxToUm)
        colonyDf = addColonyNeighborFeatures(colonyDf, pxToUm)
        colonyDf = addColonyGraphFeatures(colonyDf, pxToUm)
        colonyDf = addColonyIntensityMassFeatures(colonyDf, labels, rawImg, pxToUm)

        bg = extractBackgroundIntensityFeatures(rawImg, labels, dilateRadius=backgroundDilateRadius)
        for k, v in bg.items():
            colonyDf[k] = v

        colonyDf['frame'] = int(t)

        rows.append(colonyDf)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, copy=False)


# main entry point

def extractAndSave(
    rawStack,
    labelStack,
    frames,
    plateId,
    wellId,
    wasTracked,
    trackedLabelsPath,
    rawPath,
    outdir,
    pxToUm=0.697,
):
    """Extract colony + well-level features and save CSVs to outdir.

    Parameters
    ----------
    rawStack : ndarray (H, W, T)
    labelStack : ndarray (H, W, T)
    frames : array-like
    plateId, wellId : str
    wasTracked : bool
    trackedLabelsPath, rawPath : str
    outdir : str
        Directory for the output CSVs (typically processedImages/).

    Returns
    -------
    (colonyDf, wellDf) or (None, None) if no colonies found
    """
    os.makedirs(outdir, exist_ok=True)

    colonyDf = extractTrackedColonyFeatures(
        rawStack,
        labelStack,
        frames,
        plateId,
        wellId,
        wasTracked,
        trackedLabelsPath,
        rawPath,
        pxToUm=pxToUm,
    )

    if colonyDf.empty:
        return None, None

    wellDf = aggregateWellFeatures(colonyDf, frames, plateId, wellId)

    colonyOutCsv = os.path.join(outdir, f'{wellId}_perColonyFeatures.csv')
    colonyDf.to_csv(colonyOutCsv, index=False)

    wellOutCsv = os.path.join(outdir, f'{wellId}_wellColonyFeatures.csv')
    wellDf.to_csv(wellOutCsv, index=False)

    return colonyDf, wellDf
