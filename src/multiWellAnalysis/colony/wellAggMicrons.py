#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Columns that are not per-colony features — excluded from stat aggregation
METADATA_COLS = {
    'plateID', 'wellID', 'frame',
    'trackedLabelsPath', 'registeredRawPath', 'wasTracked', 'colonyId'
}

# Background columns are frame-global (same value for every colony in a frame)
BG_COLS = {
    'bgMeanIntensity', 'bgMedianIntensity', 'bgStdIntensity',
    'bgP10Intensity', 'bgP90Intensity', 'bgCV'
}

SKIP_COLS = METADATA_COLS | BG_COLS


def aggregateWellFeatures(colonyDf, allFrames, plateId, wellId):
    frames = np.asarray(allFrames)

    if colonyDf.empty:
        return pd.DataFrame({
            'plateID': plateId,
            'wellID': wellId,
            'frame': frames,
            'nColonies': 0
        })

    # Per-colony numeric feature columns
    featCols = [
        c for c in colonyDf.columns
        if c not in SKIP_COLS and pd.api.types.is_numeric_dtype(colonyDf[c])
    ]

    presentBgCols = [c for c in BG_COLS if c in colonyDf.columns]

    rows = []
    for frame in frames:
        g = colonyDf[colonyDf['frame'] == frame]
        n = len(g)

        out = {
            'plateID': plateId,
            'wellID': wellId,
            'frame': int(frame),
            'nColonies': n,
        }

        # Background: frame-global, pass through directly
        for col in presentBgCols:
            out[col] = g[col].iloc[0] if n > 0 else np.nan

        for col in featCols:
            if n == 0:
                out[f'{col}_mean'] = np.nan
                out[f'{col}_std'] = np.nan
                out[f'{col}_var'] = np.nan
                out[f'{col}_skewness'] = np.nan
                out[f'{col}_kurtosis'] = np.nan
            else:
                vals = g[col].dropna().values
                nv = len(vals)
                zeroVar = nv >= 2 and vals.var() == 0
                out[f'{col}_mean'] = vals.mean() if nv >= 1 else np.nan
                out[f'{col}_std'] = 0.0 if zeroVar else (vals.std(ddof=1) if nv >= 2 else np.nan)
                out[f'{col}_var'] = 0.0 if zeroVar else (vals.var(ddof=1) if nv >= 2 else np.nan)
                out[f'{col}_skewness'] = np.nan if (nv < 3 or zeroVar) else float(skew(vals))
                out[f'{col}_kurtosis'] = np.nan if (nv < 4 or zeroVar) else float(kurtosis(vals, bias=False))

        rows.append(out)

    return pd.DataFrame(rows)
