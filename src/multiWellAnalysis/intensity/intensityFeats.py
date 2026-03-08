import numpy as np
import pandas as pd
from skimage.measure import regionprops
from scipy.stats import skew, kurtosis
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


def addColonyIntensityFeatures(colonyDf, labels, img, prefix):
    """
    Per-colony intensity statistics from a given image.

    prefix examples:
        rawCol
        procCol
    """
    props = {p.label: p for p in regionprops(labels, intensity_image=img)}

    n = len(colonyDf)
    feats = {
        f'{prefix}_mean': np.full(n, np.nan),
        f'{prefix}_median': np.full(n, np.nan),
        f'{prefix}_std': np.full(n, np.nan),
        f'{prefix}_iqr': np.full(n, np.nan),
        f'{prefix}_skew': np.full(n, np.nan),
        f'{prefix}_kurtosis': np.full(n, np.nan),
        f'{prefix}_p90MeanRatio': np.full(n, np.nan),
    }

    for i, cid in enumerate(colonyDf['colonyId'].astype(int)):
        prop = props.get(cid)
        if prop is None:
            continue

        vals = prop.intensity_image[prop.image]
        if vals.size == 0:
            continue

        mean = vals.mean()
        q75, q25 = np.percentile(vals, [75, 25])

        feats[f'{prefix}_mean'][i] = mean
        feats[f'{prefix}_median'][i] = np.median(vals)
        feats[f'{prefix}_std'][i] = vals.std()
        feats[f'{prefix}_iqr'][i] = q75 - q25
        feats[f'{prefix}_skew'][i] = skew(vals)
        feats[f'{prefix}_kurtosis'][i] = kurtosis(vals)
        feats[f'{prefix}_p90MeanRatio'][i] = (
            np.percentile(vals, 90) / mean if mean > 0 else np.nan
        )

    for k, v in feats.items():
        colonyDf[k] = v

    return colonyDf


def extractBackgroundIntensityFeatures(img, labels, dilateRadius=5, prefix='bg'):
    """
    Frame-level background intensity features.

    prefix examples:
        bgRaw
        bgProc
    """
    binaryColonies = labels > 0
    dilated = binary_dilation(binaryColonies, disk(dilateRadius))
    bgMask = ~dilated

    vals = img[bgMask]

    if vals.size == 0:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_median': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_p10': np.nan,
            f'{prefix}_p90': np.nan,
            f'{prefix}_cv': np.nan,
        }

    mean = vals.mean()
    std = vals.std()

    return {
        f'{prefix}_mean': mean,
        f'{prefix}_median': np.median(vals),
        f'{prefix}_std': std,
        f'{prefix}_p10': np.percentile(vals, 10),
        f'{prefix}_p90': np.percentile(vals, 90),
        f'{prefix}_cv': std / mean if mean > 0 else np.nan,
    }
