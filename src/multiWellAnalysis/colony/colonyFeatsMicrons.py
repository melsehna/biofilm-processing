import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.measure import regionprops, regionprops_table
from scipy.stats import skew, kurtosis
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

pxToUm = 0.697
px2ToUm2 = pxToUm ** 2


def addColonyNeighborFeatures(colonyDf, k=5):
    n = len(colonyDf)
    if n < 2:
        colonyDf['nnDistance1_um'] = np.nan
        colonyDf['nnDistanceMeanK_um'] = np.nan
        colonyDf['nnDistanceVarK_um2'] = np.nan
        return colonyDf

    coords = colonyDf[['centroidX_px', 'centroidY_px']].values
    nNeighbors = min(k + 1, n)

    nbrs = NearestNeighbors(n_neighbors=nNeighbors, algorithm='kd_tree').fit(coords)
    dists, _ = nbrs.kneighbors(coords)

    colonyDf['nnDistance1_um'] = dists[:, 1] * pxToUm
    colonyDf['nnDistanceMeanK_um'] = dists[:, 1:].mean(axis=1) * pxToUm
    colonyDf['nnDistanceVarK_um2'] = dists[:, 1:].var(axis=1) * px2ToUm2

    return colonyDf


def addColonyGraphFeatures(colonyDf):
    n = len(colonyDf)
    if n < 2:
        colonyDf[['mstDegree','mstEdgeMean_um','mstEdgeMax_um']] = [0, np.nan, np.nan]
        return colonyDf

    coords = colonyDf[['centroidX_px', 'centroidY_px']].values
    distMat = squareform(pdist(coords))
    mst = minimum_spanning_tree(distMat).tocoo()

    degrees = np.zeros(n, dtype=int)
    edgeSums = np.zeros(n)
    edgeMax = np.zeros(n)

    for i, j, w in zip(mst.row, mst.col, mst.data):
        degrees[i] += 1
        degrees[j] += 1
        edgeSums[i] += w
        edgeSums[j] += w
        edgeMax[i] = max(edgeMax[i], w)
        edgeMax[j] = max(edgeMax[j], w)

    colonyDf['mstDegree'] = degrees
    colonyDf['mstEdgeMean_um'] = np.where(degrees > 0, (edgeSums / degrees) * pxToUm, np.nan)
    colonyDf['mstEdgeMax_um'] = np.where(degrees > 0, edgeMax * pxToUm, np.nan)

    return colonyDf


def extractColonyGeometry(labels, rawImg):
    props = regionprops_table(
        labels,
        intensity_image=rawImg,
        properties=[
            'label','area','perimeter','convex_area','bbox_area',
            'major_axis_length','minor_axis_length','eccentricity',
            'orientation','euler_number','centroid',
            'intensity_mean','intensity_max'
        ]
    )

    df = pd.DataFrame(props)

    df.rename(columns={
        'label': 'colonyId',
        'centroid-0': 'centroidY_px',
        'centroid-1': 'centroidX_px',
        'intensity_mean': 'meanIntensity',
        'intensity_max': 'maxIntensity'
    }, inplace=True)

    df['area_um2'] = df['area'] * px2ToUm2
    df['perimeter_um'] = df['perimeter'] * pxToUm
    df['convexHullArea_um2'] = df['convex_area'] * px2ToUm2
    df['boundingBoxArea_um2'] = df['bbox_area'] * px2ToUm2
    df['majorAxisLength_um'] = df['major_axis_length'] * pxToUm
    df['minorAxisLength_um'] = df['minor_axis_length'] * pxToUm

    df['extent'] = df['area'] / df['bbox_area']
    df['solidity'] = df['area'] / df['convex_area']
    df['aspectRatio'] = df['major_axis_length'] / df['minor_axis_length']
    df['circularity'] = 4 * np.pi * df['area'] / (df['perimeter'] ** 2)
    df['perimeterAreaRatio'] = df['perimeter'] / df['area']

    df['integratedIntensity'] = df['meanIntensity'] * df['area']
    df['maxMeanIntensityRatio'] = df['maxIntensity'] / df['meanIntensity']

    df['nHoles'] = 1 - df['euler_number']
    df['holeAreaFraction'] = np.where(df['nHoles'] > 0, 1 - df['solidity'], 0)

    df = df.drop(columns=[
        'area','perimeter','convex_area','bbox_area',
        'major_axis_length','minor_axis_length','euler_number'
    ])

    return df


def addColonySpatialFeatures(colonyDf):
    if colonyDf.empty:
        colonyDf['distanceToCenter_um'] = []
        colonyDf['angleToCenter_rad'] = []
        return colonyDf

    centerX = colonyDf['centroidX_px'].mean()
    centerY = colonyDf['centroidY_px'].mean()

    dx = colonyDf['centroidX_px'] - centerX
    dy = colonyDf['centroidY_px'] - centerY

    colonyDf['distanceToCenter_um'] = np.sqrt(dx ** 2 + dy ** 2) * pxToUm
    colonyDf['angleToCenter_rad'] = np.arctan2(dy, dx)

    return colonyDf


from skimage.measure import regionprops

def addColonyIntensityMassFeatures(colonyDf, labels, rawImg):
    props = {p.label: p for p in regionprops(labels, intensity_image=rawImg)}
    n = len(colonyDf)

    colonyDf['centroidOffset_um'] = np.nan
    colonyDf['centroidOffsetNorm'] = np.nan

    for i, cid in enumerate(colonyDf['colonyId'].astype(int)):
        prop = props.get(cid)
        if prop is None:
            continue

        ys, xs = np.nonzero(prop.image)
        xs = xs + prop.bbox[1]
        ys = ys + prop.bbox[0]
        vals = prop.intensity_image[prop.image]

        if vals.size == 0:
            continue

        weights = vals.astype(float)
        cx = np.sum(xs * weights) / weights.sum()
        cy = np.sum(ys * weights) / weights.sum()

        dx = cx - prop.centroid[1]
        dy = cy - prop.centroid[0]
        off_px = np.sqrt(dx*dx + dy*dy)

        colonyDf.loc[i, 'centroidOffset_um'] = off_px * pxToUm
        colonyDf.loc[i, 'centroidOffsetNorm'] = off_px / np.sqrt(prop.area / np.pi)

    return colonyDf


def extractBackgroundIntensityFeatures(rawImg, labels, dilateRadius=5):
    binaryColonies = labels > 0
    dilated = binary_dilation(binaryColonies, disk(dilateRadius))
    bgMask = (~dilated)
    vals = rawImg[bgMask]

    if vals.size == 0:
        return {
            'bgMeanIntensity': np.nan,
            'bgMedianIntensity': np.nan,
            'bgStdIntensity': np.nan,
            'bgP10Intensity': np.nan,
            'bgP90Intensity': np.nan,
            'bgCV': np.nan
        }

    mean = vals.mean()
    std = vals.std()

    return {
        'bgMeanIntensity': mean,
        'bgMedianIntensity': np.median(vals),
        'bgStdIntensity': std,
        'bgP10Intensity': np.percentile(vals, 10),
        'bgP90Intensity': np.percentile(vals, 90),
        'bgCV': std / mean if mean > 0 else np.nan
    }