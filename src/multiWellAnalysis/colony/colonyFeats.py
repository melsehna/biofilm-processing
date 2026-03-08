import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.measure import regionprops, regionprops_table
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

pxToUm = 0.697
px2ToUm2 = pxToUm ** 2

def addColonyNeighborFeatures(colonyDf, k=5):
    n = len(colonyDf)

    colonyDf['nnDistance1_px'] = np.nan
    colonyDf['nnDistanceMeanK_px'] = np.nan
    colonyDf['nnDistanceVarK_px'] = np.nan

    coords = colonyDf[['centroidX_px', 'centroidY_px']].values
    nNeighbors = min(k + 1, n)

    nbrs = NearestNeighbors(
        n_neighbors=nNeighbors,
        algorithm='kd_tree'
    ).fit(coords)

    dists, _ = nbrs.kneighbors(coords)

    colonyDf['nnDistance1_px'] = dists[:, 1]
    colonyDf['nnDistanceMeanK_px'] = dists[:, 1:].mean(axis=1)
    colonyDf['nnDistanceVarK_px'] = dists[:, 1:].var(axis=1)
    
    colonyDf['nnDistance1_um'] = colonyDf['nnDistance1_px'] * pxToUm
    colonyDf['nnDistanceMeanK_um'] = colonyDf['nnDistanceMeanK_px'] * pxToUm
    colonyDf['nnDistanceVarK_um2'] = colonyDf['nnDistanceVarK_px'] * px2ToUm2

    return colonyDf

def addColonyGraphFeatures(colonyDf):
    n = len(colonyDf)

    if n < 2:
        colonyDf[['mstDegree','mstEdgeMean_px','mstEdgeMax_px']] = [0, np.nan, np.nan]
        return colonyDf

    if n < 2:
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
    colonyDf['mstEdgeMean_px'] = np.where(degrees > 0, edgeSums / degrees, np.nan)
    colonyDf['mstEdgeMax_px'] = np.where(degrees > 0, edgeMax, np.nan)
    
    

    return colonyDf



def extractColonyGeometry(labels, rawImg):
    props = regionprops_table(
        labels,
        intensity_image=rawImg,
        properties=[
            'label',
            'area',
            'perimeter',
            'convex_area',
            'bbox_area',
            'major_axis_length',
            'minor_axis_length',
            'eccentricity',
            'orientation',
            'euler_number',
            'centroid',
            'intensity_mean',
            'intensity_max'
        ]
    )

    df = pd.DataFrame(props)

    df.rename(columns={
        'label': 'colonyId',
        'area': 'area_px',
        'perimeter': 'perimeter_px',
        'convex_area': 'convexHullArea_px',
        'bbox_area': 'boundingBoxArea_px',
        'major_axis_length': 'majorAxisLength_px',
        'minor_axis_length': 'minorAxisLength_px',
        'orientation': 'orientation_rad',
        'intensity_mean': 'meanIntensity',
        'intensity_max': 'maxIntensity',
        'centroid-0': 'centroidY_px',
        'centroid-1': 'centroidX_px'
    }, inplace=True)

    df['extent'] = df['area_px'] / df['boundingBoxArea_px']
    df['solidity'] = df['area_px'] / df['convexHullArea_px']
    df['aspectRatio'] = df['majorAxisLength_px'] / df['minorAxisLength_px']
    df['circularity'] = 4 * np.pi * df['area_px'] / (df['perimeter_px'] ** 2)
    df['perimeterAreaRatio'] = df['perimeter_px'] / df['area_px']

    df['integratedIntensity'] = df['meanIntensity'] * df['area_px']
    df['maxMeanIntensityRatio'] = df['maxIntensity'] / df['meanIntensity']

    df['nHoles'] = 1 - df['euler_number']
    df['holeAreaFraction'] = np.where(df['nHoles'] > 0, 1 - df['solidity'], 0)
    
    df['area_um2'] = df['area_px'] * px2ToUm2
    df['perimeter_um'] = df['perimeter_px'] * pxToUm
    df['convexHullArea_um2'] = df['convexHullArea_px'] * px2ToUm2
    df['boundingBoxArea_um2'] = df['boundingBoxArea_px'] * px2ToUm2
    df['majorAxisLength_um'] = df['majorAxisLength_px'] * pxToUm
    df['minorAxisLength_um'] = df['minorAxisLength_px'] * pxToUm

    return df

def addColonySpatialFeatures(colonyDf):
    if colonyDf.empty:
        colonyDf['distanceToCenter_px'] = []
        colonyDf['angleToCenter_rad'] = []
        return colonyDf

    centerX = colonyDf['centroidX_px'].mean()
    centerY = colonyDf['centroidY_px'].mean()

    deltaX = colonyDf['centroidX_px'] - centerX
    deltaY = colonyDf['centroidY_px'] - centerY

    colonyDf['distanceToCenter_px'] = np.sqrt(deltaX ** 2 + deltaY ** 2)
    colonyDf['angleToCenter_rad'] = np.arctan2(deltaY, deltaX)
    
    colonyDf['distanceToCenter_um'] = colonyDf['distanceToCenter_px'] * pxToUm

    return colonyDf


from skimage.measure import regionprops

def addColonyIntensityMassFeatures(colonyDf, labels, rawImg):
    props = regionprops(labels, intensity_image=rawImg)

    # Preallocate
    n = len(props)
    out = {k: np.full(n, np.nan) for k in [
        'massCentroidX_px','massCentroidY_px','centroidOffset_px',
        'centroidOffsetNorm','massFracInner','massFracMid','massFracOuter',
        'outerInnerMassRatio','meanIntensity','medianIntensity','stdIntensity',
        'iqrIntensity','skewIntensity','kurtosisIntensity','p90MeanIntensityRatio'
    ]}
    
    props = {p.label: p for p in regionprops(labels, intensity_image=rawImg)}

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

        out['massCentroidX_px'][i] = cx
        out['massCentroidY_px'][i] = cy

        dx = cx - prop.centroid[1]
        dy = cy - prop.centroid[0]
        off = np.sqrt(dx*dx + dy*dy)

        out['centroidOffset_px'][i] = off
        out['centroidOffsetNorm'][i] = off / np.sqrt(prop.area / np.pi)

        d = np.sqrt((xs - prop.centroid[1])**2 + (ys - prop.centroid[0])**2)
        dnorm = d / d.max()

        inner = weights[dnorm <= 0.33].sum()
        mid = weights[(dnorm > 0.33) & (dnorm <= 0.66)].sum()
        outer = weights[dnorm > 0.66].sum()
        tot = weights.sum()

        out['massFracInner'][i] = inner / tot
        out['massFracMid'][i] = mid / tot
        out['massFracOuter'][i] = outer / tot
        out['outerInnerMassRatio'][i] = outer / inner if inner > 0 else np.nan

        out['meanIntensity'][i] = vals.mean()
        out['medianIntensity'][i] = np.median(vals)
        out['stdIntensity'][i] = vals.std()
        q75, q25 = np.percentile(vals, [75, 25])
        out['iqrIntensity'][i] = q75 - q25
        out['skewIntensity'][i] = skew(vals)
        out['kurtosisIntensity'][i] = kurtosis(vals)
        out['p90MeanIntensityRatio'][i] = np.percentile(vals, 90) / vals.mean()

    for k, v in out.items():
        colonyDf[k] = v

    colonyDf['outerMinusInnerMassFrac'] = (
        colonyDf['massFracOuter'] - colonyDf['massFracInner']
    )

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