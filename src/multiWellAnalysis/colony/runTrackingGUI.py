#!/usr/bin/env python3
"""
GUI version of runTrackingMpTraining.py

Copy of the colony tracking pipeline adapted for the GUI:
- outdir is a required parameter (no hardcoded /mnt/data paths)
- No server-side logging (io_utils logWell/logPlate/checkpoints removed)
- No batch worker / multiprocessing main()
- Single entry point: trackAndSave()
"""

import os
import time

import numpy as np

from scipy.ndimage import distance_transform_edt, binary_fill_holes

from skimage.measure import label
from skimage.morphology import remove_small_objects

from multiWellAnalysis.colony.segmentation import segmentColonies


# config defaults

featVersion = 'trackingVec_v3'

propRadiusPx = 25
minColonyAreaPx = 200
connectivity = 2
borderMarginPx = 1

BIOMASS_THRESHOLD = 0.005


# utils

def cleanBinary(mask, min_area=minColonyAreaPx):
    binary = mask.astype(bool)
    binary = binary_fill_holes(binary)
    return remove_small_objects(binary, min_size=min_area)


def countComponents(mask):
    return label(mask.astype(bool), connectivity=connectivity).max()


def findSeedFrame(biomass, threshold=BIOMASS_THRESHOLD, minConsecutive=2):
    """Find first frame where biomass exceeds threshold for minConsecutive frames."""
    vals = np.asarray(biomass, dtype=float)
    for t in range(len(vals) - minConsecutive + 1):
        if np.all(vals[t:t + minConsecutive] >= threshold):
            return int(t)
    return None


def findBorderTouchingLabels(labels, marginPx):
    h, w = labels.shape
    border = np.zeros_like(labels, dtype=bool)
    border[:marginPx, :] = True
    border[-marginPx:, :] = True
    border[:, :marginPx] = True
    border[:, -marginPx:] = True
    return set(np.unique(labels[border])) - {0}


def stripBorderLabels(labels, marginPx):
    bad = findBorderTouchingLabels(labels, marginPx)
    if bad:
        labels = labels.copy()
        labels[np.isin(labels, list(bad))] = 0
    return labels


# tracking

def propagateLabelsFastVectorized(labelsPrev, maskNext, nextLabelId, effectiveRadius,
                                  min_area=minColonyAreaPx):
    binaryNext = cleanBinary(maskNext, min_area=min_area)
    labelsNext = np.zeros_like(labelsPrev, dtype=np.int32)

    if labelsPrev.max() == 0:
        return labelsNext, nextLabelId

    dist, indices = distance_transform_edt(
        labelsPrev == 0,
        return_indices=True
    )

    nearestLab = labelsPrev[indices[0], indices[1]]

    valid = (
        (nearestLab != 0) &
        (dist <= effectiveRadius) &
        binaryNext
    )

    labelsNext[valid] = nearestLab[valid]

    newMask = binaryNext & (labelsNext == 0)
    newLabels = label(newMask, connectivity=connectivity)

    for lab in range(1, newLabels.max() + 1):
        region = newLabels == lab
        if region.sum() >= min_area:
            labelsNext[region] = nextLabelId
            nextLabelId += 1

    return labelsNext, nextLabelId


def trackColoniesAllFrames(rawStack, maskStack, seedFrame, peakFrame,
                           min_area=minColonyAreaPx, prop_radius=propRadiusPx):
    """Core tracking algorithm. Returns (labelsByFrame, usedTracking, reason, frames)."""
    t0 = time.perf_counter()

    useTracking = True
    reason = 'forced_tracking'

    nFrames = rawStack.shape[2]
    frames = list(range(nFrames))

    # segment pre-seed frames independently
    labelsByFrame = {}

    for t in range(seedFrame):
        labs, _ = segmentColonies(
            rawStack[:, :, t],
            maskStack[:, :, t]
        )
        labelsByFrame[t] = labs.astype(np.int32)

    labels, _ = segmentColonies(
        rawStack[:, :, seedFrame],
        maskStack[:, :, seedFrame]
    )
    labels = labels.astype(np.int32)
    nextLabelId = labels.max() + 1

    labelsByFrame[seedFrame] = labels.copy()

    prevT = seedFrame

    for t in range(seedFrame + 1, nFrames):
        effRadius = prop_radius * (t - prevT if prevT < peakFrame else 1)
        labels, nextLabelId = propagateLabelsFastVectorized(
            labels,
            maskStack[:, :, t],
            nextLabelId,
            effRadius,
            min_area=min_area,
        )
        labelsByFrame[t] = labels.copy()
        prevT = t

    elapsed = time.perf_counter() - t0
    print(f'    tracking completed in {elapsed:.1f}s')

    return labelsByFrame, True, reason, frames


# main entry point

def trackAndSave(
    rawStack,
    maskStack,
    outdir,
    plateId,
    wellId,
    biomass=None,
    min_colony_area=None,
    prop_radius=None,
):
    """Run colony tracking and save labelled stack to outdir.

    Parameters
    ----------
    rawStack : ndarray (H, W, T)
    maskStack : ndarray (H, W, T)
    outdir : str
        Directory to write the tracked-labels NPZ into.
    plateId, wellId : str
    biomass : array-like or None
        Per-frame biomass values from timelapse_processing.
    min_colony_area : int or None
        Override for minimum colony area (px). Uses module default if None.
    prop_radius : int or None
        Override for propagation radius (px). Uses module default if None.

    Returns
    -------
    npzPath : str or None
        Path to the saved NPZ file, or None if tracking produced no output.
    """
    if min_colony_area is None:
        min_colony_area = minColonyAreaPx
    if prop_radius is None:
        prop_radius = propRadiusPx
    os.makedirs(outdir, exist_ok=True)

    nFrames = rawStack.shape[2]
    frames = list(range(nFrames))

    # find seed frame from biomass
    seedFrame = None
    if biomass is not None:
        seedFrame = findSeedFrame(biomass)

    if seedFrame is None:
        # fall back: use mask area as proxy
        maskAreas = np.array([maskStack[:, :, t].sum() for t in range(nFrames)], dtype=float)
        if maskAreas.max() > 0:
            normalised = maskAreas / maskAreas.max()
            seedFrame = findSeedFrame(normalised)

    if seedFrame is None:
        # no biomass detected — run per-frame segmentation only
        labelsByFrame = {}
        for t in frames:
            labs, _ = segmentColonies(rawStack[:, :, t], maskStack[:, :, t])
            labelsByFrame[t] = labs.astype(np.int32)
        usedTracking = False
        trackedFrames = np.zeros(nFrames, dtype=bool)
    else:
        # peak = frame with maximum mask area
        maskAreas = np.array([maskStack[:, :, t].sum() for t in range(nFrames)])
        peakFrame = int(np.argmax(maskAreas))

        labelsByFrame, usedTracking, reason, frames = trackColoniesAllFrames(
            rawStack, maskStack, seedFrame, peakFrame,
            min_area=min_colony_area, prop_radius=prop_radius,
        )

        trackedFrames = np.zeros(nFrames, dtype=bool)
        if usedTracking:
            trackedFrames[seedFrame:] = True

    # biomass-valid frames
    biomassValidFrames = np.zeros(nFrames, dtype=bool)
    if biomass is not None:
        for i in range(min(nFrames, len(biomass))):
            biomassValidFrames[i] = biomass[i] >= BIOMASS_THRESHOLD
    else:
        biomassValidFrames[:] = True

    # validity = tracked AND biomass-valid
    validMask = trackedFrames & biomassValidFrames

    # collect border-touching labels (global, once)
    allBadLabels = set()
    for t in frames:
        if t in labelsByFrame:
            allBadLabels |= findBorderTouchingLabels(labelsByFrame[t], borderMarginPx)

    # Build LUT once
    badMask = None
    if allBadLabels:
        maxLabel = max(l.max() for l in labelsByFrame.values() if l.max() > 0)
        if maxLabel > 0:
            badMask = np.zeros(maxLabel + 1, dtype=bool)
            badMask[list(allBadLabels)] = True

    # allocate FULL label stack
    H, W = rawStack.shape[:2]
    labelStackClean = np.zeros((H, W, nFrames), dtype=np.uint16)

    # populate ONLY valid frames
    for i, t in enumerate(frames):
        if not validMask[i]:
            continue

        lab = labelsByFrame.get(t)
        if lab is None:
            continue

        # Remove border-touching labels
        if badMask is not None:
            lab = lab.copy()
            lab[badMask[lab]] = 0

        labelStackClean[:, :, i] = lab

    # save
    npzPath = os.path.join(
        outdir,
        f'{wellId}_trackedLabels_allFrames_{featVersion}.npz'
    )
    np.savez_compressed(
        npzPath,
        labels=labelStackClean,
        frames=np.array(frames, dtype=np.int32),
        wasTracked=usedTracking,
        trackedFrames=trackedFrames,
        biomassValidFrames=biomassValidFrames,
        seedFrame=-1 if seedFrame is None else int(seedFrame),
        peakFrame=-1 if seedFrame is None else int(np.argmax(
            [maskStack[:, :, t].sum() for t in range(nFrames)]
        )),
        borderLabels=np.array(sorted(allBadLabels), dtype=np.int32)
    )

    return npzPath
