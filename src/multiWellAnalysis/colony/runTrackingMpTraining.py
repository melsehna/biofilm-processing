#!/usr/bin/env python3

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio


import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt, binary_fill_holes

from skimage.measure import label
from skimage.measure import label as cc_label

from skimage.morphology import remove_small_objects

from multiWellAnalysis.colony.io_utils import (
    loadRawStack,
    loadMaskStack,
    ensureDir,
    logPlate,
    logWell,
    checkpointExists,
    writeCheckpoint,
    timestamp
)

from multiWellAnalysis.colony.segmentation import segmentColonies


# config

indexCsv = '/mnt/data/trainingData/processed_index.csv'
replicateCsv = '/home/smellick/ImageLibrary/ReplicatePositions.csv'

featVersion = 'trackingVec_v3'
checkpointTag = f'colony_{featVersion}'
logSuffix = f'_{featVersion}'

nProc = 8

excludeMuts = {'rbmA', 'bipA', 'lapG'}

propRadiusPx = 25
minColonyAreaPx = 200
connectivity = 2
borderMarginPx = 1

areaGrowthFactor = 3.5
componentDropFrac = 0.5


# utils

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def logWellAndPlate(plateId, wellId, msg):
    logWell(plateId, wellId, msg, suffix=logSuffix)
    logPlate(plateId, f'{wellId}: {msg}', suffix=logSuffix)


def cleanBinary(mask):
    binary = mask.astype(bool)
    binary = binary_fill_holes(binary)
    return remove_small_objects(binary, min_size=minColonyAreaPx)


def countComponents(mask):
    return label(mask.astype(bool), connectivity=connectivity).max()


def findSeedFrameFromBiomass(biomassCsv, threshold=0.005, minConsecutive=2):
    df = pd.read_csv(biomassCsv)
    biomass = df['biomass'].values
    for t in range(len(biomass) - minConsecutive + 1):
        if np.all(biomass[t:t + minConsecutive] >= threshold):
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



# gating

def needsTracking(maskStack, seedFrame, peakFrame, plateId, wellId):
    seedMask = maskStack[:, :, seedFrame]
    peakMask = maskStack[:, :, peakFrame]

    seedArea = seedMask.sum()
    peakArea = peakMask.sum()

    seedN = countComponents(seedMask)
    peakN = countComponents(peakMask)

    if seedArea == 0 or seedN == 0:
        return False, 'seed_empty'

    areaRatio = peakArea / seedArea
    componentRatio = peakN / seedN

    areaGate = peakArea < seedArea or areaRatio >= areaGrowthFactor
    componentGate = componentRatio <= componentDropFrac

    logWell(
        plateId,
        wellId,
        f'gating seedArea={seedArea} peakArea={peakArea} '
        f'areaRatio={areaRatio:.2f} componentRatio={componentRatio:.2f} '
        f'areaGate={areaGate} componentGate={componentGate}',
        suffix=logSuffix
    )

    if areaGate or componentGate:
        return True, 'gate_pass'
    return False, 'gate_fail'


# tracking

def propagateLabelsFastVectorized(labelsPrev, maskNext, nextLabelId, effectiveRadius):
    binaryNext = cleanBinary(maskNext)
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
        if region.sum() >= minColonyAreaPx:
            labelsNext[region] = nextLabelId
            nextLabelId += 1

    return labelsNext, nextLabelId


def trackColoniesAllFrames(rawStack, maskStack, seedFrame, peakFrame, plateId, wellId):
    t0 = time.perf_counter()


    useTracking = True
    reason = 'forced_tracking'

    logWell(
        plateId,
        wellId,
        'useTracking=True reason=forced_tracking',
        suffix=logSuffix
    )

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


    # tracking gate (legacy, disabled to run tracking on all mutants)
    # useTracking, reason = needsTracking(maskStack, seedFrame, peakFrame, plateId, wellId)
    # logWell(plateId, wellId, f'useTracking={useTracking} reason={reason}', suffix=logSuffix)

    # nFrames = rawStack.shape[2]
    # frames = list(range(seedFrame, nFrames))

    # if not useTracking:
    #     labelsByFrame = {
    #         t: segmentColonies(rawStack[:, :, t], maskStack[:, :, t])[0].astype(np.int32)
    #         for t in frames
    #     }
    #     return labelsByFrame, False, reason, frames

    labels, _ = segmentColonies(
        rawStack[:, :, seedFrame],
        maskStack[:, :, seedFrame]
    )
    labels = labels.astype(np.int32)
    nextLabelId = labels.max() + 1

    labelsByFrame[seedFrame] = labels.copy()

    prevT = seedFrame

    for t in range(seedFrame + 1, nFrames):
        effRadius = propRadiusPx * (t - prevT if prevT < peakFrame else 1)
        labels, nextLabelId = propagateLabelsFastVectorized(
            labels,
            maskStack[:, :, t],
            nextLabelId,
            effRadius
        )
        labelsByFrame[t] = labels.copy()
        prevT = t

    logWell(
        plateId,
        wellId,
        f'timing.tracking_allFrames={time.perf_counter() - t0:.3f}s',
        suffix=logSuffix
    )

    return labelsByFrame, True, reason, frames


# worker

def saveOverlayVideo(
    rawStack,
    labelStack,
    frames,
    biomassValidFrames,
    outPath,
    fps=5,
    alpha=0.45
):
    writer = imageio.get_writer(outPath, fps=fps)

    maxLabels = int(labelStack.max()) + 1
    rng = np.random.default_rng(0)
    colors = rng.random((maxLabels, 3))

    for i, t in enumerate(frames):
        raw = rawStack[:, :, t]
        labels = labelStack[:, :, i]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(raw, cmap='gray', interpolation='nearest')

        if labels.max() > 0:
            overlay = np.zeros((*labels.shape, 4), dtype=float)
            for lab in range(1, labels.max() + 1):
                sel = labels == lab
                overlay[sel, :3] = colors[lab]
                overlay[sel, 3] = alpha
            ax.imshow(overlay, interpolation='nearest')

        if not biomassValidFrames[i]:
            ax.set_title('below biomass threshold', fontsize=8, color='red')

        ax.axis('off')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame)

        plt.close(fig)

    writer.close()




# worker

def processOneWell(row):
    plateId = row['plateID']
    wellId = row['wellID']

    if checkpointExists(plateId, wellId, checkpointTag):
        logWellAndPlate(plateId, wellId, 'skipped (checkpoint exists)')
        return

    try:
        logWellAndPlate(plateId, wellId, 'status=in_progress')

        rawStack = loadRawStack(row['registered_raw_path'])
        maskStack = loadMaskStack(row['mask_path'])

        biomassCsv = (
            row['processed_path']
            .replace('processedImages', '')
            .rsplit('/', 1)[0]
            + f'/{wellId}_timeseries.csv'
        )
        
        nFrames = rawStack.shape[2]
        frames = list(range(nFrames))

        seedFrame = findSeedFrameFromBiomass(biomassCsv)

        if seedFrame is not None:
            trackedFrames = np.zeros(len(frames), dtype=bool)
            trackedFrames[seedFrame:] = True
        else:
            trackedFrames = np.zeros(len(frames), dtype=bool)

        biomassValidFrames = np.zeros(len(frames), dtype=bool)

        if os.path.exists(biomassCsv):
            tsDf = pd.read_csv(biomassCsv)

            if 'biomass' in tsDf.columns:
                biomass = tsDf['biomass'].values

                for i, t in enumerate(frames):
                    if t < len(biomass):
                        biomassValidFrames[i] = biomass[t] >= 0.005


        labelsByFrame = {}
        usedTracking = False
        reason = 'no_seed_frame'
        
        peakFrame = None

        if seedFrame is not None:
            peakFrame = int(row['peakFrame'])

            labelsByFrame, usedTracking, reason, frames = trackColoniesAllFrames(
                rawStack,
                maskStack,
                seedFrame,
                peakFrame,
                plateId,
                wellId
            )
            
            if not usedTracking:
                trackedFrames[:] = False


        else:
            logWell(
                plateId,
                wellId,
                'no seedFrame found; using per-frame segmentation only',
                suffix=logSuffix
            )

            for t in frames:
                labs, _ = segmentColonies(
                    rawStack[:, :, t],
                    maskStack[:, :, t]
                )
                labelsByFrame[t] = labs.astype(np.int32)

        ###
        
        # full (H, W, nFrames) label stack
        # Invalid frames are explicitly zeroed
        outdir = f'/mnt/data/trainingData/{plateId}/processedImages'
        ensureDir(outdir)

        labelsTracked = labelsByFrame
        nFrames = len(frames)

        # Step 1: decide which frames are valid
        # Valid = tracked AND above biomass threshold
        validMask = trackedFrames & biomassValidFrames
        # validMask.shape == (nFrames,)

        # Step 2: collect border-touching labels (global, once)
        allBadLabels = set()
        for t in frames:
            allBadLabels |= findBorderTouchingLabels(labelsTracked[t], borderMarginPx)

        # Build LUT once
        if allBadLabels:
            maxLabel = max(l.max() for l in labelsTracked.values())
            badMask = np.zeros(maxLabel + 1, dtype=bool)
            badMask[list(allBadLabels)] = True
        else:
            badMask = None

        # Step 3: allocate FULL label stack
        H, W = labelsTracked[frames[0]].shape
        labelStackClean = np.zeros((H, W, nFrames), dtype=np.uint16)

        # Step 4: populate ONLY valid frames
        for i, t in enumerate(frames):

            # Invalid frames stay zero
            if not validMask[i]:
                continue

            lab = labelsTracked.get(t)
            if lab is None:
                continue


            # Remove border-touching labels
            if badMask is not None:
                lab = lab.copy()
                lab[badMask[lab]] = 0

            labelStackClean[:, :, i] = lab

        # Step 5: save
        np.savez_compressed(
            f'{outdir}/{wellId}_trackedLabels_allFrames_{featVersion}.npz',
            labels=labelStackClean,
            frames=np.array(frames, dtype=np.int32),
            wasTracked=usedTracking,
            trackedFrames=trackedFrames,
            biomassValidFrames=biomassValidFrames,
            seedFrame=-1 if seedFrame is None else int(seedFrame),
            peakFrame=-1 if peakFrame is None else int(peakFrame),
            borderLabels=np.array(sorted(allBadLabels), dtype=np.int32)
        )



        # if wellId in {'A5', 'A9', 'B5', 'B9', 'C5', 'C9'}:
        #     overlayPath = (
        #         f'/mnt/data/trainingData/{plateId}/processedImages/'
        #         f'{wellId}_labels_{featVersion}.mp4'
        #     )

        #     saveOverlayVideo(
        #         rawStack=rawStack,
        #         labelStack=labelStackClean,
        #         frames=frames,
        #         biomassValidFrames=biomassValidFrames,
        #         outPath=overlayPath,
        #         fps=5
        #     )


        writeCheckpoint(
            plateId,
            wellId,
            checkpointTag,
            {
                'seedFrame': seedFrame,
                'peakFrame': peakFrame,
                'usedTracking': usedTracking,
                'reason': reason,
                'timestamp': timestamp()
            }
        )

        logWellAndPlate(plateId, wellId, 'status=done')

    except Exception:
        tb = traceback.format_exc()
        logWell(plateId, wellId, 'status=error', suffix=logSuffix)
        logWell(plateId, wellId, tb.replace('\n', ' | '), suffix=logSuffix)


# main

def main():
    indexDf = pd.read_csv(indexCsv)
    repDf = pd.read_csv(replicateCsv).rename(
        columns={'Header': 'wellID', 'Replicate ID': 'mutant'}
    )

    df = (
        indexDf
        .merge(repDf, on='wellID', how='left')
        .query('mutant not in @excludeMuts')
    )

    for plateId, plateDf in df.groupby('plateID'):
        print(f'[{ts()}] START plate {plateId}', flush=True)
        logPlate(plateId, '==== START tracking (all frames) ====', suffix=logSuffix)

        rows = plateDf.to_dict(orient='records')
        with ProcessPoolExecutor(max_workers=nProc) as ex:
            list(ex.map(processOneWell, rows))

        logPlate(plateId, '==== FINISHED tracking ====', suffix=logSuffix)
        print(f'[{ts()}] FINISHED plate {plateId}', flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
