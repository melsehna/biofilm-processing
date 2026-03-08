#!/usr/bin/env python3
import os
import pandas as pd
import tifffile
from glob import glob

root = '/mnt/data/trainingData'
outCsv = os.path.join(root, 'processed_index_wholeImage.csv')
mappingCsv = '/home/smellick/ImageLibrary/ReplicatePositions.csv'

excludeMutants = {'bipA', 'lapG', 'rbmA'}

mappingDf = pd.read_csv(mappingCsv)
wellToMutant = dict(zip(mappingDf['Header'], mappingDf['Replicate ID']))

rows = []

# def inferNFrames(tifPath):
#     with tifffile.TiffFile(tifPath) as tf:
#         shape = tf.series[0].shape
#         if len(shape) == 2:
#             return 1
#         if len(shape) == 3:
#             return min(shape)
#         return None


plateDirs = glob(os.path.join(root, '*Plate*'))

for plateDir in plateDirs:

    plateId = os.path.basename(plateDir)
    procDir = os.path.join(plateDir, 'processedImages')

    if not os.path.isdir(procDir):
        continue

    processedStacks = glob(os.path.join(procDir, '*_processed.tif'))

    for procPath in processedStacks:

        base = os.path.basename(procPath)
        wellId = base.split('_', 1)[0]

        mutant = wellToMutant.get(wellId)
        if mutant is None or mutant in excludeMutants:
            continue

        rawPath = os.path.join(procDir, f'{wellId}_registered_raw.tif')
        if not os.path.isfile(rawPath):
            continue

        maskPath = os.path.join(procDir, f'{wellId}_masks.npz')
        if not os.path.isfile(maskPath):
            maskPath = None

        # try:
        #     nFrames = inferNFrames(procPath)
        # except Exception:
        #     continue

        rows.append({
            'plateId': plateId,
            'wellId': wellId,
            'mutant': mutant,
            'processedPath': procPath,
            'rawPath': rawPath,
            'maskPath': maskPath
        })

df = pd.DataFrame(rows)
df.to_csv(outCsv, index=False)

print(f'Wrote index: {outCsv}')
print(df.head())