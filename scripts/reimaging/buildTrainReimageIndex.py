#!/usr/bin/env python3

import os
import re
import pandas as pd
from glob import glob

trainingRoot = '/mnt/data/trainingData'
reimagingRoot = '/mnt/data/reimaging/processed'
replicateMapPath = '/home/smellick/ImageLibrary/ReplicatePositions.csv'

outTrainingIndex = '/mnt/data/datasets/training/index/trainingIndex.csv'
outReimagingIndex = '/mnt/data/datasets/reimaging/index/reimagingIndex.csv'

os.makedirs(os.path.dirname(outTrainingIndex), exist_ok=True)
os.makedirs(os.path.dirname(outReimagingIndex), exist_ok=True)

replicateMap = pd.read_csv(replicateMapPath)
replicateMap = replicateMap.set_index('Header')['Replicate ID'].to_dict()

def buildIndex(rootDir, mode):
    rows = []

    if mode == 'training':
        plateDirs = sorted(glob(os.path.join(rootDir, '*Plate*')))
    else:
        plateDirs = sorted(glob(os.path.join(rootDir, 'Plate*')))

    for plateDir in plateDirs:
        plateId = os.path.basename(plateDir)

        colonyFiles = glob(os.path.join(
            plateDir,
            '*_colonyFeatures_colFeats_microns_v1.csv'
        ))

        for colPath in colonyFiles:
            filename = os.path.basename(colPath)
            wellMatch = re.match(r'([A-H]\d+)_colonyFeatures', filename)
            if not wellMatch:
                continue

            wellId = wellMatch.group(1)

            biomassPath = os.path.join(
                plateDir,
                f'{wellId}_timeseries.csv'
            )

            wholePath = os.path.join(
                plateDir,
                f'{wellId}_wholeImage_mahotas_v2.csv'
            )

            if not os.path.exists(biomassPath):
                continue
            if not os.path.exists(wholePath):
                continue

            mutant = replicateMap.get(wellId, None)
            if mutant is None:
                continue

            rows.append({
                'plateId': plateId,
                'wellId': wellId,
                'mutant': mutant,
                'biomassPath': biomassPath,
                'wholePath': wholePath,
                'colPath': colPath
            })

    return pd.DataFrame(rows)


trainingIndex = buildIndex(trainingRoot, mode='training')
reimagingIndex = buildIndex(reimagingRoot, mode='reimaging')

trainingIndex.to_csv(outTrainingIndex, index=False)
reimagingIndex.to_csv(outReimagingIndex, index=False)

print('Training wells:', len(trainingIndex))
print('Reimaging wells:', len(reimagingIndex))