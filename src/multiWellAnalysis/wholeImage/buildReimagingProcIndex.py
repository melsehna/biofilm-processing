#!/usr/bin/env python3
import os
import pandas as pd

inputCsv = '/mnt/data/reimaging/index/reimaging_index_annotated.csv'
outCsv = '/mnt/data/reimaging/index/reimaging_processed_index.csv'
processedRoot = '/mnt/data/reimaging/processed'

df = pd.read_csv(inputCsv)
rows = []

for _, row in df.iterrows():

    repPlate = row['repPlate']
    repWell = row['repWell']
    plateDir = row['plateDir']

    # processed output lives under /mnt/data/reimaging/processed/<plateDir>
    plateProcessedDir = os.path.join(processedRoot, plateDir)
    processedImagesDir = os.path.join(plateProcessedDir, 'processedImages')

    if not os.path.isdir(processedImagesDir):
        print(f'skipping {plateDir} (no processedImages folder)')
        continue

    wellId = repWell

    processedPath = os.path.join(processedImagesDir, f'{wellId}_processed.tif')
    rawPath = os.path.join(processedImagesDir, f'{wellId}_registered_raw.tif')
    maskPath = os.path.join(processedImagesDir, f'{wellId}_masks.npz')
    biomassPath = os.path.join(plateProcessedDir, f'{wellId}_timeseries.csv')

    if not os.path.exists(processedPath):
        print(f'skipping {plateDir} {wellId} (missing processed stack)')
        continue

    if not os.path.exists(rawPath):
        print(f'skipping {plateDir} {wellId} (missing raw stack)')
        continue

    if not os.path.exists(maskPath):
        maskPath = None

    if not os.path.exists(biomassPath):
        biomassPath = None

    rows.append({
        'plateNo': repPlate,
        'plateId': plateDir,
        'wellId': repWell,
        'geneLocus': row['geneLocus'],
        'geneName': row['geneName'],
        'function': row['function'],
        'srcPlate': row['srcPlate'],
        'srcWell': row['srcWell'],
        'processedPath': processedPath,
        'rawPath': rawPath,
        'maskPath': maskPath,
        'biomassPath': biomassPath
    })

processedDf = pd.DataFrame(rows)
processedDf.to_csv(outCsv, index=False)

print(f'wrote: {outCsv}')
print(processedDf.head())