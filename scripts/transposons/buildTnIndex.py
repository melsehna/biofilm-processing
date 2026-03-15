import os
import re
import pandas as pd
from glob import glob

BASEDIR = '/mnt/bridgeslab/Good imaging data/TN-Library_imaging'
OUTDIR = '/mnt/data/transposonSet'
POSITIONS_CSV = '/mnt/data/transposonSet/Transposon positions.csv'

PLATES = [
    'TN-Plate01_4x_10x_20x_40x_Discontinuous_Drawer1 06-Nov-2024 14-57-44',
    'TN-Plate02_4x_10x_20x_40x_Discontinuous_Drawer2 25-Oct-2024 17-17-58',
    'TN-Plate03_4x_10x_20x_40x_Discontinuous_Drawer2 29-Oct-2024 14-46-08',
    'TN-Plate04_4x_10x_20x_40x_Discontinuous_Drawer3 29-Oct-2024 14-46-35',
    'TN-Plate07_4x_10x_20x_40x_Discontinuous_Drawer3 31-Oct-2024 13-36-05',
    'TN-Plate08_4x_10x_20x_40x_Discontinuous_Drawer1 04-Nov-2024 15-56-04',
    'TN-Plate09_4x_10x_20x_40x_Discontinuous_Drawer2 04-Nov-2024 15-56-34',
    'TN-Plate10_4x_10x_20x_40x_Discontinuous_Drawer3 04-Nov-2024 15-57-04',
    'TN-Plate11_4x_10x_20x_40x_Discontinuous_Drawer2 06-Nov-2024 14-58-16',
    'TN-Plate12_4x_10x_20x_40x_Discontinuous_Drawer3 06-Nov-2024 14-58-50',
    'TN-Plate13_4x_10x_20x_40x_Discontinuous_Drawer1 20-Nov-2024 16-25-11',
    'TN-Plate14_4x_10x_20x_40x_Discontinuous_Drawer2 20-Nov-2024 16-25-39',
    'TN-Plate15_4x_10x_20x_40x_Discontinuous_Drawer3 20-Nov-2024 16-47-34',
    'TN-Plate16_4x_10x_20x_40x_Discontinuous_Drawer1 11-Nov-2024 16-06-24',
    'TN-Plate17_4x_10x_20x_40x_Discontinuous_Drawer2 11-Nov-2024 16-06-55',
    'TN-Plate18_4x_10x_20x_40x_Discontinuous_Drawer3 11-Nov-2024 16-07-25',
    'TN-Plate19_4x_10x_20x_40x_Discontinuous_Drawer1 18-Nov-2024 11-46-38',
    'TN-Plate20_4x_10x_20x_40x_Discontinuous_Drawer2 18-Nov-2024 11-46-59',
    'TN-Plate21_4x_10x_20x_40x_Discontinuous_Drawer3 18-Nov-2024 12-47-49',
    'TN-Plate22_4x_10x_20x_40x_Discontinuous_Drawer1 12-Dec-2024 15-50-10',
    'TN-Plate23_4x_10x_20x_40x_Discontinuous_Drawer2 12-Dec-2024 15-50-38',
    'TN-Plate24_4x_10x_20x_40x_Discontinuous_Drawer2 19-Dec-2024 15-41-47',
    'TN-Plate25_4x_10x_20x_40x_Discontinuous_Drawer3 25-Nov-2024 15-53-17',
    'TN-Plate26_4x_10x_20x_40x_Discontinuous_Drawer4 25-Nov-2024 16-15-04',
    'TN-Plate27_4x_10x_20x_40x_Discontinuous_Drawer1 27-Nov-2024 13-22-12',
    'TN-Plate28_4x_10x_20x_40x_Discontinuous_Drawer2 27-Nov-2024 13-22-40',
    'TN-Plate29_4x_10x_20x_40x_Discontinuous_Drawer3 27-Nov-2024 13-23-07',
    'TN-Plate30_4x_10x_20x_40x_Discontinuous_Drawer1 04-Dec-2024 15-50-29',
    'TN-Plate31_4x_10x_20x_40x_Discontinuous_Drawer2 04-Dec-2024 15-50-59',
    'TN-Plate32_4x_10x_20x_40x_Discontinuous_Drawer3 04-Dec-2024 15-51-30',
    'TN-Plate33_4x_10x_20x_40x_Discontinuous_Drawer1 06-Dec-2024 16-04-05',
    'TN-Plate34_4x_10x_20x_40x_Discontinuous_Drawer2 06-Dec-2024 16-04-34',
]

records = []

for plateOuterName in PLATES:
    plateLabel = re.match(r'(TN-Plate\d+)', plateOuterName).group(1)
    outerDir = os.path.join(BASEDIR, plateOuterName)

    innerDirs = [
        d for d in os.listdir(outerDir)
        if os.path.isdir(os.path.join(outerDir, d)) and 'Plate' in d
    ]
    if not innerDirs:
        print(f'WARNING: no Plate subdir in {outerDir}')
        continue

    innerName = innerDirs[0]
    originalPlateDir = os.path.join(outerDir, innerName)
    outputPlateDir = os.path.join(OUTDIR, innerName)
    procDir = os.path.join(outputPlateDir, 'processedImages')

    if not os.path.isdir(outputPlateDir):
        print(f'WARNING: no output dir for {plateLabel}: {outputPlateDir}')
        continue

    timeseriesFiles = sorted(glob(os.path.join(outputPlateDir, '*_timeseries.csv')))

    for tsPath in timeseriesFiles:
        wellMag = os.path.basename(tsPath).replace('_timeseries.csv', '')
        wellId = re.sub(r'_\d+$', '', wellMag)
        plateId = f'{plateOuterName}/{innerName}'

        records.append({
            'plateId': plateId,
            'plateLabel': plateLabel,
            'wellId': wellId,
            'wellMag': wellMag,
            'originalImagesDir': originalPlateDir,
            'timeseriesCsv': tsPath if os.path.exists(tsPath) else '',
            'processedTif': os.path.join(procDir, f'{wellMag}_processed.tif') if os.path.exists(os.path.join(procDir, f'{wellMag}_processed.tif')) else '',
            'registeredRawTif': os.path.join(procDir, f'{wellMag}_registered_raw.tif') if os.path.exists(os.path.join(procDir, f'{wellMag}_registered_raw.tif')) else '',
            'masksNpz': os.path.join(procDir, f'{wellMag}_masks.npz') if os.path.exists(os.path.join(procDir, f'{wellMag}_masks.npz')) else '',
            'overlayMp4': os.path.join(procDir, f'{wellMag}_overlay.mp4') if os.path.exists(os.path.join(procDir, f'{wellMag}_overlay.mp4')) else '',
        })

df = pd.DataFrame(records)

pos = pd.read_csv(POSITIONS_CSV)
pos.columns = ['plateWell', 'geneLocus']
pos = pos.groupby('plateWell')['geneLocus'].agg(lambda x: '/'.join(x)).reset_index()

df['plateNum'] = df['plateLabel'].str.extract(r'TN-Plate(\d+)').astype(int)
df['plateWell'] = df['plateNum'].astype(str) + '-' + df['wellId']
df = df.merge(pos, on='plateWell', how='left')
df.drop(columns=['plateNum', 'plateWell'], inplace=True)

cols = ['geneLocus', 'plateId', 'plateLabel', 'wellId'] + [
    c for c in df.columns if c not in ('geneLocus', 'plateId', 'plateLabel', 'wellId')
]
df = df[cols]

outPath = os.path.join(OUTDIR, 'transposon_set_index.csv')
df.to_csv(outPath, index=False)

print(f'Index written to: {outPath}')
print(f'  {len(df)} wells across {df["plateId"].nunique()} plates')
print(f'  Gene locus matched: {df["geneLocus"].notna().sum()}/{len(df)}')
print(f'  Timeseries: {(df["timeseriesCsv"] != "").sum()}')
print(f'  Processed:  {(df["processedTif"] != "").sum()}')
print(f'  Raw reg:    {(df["registeredRawTif"] != "").sum()}')
print(f'  Masks:      {(df["masksNpz"] != "").sum()}')
print(f'  Overlays:   {(df["overlayMp4"] != "").sum()}')
