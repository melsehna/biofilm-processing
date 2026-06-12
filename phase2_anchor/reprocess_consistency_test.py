#!/usr/bin/env python3
"""Reprocess-consistency test: does rendering BOTH batches through one fixed
pipeline collapse the haralick batch effect between clean-deletions and reimaging?

Stored atlas was rendered with an OLD near-zero-centered render; the clean-del
plate with a different version. This re-renders BOTH from registered_raw with the
SAME fixed-fpMean (fpHalf=0.5) render + current extraction, then compares WT↔WT
(the purest biology-matched control). If the ~50σ haralick gap collapses, the
batch effect was processing-version drift and consistent reprocessing is the fix.
"""
import numpy as np, pandas as pd, tifffile, os, glob
from multiWellAnalysis.processing.analysis_main import cropStack
from multiWellAnalysis.processing.preprocessing import normalizeLocalContrastOutput
from multiWellAnalysis.wholeImage.extractWholeImageFeats import extractFrameFeats

FRAMES = [9, 12, 15, 18, 20]
HARS = ['haralick_0', 'haralick_2', 'haralick_12']
N_PER = 5

reimDir = '/mnt/data/reimaging/processed'
cleanDir = ('/mnt/bridgeslab/Jesse/Project - Cluster/260512_hand_cleanDeletion_3/'
            '260512_hand_cleanDeletion_3/260512_174346_Plate 1/processedImages')


def renderExtract(rawPath):
    """Render registered_raw through fixed fpHalf and extract haralick per frame."""
    raw = tifffile.imread(rawPath)
    tax = int(np.argmin(raw.shape))   # time axis = frame count (~31), the smallest
    raw = np.moveaxis(raw, tax, 0).astype(np.float32)        # (T,H,W)
    raw = np.moveaxis(cropStack(np.moveaxis(raw, 0, -1))[0], -1, 0)  # crop NaN borders
    out = {}
    for t in FRAMES:
        if t >= raw.shape[0]:
            continue
        disp = np.clip(normalizeLocalContrastOutput(raw[t], 101, 0.5), 0.0, 1.0)
        f = extractFrameFeats(disp.astype(np.float32))
        out[t] = {h: f[h] for h in HARS}
    return out


# clean-del WT wells
cdMeta = pd.read_csv('/home/smellick/biofilm-analysis/data/reimaging_updated/'
                     'cleanDeletions_hand/cleanDeletion_metadata.csv')
cdWT = cdMeta[cdMeta.mutant == 'WT'].head(N_PER)
cdJobs = [(r.wellId, os.path.join(cleanDir, f'{r.well}_registered_raw.tif'))
          for _, r in cdWT.iterrows()]

# reimaging WT wells (join index paths + metadata mutant)
idx = pd.read_csv('/home/smellick/biofilm-analysis/data/indices/reimaging/reimagingIndex.csv')
rMeta = pd.read_csv('/home/smellick/biofilm-analysis/data/reimaging_updated/'
                    'reimaging_updated_metadata.csv')
rWT = rMeta[rMeta.mutant == 'WT'][['plateId', 'wellId']].merge(
    idx[['plateId', 'wellId', 'rawPath']], on=['plateId', 'wellId']).head(N_PER)
reimJobs = [(r.wellId, r.rawPath) for _, r in rWT.iterrows()]

print(f'clean-del WT wells: {[w for w,_ in cdJobs]}')
print(f'reimaging WT wells: {[w for w,_ in reimJobs]}')


def runBatch(name, jobs):
    rows = []
    for well, path in jobs:
        if not os.path.exists(path):
            print(f'  [{name}] MISSING {well}: {path}'); continue
        try:
            res = renderExtract(path)
            for t, d in res.items():
                rows.append({'batch': name, 'well': well, 'frame': t, **d})
            print(f'  [{name}] {well} done')
        except Exception as e:
            print(f'  [{name}] {well} ERROR {type(e).__name__}: {e}')
    return pd.DataFrame(rows)


print('\nReprocessing clean-del WT (fixed fpHalf render)...')
cd = runBatch('cleanDel', cdJobs)
print('Reprocessing reimaging WT (fixed fpHalf render)...')
rd = runBatch('reimaging', reimJobs)

print('\n=== AFTER consistent reprocessing (fixed fpHalf), WT haralick means ===')
print(f'{"feature":12} {"reim-WT":>10} {"cleanDel-WT":>12} {"raw Δ":>10} {"z (reimSD)":>12}')
for h in HARS:
    r = rd[h]; c = cd[h]
    rmu, rsd = r.mean(), r.std()
    cmu = c.mean()
    z = (cmu - rmu) / rsd if rsd > 0 else np.nan
    print(f'{h:12} {rmu:10.4f} {cmu:12.4f} {cmu-rmu:10.4f} {z:12.2f}')

print('\nFor reference, STORED (old, inconsistent renders) gap was ~ -50 z for haralick_12.')
print('If z here is now small (|z|<~3), the batch effect was processing-version drift.')
