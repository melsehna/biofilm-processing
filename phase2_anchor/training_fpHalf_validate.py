#!/usr/bin/env python3
"""Phase 4 P0 — validate fpHalf render on raw training data before retiring adaptive.

For sampled training wells: render BOTH adaptive (fpMean=0.5*(max+min)) and fixed
fpHalf (0.5) from the same bit-depth-scaled, cropped raw stack, then (1) save an
adaptive-vs-fpHalf QC montage to eyeball, and (2) compare whole-image features to
quantify the adaptive-induced per-well drift fpHalf removes, and confirm fpHalf is
well-behaved (biology preserved, no over-clipping).
"""
import os, glob
import numpy as np, pandas as pd, tifffile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiWellAnalysis.processing.analysis_main import _toBitDepthScaled, cropStack
from multiWellAnalysis.processing.preprocessing import normalizeLocalContrastOutput
from multiWellAnalysis.wholeImage.extractWholeImageFeats import extractFrameFeats

PLATE = ('/mnt/bridgeslab/Good imaging data/Multi-phenotype training/'
         '241011_183053_4x_10x_20x_40x_Discontinuous_Drawer7 11-Oct-2024 16-30-30/'
         '241011_183053_Plate 1')
WELLS = ['B2', 'B5', 'B9', 'D5', 'F5']
SUFFIX = '_03'                 # 10x
BLOCK = 101
FEAT_FRAMES = [9, 12, 15, 18, 20]
QC_FRAMES = [5, 15, 25]
HARS = ['haralick_0', 'haralick_2', 'haralick_12', 'meanIntensity', 'entropy']
OUT = os.path.join(os.path.dirname(__file__), 'training_fpHalf_qc')
os.makedirs(OUT, exist_ok=True)


def loadStack(well):
    fs = sorted(glob.glob(os.path.join(PLATE, f'{well}{SUFFIX}_1_1_Bright Field_*.tif')))
    stack = np.stack([tifffile.imread(f) for f in fs], axis=-1)   # (H,W,T)
    raw = _toBitDepthScaled(stack)
    raw, _ = cropStack(raw)                                       # (H,W,T)
    return np.moveaxis(raw, -1, 0).astype(np.float32)             # (T,H,W)


def render(frame, fpMean):
    return np.clip(normalizeLocalContrastOutput(frame, BLOCK, fpMean), 0.0, 1.0)


rows = []
qcWells = WELLS[:3]
fig, axes = plt.subplots(len(qcWells), len(QC_FRAMES) * 2,
                         figsize=(3 * len(QC_FRAMES) * 2, 3 * len(qcWells)))
for wi, well in enumerate(WELLS):
    raw = loadStack(well)
    fpAdaptive = 0.5 * (float(np.nanmax(raw)) + float(np.nanmin(raw)))
    # feature comparison
    for t in FEAT_FRAMES:
        if t >= raw.shape[0]:
            continue
        a = extractFrameFeats(render(raw[t], fpAdaptive).astype(np.float32))
        h = extractFrameFeats(render(raw[t], 0.5).astype(np.float32))
        rows.append({'well': well, 'frame': t, 'fpAdaptive': fpAdaptive,
                     **{f'{k}_adaptive': a[k] for k in HARS},
                     **{f'{k}_fpHalf': h[k] for k in HARS}})
    print(f'{well}: adaptive fpMean={fpAdaptive:.4f}  done')
    # QC montage for first 3 wells
    if well in qcWells:
        for ci, t in enumerate(QC_FRAMES):
            for j, (lbl, fm) in enumerate([('adapt', fpAdaptive), ('fpHalf', 0.5)]):
                ax = axes[qcWells.index(well), ci * 2 + j]
                ax.imshow(render(raw[t], fm), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{well} {lbl} t{t}', fontsize=8); ax.axis('off')

fig.tight_layout()
mont = os.path.join(OUT, 'qc_adaptive_vs_fpHalf.png')
fig.savefig(mont, dpi=90); plt.close(fig)
print('wrote', mont)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'feature_comparison.csv'), index=False)

print('\n=== adaptive fpMean per well (does it drift well-to-well?) ===')
print(df.groupby('well').fpAdaptive.first().round(4).to_string())

print('\n=== feature: adaptive vs fpHalf (mean over wells*frames) + cross-well spread ===')
print(f'{"feature":14} {"adaptive":>10} {"fpHalf":>10} {"mean|Δ|":>10} '
      f'{"SD_adapt":>9} {"SD_fpHalf":>9}')
for k in HARS:
    a, h = df[f'{k}_adaptive'], df[f'{k}_fpHalf']
    # cross-well spread of per-well means (drift indicator)
    wa = df.groupby('well')[f'{k}_adaptive'].mean()
    wh = df.groupby('well')[f'{k}_fpHalf'].mean()
    print(f'{k:14} {a.mean():10.4f} {h.mean():10.4f} {np.abs(a-h).mean():10.4f} '
          f'{wa.std():9.4f} {wh.std():9.4f}')
print('\nQC: open', mont, '— fpHalf should look normal (biology visible, not washed/over-clipped).')
