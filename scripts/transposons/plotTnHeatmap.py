#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 56,
    'axes.labelsize': 56,
    'axes.titlesize': 56,
    'xtick.labelsize': 56,
    'ytick.labelsize': 56,
    'legend.fontsize': 56,
    'font.family': 'Gillius ADF',
    'mathtext.fontset': 'stixsans',
})

INDEX_CSV = '/mnt/data/transposonSet/transposon_set_index.csv'
TRAINING_CSV = '/mnt/data/trainingData/biomass_trajectories_clean_long.csv'
REPLICATE_CSV = '/home/smellick/ImageLibrary/ReplicatePositions.csv'
OUTDIR = '/mnt/data/transposonSet/plots'
WT_WELLS = ['A5', 'B5', 'C5', 'D5', 'E11', 'F11', 'G11', 'H11']

os.makedirs(OUTDIR, exist_ok=True)

# WT normalization
train = pd.read_csv(TRAINING_CSV).rename(columns={'plateID': 'plate', 'wellID': 'well'})
repMap = pd.read_csv(REPLICATE_CSV).rename(columns={'Header': 'well', 'Replicate ID': 'mutant'})
train = train.merge(repMap, on='well', how='left')

wt = train[(train['mutant'] == 'WT') & (train['well'].isin(WT_WELLS))]
wtPeaks = wt.groupby(['plate', 'well'])['biomass'].max()
wtPeakMean = wtPeaks.mean()
print(f'WT peak mean: {wtPeakMean:.6f}  (from {len(wtPeaks)} replicates)')

# load TN timeseries
idx = pd.read_csv(INDEX_CSV)
idx = idx[idx['geneLocus'].notna() & (idx['timeseriesCsv'] != '')]

allTs = []
for _, row in idx.iterrows():
    ts = pd.read_csv(row['timeseriesCsv'])
    ts['geneLocus'] = row['geneLocus']
    allTs.append(ts[['geneLocus', 'frame', 'biomass']])

df = pd.concat(allTs, ignore_index=True)
df['biomassNorm'] = df['biomass'] / wtPeakMean

nFrames = df['frame'].nunique()
timepoints = np.sort(df['frame'].unique())
print(f'Loaded {len(idx)} wells, {df.geneLocus.nunique()} unique loci, {nFrames} frames')

# sort loci by genomic position
def locusSortKey(locus):
    first = locus.split('/')[0]
    m = re.match(r'VC_A?(\d+)', first)
    num = int(m.group(1)) if m else 0
    isA = 1 if 'VC_A' in first else 0
    return (isA, num)

loci = sorted(df['geneLocus'].unique(), key=locusSortKey)
print(f'Gene loci range: {loci[0]} ... {loci[-1]}')

# build heatmap matrix
rows = []
rowLabels = []

for locus in loci:
    dfL = df[df['geneLocus'] == locus]
    traj = dfL.groupby('frame')['biomassNorm'].mean().reindex(timepoints)
    rows.append(traj.values)
    rowLabels.append(locus)

heatmap = np.vstack(rows)
nLoci = len(rowLabels)
labelStep = max(1, nLoci // 60)

cmap = plt.cm.RdYlBu_r.copy()
cmap.set_bad('white')

# genomic order heatmap
fig = plt.figure(figsize=(8, 48))
gs = fig.add_gridspec(
    nrows=3, ncols=1,
    height_ratios=[0.3, 20, 0.01],
    left=0.22, right=0.98,
    bottom=0.02, top=0.97,
    hspace=0.08
)

cax = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=0, vmax=3, interpolation='nearest')

cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Biofilm Biomass (a.u.)', labelpad=8, fontsize=14)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=12)

ax.set_xlabel('Time (h)', fontsize=14)
ax.set_xticks(np.arange(0, nFrames, 5))
ax.set_xticklabels(np.arange(0, nFrames, 5).astype(int), fontsize=12)

ytickPos = list(range(0, nLoci, labelStep))
ax.set_yticks(ytickPos)
ax.set_yticklabels([rowLabels[i] for i in ytickPos], fontsize=5)
ax.set_ylabel('Gene Locus', fontsize=14)

plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap.png', dpi=400, bbox_inches='tight')
plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap.svg', bbox_inches='tight')
print(f'Saved to {OUTDIR}/tn_biofilm_heatmap.png')
plt.close()

# ranked by peak biomass
peakOrder = np.argsort(-np.nanmax(heatmap, axis=1))
heatmapRanked = heatmap[peakOrder]
labelsRanked = [rowLabels[i] for i in peakOrder]

fig = plt.figure(figsize=(8, 48))
gs = fig.add_gridspec(
    nrows=3, ncols=1,
    height_ratios=[0.3, 20, 0.01],
    left=0.22, right=0.98,
    bottom=0.02, top=0.97,
    hspace=0.08
)

cax = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

im = ax.imshow(heatmapRanked, aspect='auto', cmap=cmap, vmin=0, vmax=3, interpolation='nearest')

cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Biofilm Biomass (a.u.)', labelpad=8, fontsize=14)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=12)

ax.set_xlabel('Time (h)', fontsize=14)
ax.set_xticks(np.arange(0, nFrames, 5))
ax.set_xticklabels(np.arange(0, nFrames, 5).astype(int), fontsize=12)

ytickPos = list(range(0, nLoci, labelStep))
ax.set_yticks(ytickPos)
ax.set_yticklabels([labelsRanked[i] for i in ytickPos], fontsize=5)
ax.set_ylabel('Gene Locus (ranked by peak biomass)', fontsize=14)

plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap_ranked.png', dpi=400, bbox_inches='tight')
plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap_ranked.svg', bbox_inches='tight')
print(f'Saved to {OUTDIR}/tn_biofilm_heatmap_ranked.png')
plt.close()
