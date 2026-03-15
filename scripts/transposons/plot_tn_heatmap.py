#!/usr/bin/env python3
"""
Heatmap of transposon biofilm biomass over time, normalized to WT peak.
Gene loci ordered by genomic position (VC_0001 ... VC_ANNNN).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

plt.rcParams['mathtext.fontset'] = 'stixsans'

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

# --- Paths ---
INDEX_CSV     = '/mnt/data/transposonSet/transposon_set_index.csv'
TRAINING_CSV  = '/mnt/data/trainingData/biomass_trajectories_clean_long.csv'
REPLICATE_CSV = '/home/smellick/ImageLibrary/ReplicatePositions.csv'
OUTDIR        = '/mnt/data/transposonSet/plots'
os.makedirs(OUTDIR, exist_ok=True)

# WT wells in training set
WT_WELLS = ['A5', 'B5', 'C5', 'D5', 'E11', 'F11', 'G11', 'H11']

# ===========================================================
# 1) Compute WT peak mean from training data
# ===========================================================
train = pd.read_csv(TRAINING_CSV).rename(columns={'plateID': 'plate', 'wellID': 'well'})
rep_map = pd.read_csv(REPLICATE_CSV).rename(columns={'Header': 'well', 'Replicate ID': 'mutant'})
train = train.merge(rep_map, on='well', how='left')

wt = train[(train['mutant'] == 'WT') & (train['well'].isin(WT_WELLS))]
wt_peaks = wt.groupby(['plate', 'well'])['biomass'].max()
wt_peak_mean = wt_peaks.mean()
print(f'WT peak mean: {wt_peak_mean:.6f}  (from {len(wt_peaks)} replicates)')

# ===========================================================
# 2) Load all TN timeseries from index
# ===========================================================
idx = pd.read_csv(INDEX_CSV)
idx = idx[idx['geneLocus'].notna() & (idx['timeseriesCsv'] != '')]

all_ts = []
for _, row in idx.iterrows():
    ts = pd.read_csv(row['timeseriesCsv'])
    ts['geneLocus'] = row['geneLocus']
    all_ts.append(ts[['geneLocus', 'frame', 'biomass']])

df = pd.concat(all_ts, ignore_index=True)
df['biomassNorm'] = df['biomass'] / wt_peak_mean

n_frames = df['frame'].nunique()
timepoints = np.sort(df['frame'].unique())
print(f'Loaded {len(idx)} wells, {df.geneLocus.nunique()} unique loci, {n_frames} frames')

# ===========================================================
# 3) Sort gene loci by genomic position
# ===========================================================
def locus_sort_key(locus):
    """VC_0001 -> (0, 1), VC_A0883 -> (1, 883)"""
    # Handle multi-locus entries (e.g. "VC_0005/VC_0006")
    first = locus.split('/')[0]
    m = re.match(r'VC_A?(\d+)', first)
    num = int(m.group(1)) if m else 0
    is_a = 1 if 'VC_A' in first else 0
    return (is_a, num)

loci = sorted(df['geneLocus'].unique(), key=locus_sort_key)
print(f'Gene loci range: {loci[0]} ... {loci[-1]}')

# ===========================================================
# 4) Build heatmap matrix
# ===========================================================
# Average replicates for loci that appear on multiple wells
# (most loci have 1 well, some have duplicates from multi-insertion)
rows = []
row_labels = []

for locus in loci:
    df_l = df[df['geneLocus'] == locus]
    # Pivot: mean biomass across replicates per frame
    traj = df_l.groupby('frame')['biomassNorm'].mean().reindex(timepoints)
    rows.append(traj.values)
    row_labels.append(locus)

heatmap = np.vstack(rows)

# ===========================================================
# 5) Plot
# ===========================================================
cmap = plt.cm.RdYlBu_r.copy()
cmap.set_bad('white')

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

im = ax.imshow(
    heatmap,
    aspect='auto',
    cmap=cmap,
    vmin=0, vmax=3,
    interpolation='nearest'
)

cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Biofilm Biomass (a.u.)', labelpad=8, fontsize=14)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=12)

ax.set_xlabel('Time (h)', fontsize=14)
ax.set_xticks(np.arange(0, n_frames, 5))
ax.set_xticklabels(np.arange(0, n_frames, 5).astype(int), fontsize=12)

# Y-axis: show every Nth label to avoid overlap
n_loci = len(row_labels)
label_step = max(1, n_loci // 60)
ytick_pos = list(range(0, n_loci, label_step))
ytick_labels = [row_labels[i] for i in ytick_pos]

ax.set_yticks(ytick_pos)
ax.set_yticklabels(ytick_labels, fontsize=5)
ax.set_ylabel('Gene Locus', fontsize=14)

plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap.png', dpi=400, bbox_inches='tight')
plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap.svg', bbox_inches='tight')
print(f'Saved to {OUTDIR}/tn_biofilm_heatmap.png')
plt.close()

# ===========================================================
# 6) Also save a ranked version (by peak biomass, descending)
# ===========================================================
peak_order = np.argsort(-np.nanmax(heatmap, axis=1))
heatmap_ranked = heatmap[peak_order]
labels_ranked = [row_labels[i] for i in peak_order]

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

im = ax.imshow(
    heatmap_ranked,
    aspect='auto',
    cmap=cmap,
    vmin=0, vmax=3,
    interpolation='nearest'
)

cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Biofilm Biomass (a.u.)', labelpad=8, fontsize=14)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=12)

ax.set_xlabel('Time (h)', fontsize=14)
ax.set_xticks(np.arange(0, n_frames, 5))
ax.set_xticklabels(np.arange(0, n_frames, 5).astype(int), fontsize=12)

ytick_pos = list(range(0, n_loci, label_step))
ytick_labels = [labels_ranked[i] for i in ytick_pos]
ax.set_yticks(ytick_pos)
ax.set_yticklabels(ytick_labels, fontsize=5)
ax.set_ylabel('Gene Locus (ranked by peak biomass)', fontsize=14)

plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap_ranked.png', dpi=400, bbox_inches='tight')
plt.savefig(f'{OUTDIR}/tn_biofilm_heatmap_ranked.svg', bbox_inches='tight')
print(f'Saved to {OUTDIR}/tn_biofilm_heatmap_ranked.png')
plt.close()
