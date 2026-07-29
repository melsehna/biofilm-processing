"""
Static matplotlib UMAP plots: a single canonical panel and a 3x3 grid of all
fitted (n_neighbors, min_dist) pairs.

Coloring conventions follow the upstream reference: WT (if present) drawn
black with a larger marker on top; other labels assigned colors from a
turbo slice; unlabeled wells gray and drawn underneath.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_CANONICAL_NN = 20
_CANONICAL_MD = 0.2
_NN_GRID = (10, 20, 30)
_MD_GRID = (0.1, 0.2, 0.3)
_WT_LABEL = 'WT'
_TURBO_RANGE = (0.10, 0.80)
_MAX_LEGEND_CATEGORIES = 20


def _categoricalColors(categories):
    """Map non-WT categories to colors sliced from the turbo colormap."""
    cmap = plt.get_cmap('turbo')
    sortedCats = sorted(categories)
    if not sortedCats:
        return {}
    if len(sortedCats) == 1:
        return {sortedCats[0]: cmap(0.5)}
    lo, hi = _TURBO_RANGE
    return {c: cmap(lo + (hi - lo) * i / (len(sortedCats) - 1))
            for i, c in enumerate(sortedCats)}


def _splitByLabel(embeddings, labels):
    """Return (unlabeledIdx, wtIdx, {category: idx}) for plotting in z-order.

    Labels can be keyed either by (plateId, wellId) tuple (preferred for
    multi-plate runs) or by bare wellId. Tuple lookup is tried first.
    """
    def _lookup(row):
        key = (str(row['plateId']), str(row['wellId']))
        return labels.get(key) or labels.get(str(row['wellId'])) or None
    series = embeddings.apply(_lookup, axis=1)
    unlabeled = series.isna().to_numpy()
    wt = (series == _WT_LABEL).to_numpy()
    other = ~unlabeled & ~wt
    catGroups = {}
    if other.any():
        for cat in series[other].unique():
            catGroups[cat] = (series == cat).to_numpy()
    return unlabeled, wt, catGroups


def _scatterOnePanel(ax, embeddings, labels, nn, md, drawLegend=True):
    xCol = f'umap_x_nn{nn}_md{md}'
    yCol = f'umap_y_nn{nn}_md{md}'
    if xCol not in embeddings.columns or yCol not in embeddings.columns:
        ax.text(0.5, 0.5, f'no fit for nn={nn}, md={md}',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x = embeddings[xCol].to_numpy()
    y = embeddings[yCol].to_numpy()
    unlabeled, wt, catGroups = _splitByLabel(embeddings, labels)
    catColors = _categoricalColors(list(catGroups.keys()))

    if unlabeled.any():
        ax.scatter(x[unlabeled], y[unlabeled], c='lightgray', s=8, alpha=0.6,
                   linewidths=0, label='unlabeled' if drawLegend else None)
    for cat, mask in catGroups.items():
        ax.scatter(x[mask], y[mask], c=[catColors[cat]], s=14,
                   linewidths=0, label=cat if drawLegend else None)
    if wt.any():
        ax.scatter(x[wt], y[wt], c='black', s=24, linewidths=0,
                   label=_WT_LABEL if drawLegend else None)

    ax.set_xlabel('UMAP-1', fontsize=8)
    ax.set_ylabel('UMAP-2', fontsize=8)
    ax.tick_params(labelsize=7)


def plotStatic(embeddings, labels, outPath, nn=_CANONICAL_NN, md=_CANONICAL_MD,
               title=None):
    """Single-panel canonical UMAP PNG."""
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatterOnePanel(ax, embeddings, labels, nn, md, drawLegend=True)
    if title:
        ax.set_title(title, fontsize=11)
    nLabels = len(set(labels.values()))
    if 0 < nLabels <= _MAX_LEGEND_CATEGORIES:
        ax.legend(loc='best', fontsize=7, frameon=False, markerscale=1.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    return outPath


def plotGrid(embeddings, labels, outPath, nnList=_NN_GRID, mdList=_MD_GRID,
             title=None):
    """3x3 grid PNG: one panel per (nn, md) pair, no per-panel legends."""
    nRows, nCols = len(nnList), len(mdList)
    fig, axes = plt.subplots(nRows, nCols, figsize=(4 * nCols, 4 * nRows),
                             squeeze=False)
    for i, nn in enumerate(nnList):
        for j, md in enumerate(mdList):
            ax = axes[i][j]
            _scatterOnePanel(ax, embeddings, labels, nn, md, drawLegend=False)
            ax.set_title(f'nn={nn}, md={md}', fontsize=10)

    # one legend on the figure if labels are reasonable
    nLabels = len(set(labels.values()))
    if 0 < nLabels <= _MAX_LEGEND_CATEGORIES:
        handles, plotLabels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, plotLabels, loc='upper right',
                       bbox_to_anchor=(0.99, 0.99), fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92 if nLabels else 1, 0.97 if title else 1])
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    return outPath
