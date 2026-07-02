"""
Filter, scale, and UMAP-embed a wide per-well feature table.

Fits the 3x3 grid (n_neighbors x min_dist) so the interactive viewer can
toggle between embeddings without re-running the fit. The filter rules
match the upstream biofilm-analysis reference pipeline: keep biomass and
whole-image entropy/haralick features, drop higher-moment summaries that
empirically add noise.
"""
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP


_NN_GRID = (10, 20, 30)
_MD_GRID = (0.1, 0.2, 0.3)
_BIOMASS_FLOOR = 0.005


def _isUmapFeature(col):
    """Reference filter: keep biomass* and whole_*entropy*/whole_*haralick*;
    drop everything ending in _skew, _kurtosis, or _cv (with or without _t<frame>).
    """
    base = col.split('_t')[0] if '_t' in col else col
    if base.endswith(('_skew', '_kurtosis', '_cv')):
        return False
    if base.startswith('biomass'):
        return True
    if base.startswith('whole_'):
        return ('entropy' in base) or ('haralick' in base)
    return False


def selectUmapFeatures(wideDf):
    """Return the list of columns in wideDf that pass the UMAP feature filter."""
    return [c for c in wideDf.columns if _isUmapFeature(c)]


def applyBiomassFloor(wideDf, floor=_BIOMASS_FLOOR):
    """Return (kept, excluded) split of wideDf based on max biomass per row."""
    biomassCols = [c for c in wideDf.columns
                   if c.startswith('biomass_t') or c == 'biomass']
    if not biomassCols:
        return wideDf.copy(), wideDf.iloc[:0].copy()
    maxBiomass = wideDf[biomassCols].max(axis=1, skipna=True)
    keepMask = maxBiomass > floor
    return wideDf[keepMask].copy(), wideDf[~keepMask].copy()


def fitUmapGrid(
    wideDf,
    nNeighborsList=_NN_GRID,
    minDistList=_MD_GRID,
    randomState=42,
    biomassFloor=_BIOMASS_FLOOR,
):
    """Filter, scale, and fit a grid of UMAP embeddings.

    Parameters
    ----------
    wideDf : pd.DataFrame
        Wide per-well table from buildWideTable. Must include columns
        ``plateId`` and ``wellId``.
    nNeighborsList, minDistList : iterable
        Grid axes. Defaults to (10, 20, 30) x (0.1, 0.2, 0.3) -> 9 fits.
    randomState : int
    biomassFloor : float
        Wells with max biomass <= floor are excluded from the embedding.

    Returns
    -------
    embeddings : pd.DataFrame
        Columns: plateId, wellId, then for each (nn, md) pair
        ``umap_x_nn{nn}_md{md}`` and ``umap_y_nn{nn}_md{md}``.
    excluded : pd.DataFrame
        plateId/wellId for rows dropped by the biomass floor.
    scaler : StandardScaler
    reducers : dict[tuple[int, float], UMAP]
        Fitted UMAP reducers, keyed by (n_neighbors, min_dist). Persist
        these alongside the scaler to project new wells onto the same grid.
    featureCols : list[str]
        Columns retained after filtering and zero-variance pruning.
    """
    if 'plateId' not in wideDf.columns or 'wellId' not in wideDf.columns:
        raise ValueError("wideDf must include 'plateId' and 'wellId' columns")

    kept, excluded = applyBiomassFloor(wideDf, floor=biomassFloor)
    if kept.empty:
        raise ValueError(
            f'No wells pass the biomass floor of {biomassFloor}. '
            f'Cannot fit UMAP on an empty input.'
        )

    featureCols = selectUmapFeatures(kept)
    if not featureCols:
        raise ValueError(
            'No columns matched the UMAP feature filter. '
            'Expected biomass* and/or whole_*entropy*/whole_*haralick*.'
        )

    X = kept[featureCols].fillna(0).to_numpy(dtype=np.float64)

    # drop zero-variance columns — UMAP and the scaler both choke on them
    variances = X.var(axis=0)
    nonzero = variances > 0
    if not nonzero.any():
        raise ValueError('All retained features have zero variance.')
    if (~nonzero).any():
        featureCols = [c for c, keep in zip(featureCols, nonzero) if keep]
        X = X[:, nonzero]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    nSamples = Xs.shape[0]
    out = kept[['plateId', 'wellId']].reset_index(drop=True).copy()
    reducers = {}

    for nn in nNeighborsList:
        nnEff = min(nn, nSamples - 1)
        if nnEff < 2:
            warnings.warn(
                f'Only {nSamples} samples — UMAP needs at least 2 neighbors; skipping nn={nn}.'
            )
            continue
        if nnEff != nn:
            warnings.warn(
                f'Clamping n_neighbors={nn} -> {nnEff} ({nSamples} samples).'
            )
        for md in minDistList:
            reducer = UMAP(
                n_neighbors=nnEff,
                min_dist=md,
                n_components=2,
                metric='euclidean',
                random_state=randomState,
                low_memory=True,
            )
            emb = reducer.fit_transform(Xs)
            out[f'umap_x_nn{nn}_md{md}'] = emb[:, 0]
            out[f'umap_y_nn{nn}_md{md}'] = emb[:, 1]
            reducers[(nn, md)] = reducer

    return out, excluded[['plateId', 'wellId']].reset_index(drop=True), scaler, reducers, featureCols
