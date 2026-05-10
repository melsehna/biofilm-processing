"""
Pivot the run-level master_frame_features.csv into the wide format expected
by the UMAP fit step: one row per well, one column per (feature, frame).

Filters to a single magnification — feature scales differ across objectives,
so cross-mag UMAPs are not meaningful.
"""
import pandas as pd


def _renameMasterColumn(col):
    if col == 'colAgg_nColonies':
        return 'nColonies'
    if col.startswith('colAgg_'):
        col = 'colony_' + col[len('colAgg_'):]
    if col.endswith('_skewness'):
        col = col[:-len('_skewness')] + '_skew'
    return col


def buildWideTable(masterFrameCsv, magnification, framesRange=None):
    """Read master_frame_features.csv and return a wide table for one magnification.

    Parameters
    ----------
    masterFrameCsv : str
        Path to master_frame_features.csv produced by assembleMasterCsvs.
    magnification : str
        Mag suffix to filter on, e.g. '_03'. Required — UMAP is per-mag.
    framesRange : tuple[int, int] or None
        Inclusive (min, max) frame range. If None, keep all frames present.

    Returns
    -------
    wide : pd.DataFrame
        Columns: plateId, wellId, then <feature>_t<frame> for every feature
        and frame present. One row per (plateId, wellId).
    """
    frame = pd.read_csv(masterFrameCsv)
    frame = frame[frame['mag'] == magnification].copy()
    if frame.empty:
        raise ValueError(f'No rows in {masterFrameCsv} with mag={magnification!r}')

    if framesRange is not None:
        lo, hi = framesRange
        frame = frame[frame['frame'].between(lo, hi)].copy()
        if frame.empty:
            raise ValueError(f'No rows in {masterFrameCsv} with mag={magnification!r} and frame in [{lo}, {hi}]')

    frame = frame.rename(columns={'plateID': 'plateId', 'wellID': 'wellId'})

    idCols = ['drawerID', 'plateId', 'wellId', 'mag', 'frame']
    featureCols = [c for c in frame.columns if c not in idCols]
    renameMap = {c: _renameMasterColumn(c) for c in featureCols}
    renamedFeatures = [renameMap[c] for c in featureCols]

    long = frame[['plateId', 'wellId', 'frame'] + featureCols].rename(columns=renameMap)
    long = long.melt(
        id_vars=['plateId', 'wellId', 'frame'],
        value_vars=renamedFeatures,
        var_name='feature',
        value_name='value',
    )
    long['feature'] = long['feature'] + '_t' + long['frame'].astype(int).astype(str)

    wide = long.pivot_table(
        index=['plateId', 'wellId'],
        columns='feature',
        values='value',
        aggfunc='first',
    ).reset_index()
    wide.columns.name = None
    return wide
