"""
Resolve a per-well coloring label from one of two sources:

1. A per-plate sidecar CSV with 'layout' in its filename (preferred). The first
   column is treated as wellId; the user picks any other column to color by,
   defaulting to column index 1 if unspecified.

2. The GUI Conditions mapping ({conditionName: [wellIds]}), inverted into
   {wellId: conditionName}.

Returns an empty dict when neither source resolves anything — callers should
treat that as "no coloring; metadata still in tooltip."
"""
import glob
import os

import pandas as pd


def _findLayoutCsv(plateDir):
    candidates = [p for p in glob.glob(os.path.join(plateDir, '*'))
                  if 'layout' in os.path.basename(p).lower()
                  and p.lower().endswith('.csv')]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def loadLabels(plateDir=None, conditionsMap=None, columnName=None):
    """Resolve {wellId: label} for a single plate.

    Parameters
    ----------
    plateDir : str or None
        Plate directory to search for a `*layout*.csv` sidecar. If None, the
        layout-CSV path is skipped.
    conditionsMap : dict[str, list[str]] or None
        GUI Conditions mapping ({conditionName: [wellIds]}). Used as a fallback
        when no layout CSV is found.
    columnName : str or None
        Name of the column in the layout CSV to color by. If None, defaults to
        the column at index 1 (i.e. the first column after wellId).

    Returns
    -------
    labels : dict[str, str]
        Mapping from wellId to a string label. Empty if no source resolved.
    """
    if plateDir is not None:
        layoutPath = _findLayoutCsv(plateDir)
        if layoutPath is not None:
            df = pd.read_csv(layoutPath)
            if df.shape[1] < 2:
                raise ValueError(
                    f'Layout CSV {layoutPath} has only {df.shape[1]} column(s); '
                    f'need at least 2 (wellId + label).'
                )
            wellCol = df.columns[0]
            if columnName is None:
                labelCol = df.columns[1]
            elif columnName in df.columns:
                labelCol = columnName
            else:
                raise ValueError(
                    f'Column {columnName!r} not in layout CSV {layoutPath}. '
                    f'Available: {list(df.columns)}'
                )
            sub = df[[wellCol, labelCol]].dropna(subset=[wellCol])
            return {str(w): str(v) for w, v in zip(sub[wellCol], sub[labelCol])
                    if pd.notna(v)}

    if conditionsMap:
        inverted = {}
        for cond, wells in conditionsMap.items():
            for w in wells:
                inverted[str(w)] = str(cond)
        return inverted

    return {}
