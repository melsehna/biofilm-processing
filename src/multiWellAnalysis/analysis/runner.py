"""
Top-level orchestration: master_frame_features.csv -> filtered wide table ->
fitted UMAP grid -> persisted embeddings/scaler/reducers -> static PNGs and
interactive HTML.

Importable from the GUI Run tab and from the CLI script.
"""
import json
import os
import pickle

import pandas as pd

from ..processing.master_csv import _magSuffixToObj
from .embedding import fitUmapGrid
from .interactive_html import writeInteractiveHtml
from .labels import loadLabels
from .static_plot import plotGrid, plotStatic
from .wide_table import buildWideTable


def _resolveLabels(plateIds, plateDirMap, conditionsByPlate, columnName):
    """Build a {(plateId, wellId): label} dict by calling loadLabels per plate.

    conditionsByPlate may be keyed by plateId; each value is the
    {conditionName: [wellIds]} dict for that plate. When a plate has no entry,
    label resolution falls back to its layout CSV (if any) and otherwise yields
    no labels for that plate.
    """
    out = {}
    cbp = conditionsByPlate or {}
    for plateId in plateIds:
        plateDir = (plateDirMap or {}).get(plateId)
        perPlate = loadLabels(
            plateDir=plateDir,
            conditionsMap=cbp.get(plateId),
            columnName=columnName,
        )
        for wellId, label in perPlate.items():
            out[(plateId, wellId)] = label
    return out


def _resolveMp4Paths(embeddings, outputRoot):
    """Compute relative MP4 paths from <outputRoot>/analysis/ to each well's overlay."""
    paths = []
    for _, row in embeddings.iterrows():
        plateId, wellId = str(row['plateId']), str(row['wellId'])
        mp4Abs = os.path.join(outputRoot, plateId, 'processedImages',
                              f'{wellId}_overlay.mp4')
        if os.path.exists(mp4Abs):
            paths.append(f'../{plateId}/processedImages/{wellId}_overlay.mp4')
        else:
            paths.append('')
    return paths


def _magObjectiveLabel(magnification, plateMeta=None):
    """Return a human label like '10X' for a mag suffix like '_03'.

    Prefers plateMeta (TIFF-resolved objective, authoritative). Falls back
    to the historical _magSuffixToObj convention used elsewhere in the
    codebase, then to the suffix verbatim.
    """
    if plateMeta:
        for plate, suffixMap in plateMeta.items():
            if magnification in suffixMap:
                obj = suffixMap[magnification].get('objective')
                if obj:
                    return f'{int(obj)}X'
    return _magSuffixToObj.get(magnification, magnification.lstrip('_') + 'X')


def runUmap(
    outputRoot,
    magnification,
    doStatic=True,
    doInteractive=True,
    plateDirMap=None,
    conditionsByPlate=None,
    columnName=None,
    plateMeta=None,
    framesRange=None,
    randomState=42,
    logFn=None,
):
    """Build UMAP outputs for one magnification.

    Parameters
    ----------
    outputRoot : str
        Directory containing master_frame_features.csv plus per-plate output dirs.
    magnification : str
        Mag suffix like '_03'. One UMAP per magnification.
    doStatic, doInteractive : bool
        Toggle the static PNGs and interactive HTML respectively.
    plateDirMap : dict[str, str] or None
        Maps plateId -> raw plate directory, used to find per-plate
        *layout*.csv sidecars for coloring.
    conditionsByPlate : dict[str, dict[str, list[str]]] or None
        Per-plate {conditionName: [wellIds]} from the GUI Conditions tab,
        keyed by plateId. Used for any plate without a layout CSV.
    columnName : str or None
        Column in the layout CSV to color by; defaults to col index 1.
    plateMeta : dict or None
        AppState plateMeta for resolving mag suffix -> objective label
        (e.g. '_03' -> '10X'). Cosmetic only.
    framesRange : tuple[int, int] or None
        Inclusive (min, max) frame range to include.
    randomState : int
    logFn : callable or None
        Receives one-line status messages.

    Returns
    -------
    dict[str, str]
        Map of artifact name -> output path.
    """
    def log(msg):
        if logFn:
            logFn(msg)

    masterPath = os.path.join(outputRoot, 'master_frame_features.csv')
    if not os.path.exists(masterPath):
        raise FileNotFoundError(f'master_frame_features.csv not found at {masterPath}')

    objLabel = _magObjectiveLabel(magnification, plateMeta)
    analysisDir = os.path.join(outputRoot, 'analysis')
    os.makedirs(analysisDir, exist_ok=True)
    prefix = os.path.join(analysisDir, f'umap_{objLabel}')

    log(f'  [UMAP {objLabel}] reading master_frame_features.csv ...')
    wide = buildWideTable(masterPath, magnification=magnification, framesRange=framesRange)
    log(f'  [UMAP {objLabel}] wide table: {wide.shape[0]} wells x {wide.shape[1]} cols')

    log(f'  [UMAP {objLabel}] fitting 9-cell UMAP grid (this may take a few minutes) ...')
    embeddings, excluded, scaler, reducers, featureCols = fitUmapGrid(
        wide, randomState=randomState,
    )
    log(f'  [UMAP {objLabel}] embedded {len(embeddings)} wells; {len(excluded)} below biomass floor')

    plateIds = sorted(embeddings['plateId'].astype(str).unique())
    labels = _resolveLabels(plateIds, plateDirMap, conditionsByPlate, columnName)
    nLabeled = sum(1 for v in labels.values() if v)
    log(f'  [UMAP {objLabel}] resolved labels for {nLabeled}/{len(embeddings)} wells')

    embeddings['mp4'] = _resolveMp4Paths(embeddings, outputRoot)
    nMp4 = sum(1 for p in embeddings['mp4'] if p)
    log(f'  [UMAP {objLabel}] overlay videos found for {nMp4}/{len(embeddings)} wells')

    out = {}

    embPath = f'{prefix}_embeddings.parquet'
    embeddings.to_parquet(embPath, index=False)
    out['embeddings'] = embPath

    if not excluded.empty:
        excPath = f'{prefix}_excluded.csv'
        excluded.to_csv(excPath, index=False)
        out['excluded'] = excPath

    scalerPath = f'{prefix}_scaler.pkl'
    with open(scalerPath, 'wb') as f:
        pickle.dump(scaler, f)
    out['scaler'] = scalerPath

    reducersPath = f'{prefix}_reducers.pkl'
    with open(reducersPath, 'wb') as f:
        pickle.dump(reducers, f)
    out['reducers'] = reducersPath

    featPath = f'{prefix}_features.json'
    with open(featPath, 'w') as f:
        json.dump(featureCols, f)
    out['features'] = featPath

    if doStatic:
        staticPath = f'{prefix}_static.png'
        plotStatic(embeddings, labels, staticPath, title=f'UMAP {objLabel}')
        out['static'] = staticPath
        log(f'  [UMAP {objLabel}] wrote {staticPath}')

        gridPath = f'{prefix}_grid.png'
        plotGrid(embeddings, labels, gridPath, title=f'UMAP {objLabel} (nn x min_dist grid)')
        out['grid'] = gridPath
        log(f'  [UMAP {objLabel}] wrote {gridPath}')

    if doInteractive:
        htmlPath = f'{prefix}_interactive.html'
        writeInteractiveHtml(embeddings, labels, htmlPath,
                             title=f'UMAP {objLabel}')
        out['interactive'] = htmlPath
        log(f'  [UMAP {objLabel}] wrote {htmlPath}')

    return out
