"""
Assemble run-level master CSVs from per-plate index.csv files.

Each plate's processedImages/index.csv has columns:
    plate, plate_path, well, mag, biomass, whole_image_feats,
    tracked_labels, colony_feats, well_colony_feats, ...

master_frame_features.csv   — one row per (drawerID, plateID, wellID, mag, frame)
master_colony_features.csv  — one row per (drawerID, plateID, wellID, mag, frame, colonyId)

assemblePlateNumericalData() writes per-plate, per-magnification CSVs into
a numericalData/ folder next to processedImages/:

  numericalData/<mag>X_BF.csv              — biomass timeseries, all wells
  numericalData/<mag>X_wholeImage.csv      — whole-image features, all wells (if run)
  numericalData/<mag>X_colonyFeatures.csv  — per-colony features, all wells (if run)
  numericalData/<mag>X_colonyAgg.csv       — well-level colony aggregates, all wells (if run)
"""

import os
import re
import pandas as pd

_wiIdentity  = {'plateid', 'wellid', 'processedpath'}
_wcaIdentity = {'plateid', 'wellid'}
_colIdentity = {'plateid', 'wellid'}

_magSuffixToObj = {
    '_02': '4X',
    '_03': '10X',
    '_04': '20X',
    '_05': '40X',
}


def _objLabel(magSuffix, rows):
    """Derive e.g. '10X' from index rows (preferred) or suffix fallback."""
    for irow in rows:
        obj = irow.get('objective', '')
        try:
            if obj:
                return f'{int(float(obj))}X'
        except (ValueError, TypeError):
            pass
    return _magSuffixToObj.get(magSuffix, magSuffix.lstrip('_') + 'X') if magSuffix else 'unknownX'


def _dropCols(df, lowerNameSet):
    toDrop = [c for c in df.columns if c.lower() in lowerNameSet]
    return df.drop(columns=toDrop)


def _prefixCols(df, prefix, keep=('frame',)):
    return df.rename(columns={c: f'{prefix}{c}' for c in df.columns if c not in keep})


def assembleMasterCsvs(plateOutdirs, drawerMap, outputRoot, logFn=None):
    """Read per-plate index files and write master CSVs to outputRoot.

    Parameters
    ----------
    plateOutdirs : list[str]
        Paths to processedImages/ directories, one per plate that ran.
    drawerMap : dict[str, str]
        Maps plate_name → drawer_name.
    outputRoot : str
        Directory where master_*.csv files will be written.
    logFn : callable, optional
    """
    def log(msg):
        if logFn:
            logFn(msg)

    frameRows = []
    colonyRows = []

    for outdir in plateOutdirs:
        indexPath = os.path.join(outdir, 'index.csv')
        if not os.path.exists(indexPath):
            log(f'  [master CSV] no index.csv in {outdir}, skipping')
            continue

        index = pd.read_csv(indexPath, dtype=str).fillna('')

        for _, irow in index.iterrows():
            plateName = irow['plate']
            wellId    = irow['well']
            mag       = irow.get('mag', '')
            drawerId  = drawerMap.get(plateName, plateName)

            biomassPath = irow.get('biomass', '')
            if not biomassPath or not os.path.exists(biomassPath):
                log(f'  [master CSV] {plateName}/{wellId}: no biomass CSV, skipping')
                continue

            merged = pd.read_csv(biomassPath)[['frame', 'biomass']]

            wiPath = irow.get('whole_image_feats', '')
            if wiPath and os.path.exists(wiPath):
                wi = _dropCols(pd.read_csv(wiPath), _wiIdentity)
                merged = merged.merge(wi, on='frame', how='outer')

            wcaPath = irow.get('well_colony_feats', '')
            if wcaPath and os.path.exists(wcaPath):
                wca = _dropCols(pd.read_csv(wcaPath), _wcaIdentity)
                wca = _prefixCols(wca, 'colAgg_')
                merged = merged.merge(wca, on='frame', how='outer')

            merged.insert(0, 'drawerID', drawerId)
            merged.insert(1, 'plateID',  plateName)
            merged.insert(2, 'wellID',   wellId)
            merged.insert(3, 'mag',      mag)

            frameRows.append(merged)

            cfPath = irow.get('colony_feats', '')
            if cfPath and os.path.exists(cfPath):
                cf = _dropCols(pd.read_csv(cfPath), _colIdentity)
                cf.insert(0, 'drawerID', drawerId)
                cf.insert(1, 'plateID',  plateName)
                cf.insert(2, 'wellID',   wellId)
                cf.insert(3, 'mag',      mag)
                colonyRows.append(cf)

    results = {}

    if frameRows:
        master = (pd.concat(frameRows, ignore_index=True)
                    .sort_values(['drawerID', 'plateID', 'wellID', 'mag', 'frame'])
                    .reset_index(drop=True))
        path = os.path.join(outputRoot, 'master_frame_features.csv')
        master.to_csv(path, index=False)
        results['frame'] = (path, len(master))
        log(f'  Master CSV (frame):  {path}  ({len(master):,} rows, {len(master.columns)} cols)')

    if colonyRows:
        master = (pd.concat(colonyRows, ignore_index=True)
                    .sort_values(['drawerID', 'plateID', 'wellID', 'mag', 'frame'])
                    .reset_index(drop=True))
        path = os.path.join(outputRoot, 'master_colony_features.csv')
        master.to_csv(path, index=False)
        results['colony'] = (path, len(master))
        log(f'  Master CSV (colony): {path}  ({len(master):,} rows, {len(master.columns)} cols)')

    return results


def assemblePlateNumericalData(processedImagesDir, logFn=None):
    """Write per-magnification CSVs into <plate_dir>/numericalData/.

    Reads the on-disk index.csv so that resumed runs include all wells.

    Parameters
    ----------
    processedImagesDir : str
        Path to the plate's processedImages/ directory.
    logFn : callable, optional
    """
    def log(msg):
        if logFn:
            logFn(msg)

    indexPath = os.path.join(processedImagesDir, 'index.csv')
    if not os.path.exists(indexPath):
        log(f'  [numericalData] no index.csv in {processedImagesDir}, skipping')
        return

    index = pd.read_csv(indexPath, dtype=str).fillna('')

    numericalDir = os.path.join(os.path.dirname(processedImagesDir), 'numericalData')
    os.makedirs(numericalDir, exist_ok=True)

    groups = {}
    for _, irow in index.iterrows():
        magSuffix = irow.get('mag', '')
        groups.setdefault(magSuffix, []).append(irow)

    for magSuffix, rows in sorted(groups.items()):
        objLabel = _objLabel(magSuffix, rows)

        bfDfs = []
        for irow in rows:
            wellId = irow['well']
            p = irow.get('biomass', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)[['frame', 'biomass']]
                df.insert(0, 'well', wellId)
                bfDfs.append(df)
        if bfDfs:
            out = os.path.join(numericalDir, f'{objLabel}_BF.csv')
            (pd.concat(bfDfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {objLabel}_BF.csv  ({len(bfDfs)} wells)')

        wiDfs = []
        for irow in rows:
            wellId = irow['well']
            p = irow.get('whole_image_feats', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, 'well', wellId)
                wiDfs.append(df)
        if wiDfs:
            out = os.path.join(numericalDir, f'{objLabel}_wholeImage.csv')
            (pd.concat(wiDfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {objLabel}_wholeImage.csv  ({len(wiDfs)} wells)')

        cfDfs = []
        for irow in rows:
            wellId = irow['well']
            p = irow.get('colony_feats', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, 'well', wellId)
                cfDfs.append(df)
        if cfDfs:
            out = os.path.join(numericalDir, f'{objLabel}_colonyFeatures.csv')
            (pd.concat(cfDfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {objLabel}_colonyFeatures.csv  ({len(cfDfs)} wells)')

        wcaDfs = []
        for irow in rows:
            wellId = irow['well']
            p = irow.get('well_colony_feats', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, 'well', wellId)
                wcaDfs.append(df)
        if wcaDfs:
            out = os.path.join(numericalDir, f'{objLabel}_colonyAgg.csv')
            (pd.concat(wcaDfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {objLabel}_colonyAgg.csv  ({len(wcaDfs)} wells)')
