"""
Assemble run-level master CSVs from per-plate index.csv files.

Each plate's processedImages/index.csv has columns:
    plate, plate_path, well, mag, biomass, whole_image_feats,
    tracked_labels, colony_feats, well_colony_feats, ...

This module reads those index files and joins the per-well feature CSVs
into two flat tables:

  master_frame_features.csv   — one row per (drawerID, plateID, wellID, mag, frame)
  master_colony_features.csv  — one row per (drawerID, plateID, wellID, mag, frame, colonyId)

It also provides assemble_plate_numerical_data() which writes per-plate,
per-magnification CSVs into a numericalData/ folder next to processedImages/:

  numericalData/<mag>X_BF.csv              — biomass timeseries, all wells
  numericalData/<mag>X_wholeImage.csv      — whole-image features, all wells (if run)
  numericalData/<mag>X_colonyFeatures.csv  — per-colony features, all wells (if run)
  numericalData/<mag>X_colonyAgg.csv       — well-level colony aggregates, all wells (if run)
"""

import os
import re
import pandas as pd


# Magnification suffix → objective label used in output filenames
_MAG_SUFFIX_TO_OBJ = {
    '_02': '4X',
    '_03': '10X',
    '_04': '20X',
    '_05': '40X',
}


# Identity columns to drop from each source before joining.
# Kept lowercase for case-insensitive matching.
_WI_IDENTITY  = {'plateid', 'wellid', 'processedpath'}   # whole-image CSV
_WCA_IDENTITY = {'plateid', 'wellid'}                     # well-colony-agg CSV
_COL_IDENTITY = {'plateid', 'wellid'}                     # per-colony CSV


def _drop_cols(df, lower_name_set):
    """Drop columns whose lowercase name is in lower_name_set."""
    to_drop = [c for c in df.columns if c.lower() in lower_name_set]
    return df.drop(columns=to_drop)


def _prefix_cols(df, prefix, keep=('frame',)):
    """Prefix all columns except those listed in keep."""
    return df.rename(columns={c: f'{prefix}{c}' for c in df.columns if c not in keep})


def assemble_master_csvs(plate_outdirs, drawer_map, output_root, log_fn=None):
    """Read per-plate index files and write master CSVs to output_root.

    Parameters
    ----------
    plate_outdirs : list[str]
        Paths to processedImages/ directories, one per plate that ran.
    drawer_map : dict[str, str]
        Maps plate_name → drawer_name.  If there is no drawer, pass
        plate_name as the value (drawerID == plateID).
    output_root : str
        Directory where master_*.csv files will be written.
    log_fn : callable, optional
        Called with progress/warning strings.
    """

    def log(msg):
        if log_fn:
            log_fn(msg)

    frame_rows = []
    colony_rows = []

    for outdir in plate_outdirs:
        index_path = os.path.join(outdir, 'index.csv')
        if not os.path.exists(index_path):
            log(f'  [master CSV] no index.csv in {outdir}, skipping')
            continue

        # Read all values as strings so missing paths stay as empty strings.
        index = pd.read_csv(index_path, dtype=str).fillna('')

        for _, irow in index.iterrows():
            plate_name = irow['plate']
            well_id    = irow['well']
            mag        = irow.get('mag', '')
            drawer_id  = drawer_map.get(plate_name, plate_name)

            # ── biomass (required — skip well if absent) ──────────────
            biomass_path = irow.get('biomass', '')
            if not biomass_path or not os.path.exists(biomass_path):
                log(f'  [master CSV] {plate_name}/{well_id}: no biomass CSV, skipping')
                continue

            merged = pd.read_csv(biomass_path)[['frame', 'biomass']]

            # ── whole-image features (proc_* columns) ─────────────────
            wi_path = irow.get('whole_image_feats', '')
            if wi_path and os.path.exists(wi_path):
                wi = _drop_cols(pd.read_csv(wi_path), _WI_IDENTITY)
                # whole-image features are already named proc_*, no extra prefix
                merged = merged.merge(wi, on='frame', how='outer')

            # ── well-colony aggregate features ────────────────────────
            wca_path = irow.get('well_colony_feats', '')
            if wca_path and os.path.exists(wca_path):
                wca = _drop_cols(pd.read_csv(wca_path), _WCA_IDENTITY)
                wca = _prefix_cols(wca, 'colAgg_')  # colAgg_nColonies, colAgg_meanColonyArea_um2, …
                merged = merged.merge(wca, on='frame', how='outer')

            # tag with identity columns (insert at front)
            merged.insert(0, 'drawerID', drawer_id)
            merged.insert(1, 'plateID',  plate_name)
            merged.insert(2, 'wellID',   well_id)
            merged.insert(3, 'mag',      mag)

            frame_rows.append(merged)

            # ── per-colony features ───────────────────────────────────
            cf_path = irow.get('colony_feats', '')
            if cf_path and os.path.exists(cf_path):
                cf = _drop_cols(pd.read_csv(cf_path), _COL_IDENTITY)
                cf.insert(0, 'drawerID', drawer_id)
                cf.insert(1, 'plateID',  plate_name)
                cf.insert(2, 'wellID',   well_id)
                cf.insert(3, 'mag',      mag)
                colony_rows.append(cf)

    results = {}

    if frame_rows:
        master = (pd.concat(frame_rows, ignore_index=True)
                    .sort_values(['drawerID', 'plateID', 'wellID', 'mag', 'frame'])
                    .reset_index(drop=True))
        path = os.path.join(output_root, 'master_frame_features.csv')
        master.to_csv(path, index=False)
        results['frame'] = (path, len(master))
        log(f'  Master CSV (frame):  {path}  ({len(master):,} rows, {len(master.columns)} cols)')

    if colony_rows:
        master = (pd.concat(colony_rows, ignore_index=True)
                    .sort_values(['drawerID', 'plateID', 'wellID', 'mag', 'frame'])
                    .reset_index(drop=True))
        path = os.path.join(output_root, 'master_colony_features.csv')
        master.to_csv(path, index=False)
        results['colony'] = (path, len(master))
        log(f'  Master CSV (colony): {path}  ({len(master):,} rows, {len(master.columns)} cols)')

    return results


def assemble_plate_numerical_data(processed_images_dir, index, log_fn=None):
    """Write per-magnification CSVs into <plate_dir>/numericalData/.

    Parameters
    ----------
    processed_images_dir : str
        Path to the plate's processedImages/ directory.
    index : dict
        Well-keyed dict of output file paths (as built by ProcessingWorker).
        Keys per well may include: 'biomass', 'whole_image_feats',
        'colony_feats', 'well_colony_feats'.
    log_fn : callable, optional
        Called with progress strings.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    numerical_dir = os.path.join(os.path.dirname(processed_images_dir), 'numericalData')
    os.makedirs(numerical_dir, exist_ok=True)

    # Group wells by magnification suffix (e.g. '_03')
    groups = {}
    for well_id, row in index.items():
        m = re.match(r'^[A-P]\d+(_\d+)$', well_id)
        mag_suffix = m.group(1) if m else ''
        groups.setdefault(mag_suffix, []).append((well_id, row))

    for mag_suffix, well_rows in sorted(groups.items()):
        obj_label = _MAG_SUFFIX_TO_OBJ.get(mag_suffix, mag_suffix.lstrip('_') + 'X') if mag_suffix else 'unknownX'

        # ── BF biomass ──────────────────────────────────────────────
        bf_dfs = []
        for well_id, row in well_rows:
            p = row.get('biomass', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)[['frame', 'biomass']]
                df.insert(0, 'well', well_id)
                bf_dfs.append(df)
        if bf_dfs:
            out = os.path.join(numerical_dir, f'{obj_label}_BF.csv')
            (pd.concat(bf_dfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {obj_label}_BF.csv  ({len(bf_dfs)} wells)')

        # ── Whole-image features ────────────────────────────────────
        wi_dfs = []
        for well_id, row in well_rows:
            p = row.get('whole_image_feats', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, 'well', well_id)
                wi_dfs.append(df)
        if wi_dfs:
            out = os.path.join(numerical_dir, f'{obj_label}_wholeImage.csv')
            (pd.concat(wi_dfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {obj_label}_wholeImage.csv  ({len(wi_dfs)} wells)')

        # ── Per-colony features ─────────────────────────────────────
        cf_dfs = []
        for well_id, row in well_rows:
            p = row.get('colony_feats', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, 'well', well_id)
                cf_dfs.append(df)
        if cf_dfs:
            out = os.path.join(numerical_dir, f'{obj_label}_colonyFeatures.csv')
            (pd.concat(cf_dfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {obj_label}_colonyFeatures.csv  ({len(cf_dfs)} wells)')

        # ── Well-level colony aggregates ────────────────────────────
        wca_dfs = []
        for well_id, row in well_rows:
            p = row.get('well_colony_feats', '')
            if p and os.path.exists(p):
                df = pd.read_csv(p)
                df.insert(0, 'well', well_id)
                wca_dfs.append(df)
        if wca_dfs:
            out = os.path.join(numerical_dir, f'{obj_label}_colonyAgg.csv')
            (pd.concat(wca_dfs, ignore_index=True)
               .sort_values(['well', 'frame'])
               .to_csv(out, index=False))
            log(f'  [numericalData] {obj_label}_colonyAgg.csv  ({len(wca_dfs)} wells)')
