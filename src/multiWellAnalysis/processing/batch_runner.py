import os
import re
import glob
import json
import numpy as np
import pandas as pd
import imageio.v3 as iio
from collections import defaultdict
from tqdm import tqdm
from .analysis_main import timelapse_processing, frame_index_from_filename
from .io_utils import read_images_inplace
from .helpers import round_odd


# ----------------------------------------------------------
# Magnification discovery
# ----------------------------------------------------------
def _mag_groups_from_protocol(plate_dir, tif_files):
    """Group BF files by magnification using protocol.csv step→mag mapping.

    Returns dict: {mag_label: {well: [files]}} or None if protocol unavailable.
    """
    protocol_path = os.path.join(plate_dir, "protocol.csv")
    if not os.path.exists(protocol_path):
        return None

    protocol = pd.read_csv(protocol_path)
    bf_reads = protocol[
        (protocol["action"] == "Imaging Read") &
        (protocol["channel"] == "Bright Field")
    ]

    if bf_reads.empty or "magnification" not in bf_reads.columns:
        return None

    # Build step → magnification mapping
    step_to_mag = {}
    for _, row in bf_reads.iterrows():
        step_to_mag[row["step"]] = row["magnification"]

    # Group files by magnification
    groups = defaultdict(lambda: defaultdict(list))
    for f in tif_files:
        base = os.path.basename(f)
        # Parse step from filename: WELL_STEP_..._Bright[_ ]Field_NNN.tif
        m = re.match(r'^([A-H]\d+)_0?(\d+)_', base)
        if not m:
            continue
        well = m.group(1)
        step = int(m.group(2))
        mag = step_to_mag.get(step)
        if mag is not None:
            groups[str(mag)][well].append(f)

    return dict(groups) if groups else None


def _mag_groups_from_filenames(tif_files):
    """Group BF files by mag suffix parsed from filenames.

    Filename convention: WELL_MAGSUFFIX_..._Bright[_ ]Field_NNN.tif
    e.g. A10_02_1_1_Bright Field_001.tif → mag suffix = _02, well = A10

    Returns dict: {mag_suffix: {well: [files]}}
    """
    groups = defaultdict(lambda: defaultdict(list))
    for f in tif_files:
        base = os.path.basename(f)
        m = re.match(r'^([A-H]\d+)(_\d+)_', base)
        if not m:
            continue
        well = m.group(1)
        mag_suffix = m.group(2)
        groups[mag_suffix][well].append(f)

    return dict(groups) if groups else None


def discover_mag_groups(plate_dir, tif_files):
    """Discover magnification groups, preferring protocol.csv when available."""
    groups = _mag_groups_from_protocol(plate_dir, tif_files)
    if groups:
        return groups
    return _mag_groups_from_filenames(tif_files) or {}


# ----------------------------------------------------------
# Process one plate
# ----------------------------------------------------------
def run_plate(plate_dir, mutant_map, params, force=False, skip_overlay=False):
    """
    Run full processing on one plate directory, grouped by magnification.
    Skips already-processed plates unless force=True.
    """
    plate_name = os.path.basename(os.path.normpath(plate_dir))
    processed_dir = os.path.join(plate_dir, "Processed_images_py")
    numeric_dir = os.path.join(plate_dir, "Numerical_data_py")

    for d in [processed_dir, numeric_dir]:
        os.makedirs(d, exist_ok=True)

    block_diam   = params["blockDiam"]
    fixed_thresh = params["fixed_thresh"]
    shift_thresh = params["shift_thresh"]
    dust         = params["dust_correction"]
    Imin         = params["Imin"]
    Imax         = params["Imax"]

    # Discover all BF tifs (handles both "Bright Field" and "Bright_Field")
    all_tifs = sorted(glob.glob(os.path.join(plate_dir, "*.tif")))
    bf_tifs = [f for f in all_tifs if 'Bright Field' in f or 'Bright_Field' in f]

    if not bf_tifs:
        print(f"  No Bright Field images found in {plate_dir}")
        return None

    mag_groups = discover_mag_groups(plate_dir, bf_tifs)

    if not mag_groups:
        print(f"  Could not group files by magnification in {plate_dir}")
        return None

    print(f"  Found {len(mag_groups)} magnification(s): {', '.join(sorted(mag_groups.keys()))}")

    all_dfs = []

    for mag_label, wells_dict in sorted(mag_groups.items()):
        csv_path = os.path.join(numeric_dir, f"{mag_label}_BF_biomass.csv")

        if os.path.exists(csv_path) and not force:
            print(f"  Skipping {plate_name} mag={mag_label} -- existing results found.")
            continue

        print(f"  Processing mag={mag_label} ({len(wells_dict)} wells)")

        biomass_records = []
        timeseries_records = []

        for well in tqdm(sorted(wells_dict.keys()), desc=f"{plate_name} {mag_label}"):
            mutant = mutant_map.get(well)
            if mutant is None or (isinstance(mutant, float) and pd.isna(mutant)):
                continue

            well_files = sorted(wells_dict[well], key=frame_index_from_filename)
            if not well_files:
                continue

            well_label = f"{well}_{mag_label}"

            img0 = iio.imread(well_files[0])
            if img0.ndim == 3:
                stack = img0.astype(np.float64)
                nframes = stack.shape[2]
            else:
                nframes = len(well_files)
                h, w = img0.shape
                stack = np.empty((h, w, nframes), dtype=np.float64)
                read_images_inplace(nframes, stack, well_files)

            masks, biomass, odMean = timelapse_processing(
                images=stack,
                block_diameter=block_diam,
                ntimepoints=nframes,
                shift_thresh=shift_thresh,
                fixed_thresh=fixed_thresh,
                dust_correction=dust,
                outdir=plate_dir,
                filename=well_label,
                image_records=None,
                Imin=Imin,
                Imax=Imax,
                skip_overlay=skip_overlay,
            )

            biomass_records.append((well, biomass))

            for t in range(nframes):
                timeseries_records.append({
                    'plate': plate_name,
                    'well': well,
                    'mag': mag_label,
                    'mutant': mutant,
                    'frame': t,
                    'biomass': biomass[t],
                    'od_mean': odMean[t] if odMean is not None else np.nan,
                })

        # Write per-mag wide-format CSV (columns = wells, matching Julia output)
        if biomass_records:
            max_frames = max(len(b) for _, b in biomass_records)
            data = {
                well: np.pad(b, (0, max_frames - len(b)), constant_values=np.nan)
                for well, b in biomass_records
            }
            df_wide = pd.DataFrame(data)
            df_wide.to_csv(csv_path, index=False)
            print(f"  Wrote: {csv_path}")

        # Write per-mag long-format timeseries
        if timeseries_records:
            df_long = pd.DataFrame(timeseries_records)
            long_path = os.path.join(numeric_dir, f"{mag_label}_BF_timeseries.csv")
            df_long.to_csv(long_path, index=False)
            all_dfs.append(df_long)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


def batch_run(config_path, replicate_csv, force=False, skip_overlay=False):
    """
    Master batch runner.
    Uses experiment_config.json if present, else autodetects plates.
    Skips already processed plates unless force=True.
    """
    if os.path.exists(config_path):
        print(f"Using experiment config: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        params = {
            "blockDiam": round_odd(config.get("blockDiam", 101)),
            "fixed_thresh": float(config.get("fixed_thresh", 0.014)),
            "shift_thresh": config.get("shift_thresh", 50),
            "dust_correction": str(config.get("dust_correction", "True")).lower() in ["true", "1"],
            "Imin": None,
            "Imax": None,
        }
        if config.get("Imin_path"):
            params["Imin"] = iio.imread(config["Imin_path"]).astype(np.float64)
        if config.get("Imax_path"):
            params["Imax"] = iio.imread(config["Imax_path"]).astype(np.float64)
        plates = config.get("images_directory", [])
        if not plates:
            print("No plate directories listed -- autodetecting instead.")
            plates = _find_plate_dirs(os.path.dirname(config_path))
    else:
        print("No experiment_config.json found -- autodetecting plates...")
        params = {
            "blockDiam": round_odd(101),
            "fixed_thresh": 0.014,
            "shift_thresh": 50,
            "dust_correction": True,
            "Imin": None,
            "Imax": None,
        }
        plates = _find_plate_dirs(os.path.dirname(config_path))

    mutant_map = pd.read_csv(replicate_csv).set_index("Header")["Replicate ID"].to_dict()

    for plate_dir in plates:
        metadata_path = os.path.join(plate_dir, "metadata.csv")
        meta_params = params.copy()
        if os.path.exists(metadata_path):
            print(f"Reading metadata for {plate_dir}")
            meta_params = _update_params_from_metadata(metadata_path, meta_params)
        else:
            print(f"No metadata.csv found in {plate_dir}, using defaults/global config.")

        run_plate(plate_dir, mutant_map, meta_params, force=force, skip_overlay=skip_overlay)

    # Final stage: automatic plotting and summary aggregation
    print("\nGenerating summary plots for all processed plates...")
    try:
        from .plotting import plotting_main
        root_dir = os.path.dirname(replicate_csv)
        plotting_main(root_dir)
        print("Summary plots generated for all plates.")
    except Exception as e:
        print(f"Plotting stage failed: {e}")

    print("\nFull pipeline complete.")


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _find_plate_dirs(base_dir):
    """Find all directories directly under base_dir/plates that contain metadata.csv and protocol.csv."""
    plate_root = os.path.join(base_dir, "plates")
    if not os.path.isdir(plate_root):
        plate_root = base_dir

    plate_dirs = []
    for d in os.listdir(plate_root):
        dir_path = os.path.join(plate_root, d)
        if not os.path.isdir(dir_path):
            continue
        if (
            os.path.exists(os.path.join(dir_path, "metadata.csv"))
            and os.path.exists(os.path.join(dir_path, "protocol.csv"))
        ):
            plate_dirs.append(dir_path)

    plate_dirs = sorted(plate_dirs)
    print(f"Found {len(plate_dirs)} plate(s):")
    for p in plate_dirs:
        print(f"   {p}")
    return plate_dirs


def _update_params_from_metadata(metadata_path, params):
    """
    Simple heuristics: parse Cytation metadata.csv for integration time, gain, and autofocus hints.
    Adjusts blockDiam, fixed_thresh, and dust_correction accordingly.
    """
    try:
        md = pd.read_csv(
            metadata_path,
            header=None,
            on_bad_lines="skip",
            encoding_errors="ignore",
            encoding="latin1"
        )
        meta_text = " ".join(md.astype(str).fillna("").values.ravel())

        # Integration time -> affects block diameter
        if "integration" in meta_text.lower():
            val = _extract_first_number(meta_text, "Integration time")
            if val:
                params["blockDiam"] = max(3, min(31, int(val // 5)))

        # Gain -> affects threshold
        if "gain" in meta_text.lower():
            val = _extract_first_number(meta_text, "gain")
            if val:
                params["fixed_thresh"] = min(1.0, 0.2 + val / 100.0)

        # Laser autofocus mention -> enable dust correction
        if "laser autofocus" in meta_text.lower():
            params["dust_correction"] = True

        print(f"  Derived params: blockDiam={params['blockDiam']} fixed_thresh={params['fixed_thresh']} dust={params['dust_correction']}")
        return params
    except Exception as e:
        print(f"  Could not parse {metadata_path}: {e}")
        return params


def _extract_first_number(text, keyword):
    """Utility: find first number after keyword in metadata text."""
    m = re.search(fr"{keyword}[^0-9]*([0-9]+\.?[0-9]*)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


if __name__ == "__main__":
    batch_run(
        config_path="/home/smellick/ImageLibrary/experiment_config.json",
        replicate_csv="/home/smellick/ImageLibrary/ReplicatePositions.csv"
    )
