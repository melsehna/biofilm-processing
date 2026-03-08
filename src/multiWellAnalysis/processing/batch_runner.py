import os
import glob
import json
import numpy as np
import pandas as pd
import imageio.v3 as iio
from tqdm import tqdm
from skimage.color import gray2rgb
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from .analysis_main import timelapse_processing
from .io_utils import read_images_inplace


def make_overlay(image, mask):
    image = rescale_intensity(image, out_range=(0, 1))
    rgb = gray2rgb(image)
    rgb[..., 1] = np.maximum(rgb[..., 1], mask)
    rgb[..., 2] = np.maximum(rgb[..., 2], mask)
    return img_as_ubyte(np.clip(rgb, 0, 1))


# ----------------------------------------------------------
# Process one plate
# ----------------------------------------------------------
def run_plate(plate_dir, mutant_map, params, force=False):
    """
    Run full processing on one plate directory, unless already processed (unless force=True).
    """
    plate_name = os.path.basename(os.path.normpath(plate_dir))
    processed_dir = os.path.join(plate_dir, "Processed_images_py")
    numeric_dir = os.path.join(plate_dir, "Numerical_data_py")
    csv_path = os.path.join(numeric_dir, f"{plate_name}_BF_biomass.csv")

    # --- Skip if already processed ---
    if os.path.exists(csv_path) and not force:
        print(f"⏭️  Skipping {plate_name} — existing results found.")
        return None

    for d in [processed_dir, numeric_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"📁 Output directories:\n  Images → {processed_dir}\n  Data   → {numeric_dir}")

    block_diam   = params["blockDiam"]
    fixed_thresh = params["fixed_thresh"]
    sigma        = params["sigma"]
    shift_thresh = params["shift_thresh"]
    dust         = params["dust_correction"]
    Imin         = params["Imin"]
    Imax         = params["Imax"]

    biomass_records = []

    for well, mutant in tqdm(mutant_map.items(), desc=f"Processing {plate_name}"):
        pattern = os.path.join(plate_dir, f"{well}_*_Bright_Field_*.tif")
        well_files = sorted(glob.glob(pattern))
        if not well_files:
            continue

        img0 = iio.imread(well_files[0])
        if img0.ndim == 3:
            stack = img0.astype(np.float64)
            nframes = stack.shape[2]
        else:
            nframes = len(well_files)
            h, w = img0.shape
            stack = np.empty((h, w, nframes), dtype=np.float64)
            read_images_inplace(nframes, stack, well_files)

        shifts, crop, masks, biomass = timelapse_processing(
            stack, block_diam, nframes,
            shift_thresh, fixed_thresh, sigma,
            dust, processed_dir, well, Imin, Imax
        )

        # Save overlay TIFFs
        for t in range(nframes):
            overlay = make_overlay(stack[..., t], masks[..., t])
            out_name = os.path.join(processed_dir, f"{well}_mask_t{t+1}.tif")
            iio.imwrite(out_name, overlay)

        biomass_records.append((well, biomass))

    if not biomass_records:
        return None

    max_frames = max(len(b) for _, b in biomass_records)
    data = {
        well: np.pad(b, (0, max_frames - len(b)), constant_values=np.nan)
        for well, b in biomass_records
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"✅ Wrote: {csv_path}")
    return df


def batch_run(config_path, replicate_csv, force=False):
    """
    Master batch runner.
    Uses experiment_config.json if present, else autodetects plates.
    Skips already processed plates unless force=True.
    """
    if os.path.exists(config_path):
        print(f"📖 Using experiment config: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        params = {
            "blockDiam": int(round(config.get("blockDiam", 15))),
            "fixed_thresh": float(config.get("fixed_thresh", 0.5)),
            "sigma": config.get("sigma", 2),
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
            print("⚠️ No plate directories listed — autodetecting instead.")
            plates = _find_plate_dirs(os.path.dirname(config_path))
    else:
        print("⚠️ No experiment_config.json found — autodetecting plates...")
        params = {
            "blockDiam": 15,
            "fixed_thresh": 0.5,
            "sigma": 2,
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
            print(f"🧾 Reading metadata for {plate_dir}")
            meta_params = _update_params_from_metadata(metadata_path, meta_params)
        else:
            print(f"ℹ️ No metadata.csv found in {plate_dir}, using defaults/global config.")

        # 🔍 Detect wells in this plate based on available Bright_Field TIFFs
        tiff_files = sorted(glob.glob(os.path.join(plate_dir, "*_Bright_Field_*.tif")))
        wells = sorted({os.path.basename(f).split('_')[0] for f in tiff_files})
        if wells:
            print(f"   🧫 Wells detected ({len(wells)}): {', '.join(wells)}")
        else:
            print(f"   ⚠️ No Bright_Field images found in {plate_dir}!")

        run_plate(plate_dir, mutant_map, meta_params, force=force)

    # ------------------------------------------------------------
    # ✅ FINAL STAGE: automatic plotting and summary aggregation
    # ------------------------------------------------------------
    print("\n📊 Generating summary plots for all processed plates...")
    try:
        from .plotting import plotting_main
        root_dir = os.path.dirname(replicate_csv)
        plotting_main(root_dir)
        print("✅ Summary plots generated for all plates.")
    except Exception as e:
        print(f"⚠️ Plotting stage failed: {e}")

    print("\n Full pipeline complete — all analyses and plots generated.")


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
    print(f"🧫 Found {len(plate_dirs)} plate(s):")
    for p in plate_dirs:
        print(f"   • {p}")
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

        # Integration time → affects block diameter
        if "integration" in meta_text.lower():
            val = _extract_first_number(meta_text, "Integration time")
            if val:
                params["blockDiam"] = max(3, min(31, int(val // 5)))

        # Gain → affects threshold
        if "gain" in meta_text.lower():
            val = _extract_first_number(meta_text, "gain")
            if val:
                params["fixed_thresh"] = min(1.0, 0.2 + val / 100.0)

        # Laser autofocus mention → enable dust correction
        if "laser autofocus" in meta_text.lower():
            params["dust_correction"] = True

        print(f"🧩 Derived params: blockDiam={params['blockDiam']} fixed_thresh={params['fixed_thresh']} dust={params['dust_correction']}")
        return params
    except Exception as e:
        print(f"⚠️ Could not parse {metadata_path}: {e}")
        return params


def _extract_first_number(text, keyword):
    """Utility: find first number after keyword in metadata text."""
    import re
    m = re.search(fr"{keyword}[^0-9]*([0-9]+\.?[0-9]*)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


if __name__ == "__main__":
    pipeline(
        config_path="/home/smellick/ImageLibrary/experiment_config.json",
        replicate_csv="/home/smellick/ImageLibrary/ReplicatePositions.csv"
    )
