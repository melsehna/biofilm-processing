import os
import re
import glob
import json
import numpy as np
import pandas as pd
import imageio.v3 as iio
from collections import defaultdict
from tqdm import tqdm
from .analysis_main import timelapseProcessing, frameIndexFromFilename
from .io_utils import readImagesInplace
from .helpers import roundOdd


def _magGroupsFromProtocol(plateDir, tifFiles):
    """Group BF files by magnification using protocol.csv step→mag mapping.

    Returns dict: {mag_label: {well: [files]}} or None if protocol unavailable.
    """
    protocolPath = os.path.join(plateDir, "protocol.csv")
    if not os.path.exists(protocolPath):
        return None

    protocol = pd.read_csv(protocolPath)
    bfReads = protocol[
        (protocol["action"] == "Imaging Read") &
        (protocol["channel"] == "Bright Field")
    ]

    if bfReads.empty or "magnification" not in bfReads.columns:
        return None

    stepToMag = {}
    for _, row in bfReads.iterrows():
        stepToMag[row["step"]] = row["magnification"]

    groups = defaultdict(lambda: defaultdict(list))
    for f in tifFiles:
        base = os.path.basename(f)
        m = re.match(r'^([A-P]\d+)_0?(\d+)_', base)
        if not m:
            continue
        well = m.group(1)
        step = int(m.group(2))
        mag = stepToMag.get(step)
        if mag is not None:
            groups[str(mag)][well].append(f)

    return dict(groups) if groups else None


def _magGroupsFromFilenames(tifFiles):
    """Group BF files by mag suffix parsed from filenames.

    Filename convention: WELL_MAGSUFFIX_..._Bright[_ ]Field_NNN.tif
    e.g. A10_02_1_1_Bright Field_001.tif → mag suffix = _02, well = A10

    Returns dict: {mag_suffix: {well: [files]}}
    """
    groups = defaultdict(lambda: defaultdict(list))
    for f in tifFiles:
        base = os.path.basename(f)
        m = re.match(r'^([A-P]\d+)(_\d+)_', base)
        if not m:
            continue
        well = m.group(1)
        magSuffix = m.group(2)
        groups[magSuffix][well].append(f)

    return dict(groups) if groups else None


def discoverMagGroups(plateDir, tifFiles):
    """Discover magnification groups, preferring protocol.csv when available."""
    groups = _magGroupsFromProtocol(plateDir, tifFiles)
    if groups:
        return groups
    return _magGroupsFromFilenames(tifFiles) or {}


def runPlate(plateDir, mutantMap, params, force=False, skipOverlay=False):
    """Run full processing on one plate directory, grouped by magnification."""
    plateName = os.path.basename(os.path.normpath(plateDir))
    processedDir = os.path.join(plateDir, "Processed_images_py")
    numericDir = os.path.join(plateDir, "Numerical_data_py")

    for d in [processedDir, numericDir]:
        os.makedirs(d, exist_ok=True)

    blockDiam    = params["blockDiam"]
    fixedThresh  = params["fixed_thresh"]
    shiftThresh  = params["shift_thresh"]
    dust         = params["dust_correction"]
    Imin         = params["Imin"]
    Imax         = params["Imax"]

    allTifs = sorted(glob.glob(os.path.join(plateDir, "*.tif")))
    bfTifs = [f for f in allTifs if 'Bright Field' in f or 'Bright_Field' in f]

    if not bfTifs:
        print(f"  No Bright Field images found in {plateDir}")
        return None

    magGroups = discoverMagGroups(plateDir, bfTifs)

    if not magGroups:
        print(f"  Could not group files by magnification in {plateDir}")
        return None

    print(f"  Found {len(magGroups)} magnification(s): {', '.join(sorted(magGroups.keys()))}")

    allDfs = []

    for magLabel, wellsDict in sorted(magGroups.items()):
        csvPath = os.path.join(numericDir, f"{magLabel}_BF_biomass.csv")

        if os.path.exists(csvPath) and not force:
            print(f"  Skipping {plateName} mag={magLabel} -- existing results found.")
            continue

        print(f"  Processing mag={magLabel} ({len(wellsDict)} wells)")

        biomassRecords = []
        timeseriesRecords = []

        for well in tqdm(sorted(wellsDict.keys()), desc=f"{plateName} {magLabel}"):
            mutant = mutantMap.get(well)
            if mutant is None or (isinstance(mutant, float) and pd.isna(mutant)):
                continue

            wellFiles = sorted(wellsDict[well], key=frameIndexFromFilename)
            if not wellFiles:
                continue

            wellLabel = f"{well}_{magLabel}"

            img0 = iio.imread(wellFiles[0])
            if img0.ndim == 3:
                stack = img0.astype(np.float64)
                nFrames = stack.shape[2]
            else:
                nFrames = len(wellFiles)
                h, w = img0.shape
                stack = np.empty((h, w, nFrames), dtype=np.float64)
                readImagesInplace(nFrames, stack, wellFiles)

            overlayLabel = f"{mutant}  {plateName}-{well}"

            masks, biomass, odMean = timelapseProcessing(
                images=stack,
                blockDiameter=blockDiam,
                ntimepoints=nFrames,
                shiftThresh=shiftThresh,
                fixedThresh=fixedThresh,
                dustCorrection=dust,
                outdir=plateDir,
                filename=wellLabel,
                imageRecords=None,
                Imin=Imin,
                Imax=Imax,
                skipOverlay=skipOverlay,
                label=overlayLabel,
            )

            biomassRecords.append((well, biomass))

            for t in range(nFrames):
                timeseriesRecords.append({
                    'plate': plateName,
                    'well': well,
                    'mag': magLabel,
                    'mutant': mutant,
                    'frame': t,
                    'biomass': biomass[t],
                    'od_mean': odMean[t] if odMean is not None else np.nan,
                })

        if biomassRecords:
            maxFrames = max(len(b) for _, b in biomassRecords)
            data = {
                well: np.pad(b, (0, maxFrames - len(b)), constant_values=np.nan)
                for well, b in biomassRecords
            }
            dfWide = pd.DataFrame(data)
            dfWide.to_csv(csvPath, index=False)
            print(f"  Wrote: {csvPath}")

        if timeseriesRecords:
            dfLong = pd.DataFrame(timeseriesRecords)
            longPath = os.path.join(numericDir, f"{magLabel}_BF_timeseries.csv")
            dfLong.to_csv(longPath, index=False)
            allDfs.append(dfLong)

    return pd.concat(allDfs, ignore_index=True) if allDfs else None


def batchRun(configPath, replicateCsv, force=False, skipOverlay=False):
    """Master batch runner. Uses experiment_config.json if present."""
    if os.path.exists(configPath):
        print(f"Using experiment config: {configPath}")
        with open(configPath, "r") as f:
            config = json.load(f)
        params = {
            "blockDiam": roundOdd(config.get("blockDiam", 101)),
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
            plates = _findPlateDirs(os.path.dirname(configPath))
    else:
        print("No experiment_config.json found -- autodetecting plates...")
        params = {
            "blockDiam": roundOdd(101),
            "fixed_thresh": 0.014,
            "shift_thresh": 50,
            "dust_correction": True,
            "Imin": None,
            "Imax": None,
        }
        plates = _findPlateDirs(os.path.dirname(configPath))

    mutantMap = pd.read_csv(replicateCsv).set_index("Header")["Replicate ID"].to_dict()

    for plateDir in plates:
        metadataPath = os.path.join(plateDir, "metadata.csv")
        metaParams = params.copy()
        if os.path.exists(metadataPath):
            print(f"Reading metadata for {plateDir}")
            metaParams = _updateParamsFromMetadata(metadataPath, metaParams)
        else:
            print(f"No metadata.csv found in {plateDir}, using defaults/global config.")

        runPlate(plateDir, mutantMap, metaParams, force=force, skipOverlay=skipOverlay)

    print("\nGenerating summary plots for all processed plates...")
    try:
        from .plotting import plotting_main
        rootDir = os.path.dirname(replicateCsv)
        plotting_main(rootDir)
        print("Summary plots generated for all plates.")
    except Exception as e:
        print(f"Plotting stage failed: {e}")

    print("\nFull pipeline complete.")


def _findPlateDirs(baseDir):
    """Find all plate directories directly under baseDir/plates."""
    plateRoot = os.path.join(baseDir, "plates")
    if not os.path.isdir(plateRoot):
        plateRoot = baseDir

    plateDirs = []
    for d in os.listdir(plateRoot):
        dirPath = os.path.join(plateRoot, d)
        if not os.path.isdir(dirPath):
            continue
        if (
            os.path.exists(os.path.join(dirPath, "metadata.csv"))
            and os.path.exists(os.path.join(dirPath, "protocol.csv"))
        ):
            plateDirs.append(dirPath)

    plateDirs = sorted(plateDirs)
    print(f"Found {len(plateDirs)} plate(s):")
    for p in plateDirs:
        print(f"   {p}")
    return plateDirs


def _updateParamsFromMetadata(metadataPath, params):
    """Parse Cytation metadata.csv for integration time, gain, and autofocus hints."""
    try:
        md = pd.read_csv(
            metadataPath,
            header=None,
            on_bad_lines="skip",
            encoding_errors="ignore",
            encoding="latin1"
        )
        metaText = " ".join(md.astype(str).fillna("").values.ravel())

        if "integration" in metaText.lower():
            val = _extractFirstNumber(metaText, "Integration time")
            if val:
                params["blockDiam"] = max(3, min(31, int(val // 5)))

        if "gain" in metaText.lower():
            val = _extractFirstNumber(metaText, "gain")
            if val:
                params["fixed_thresh"] = min(1.0, 0.2 + val / 100.0)

        if "laser autofocus" in metaText.lower():
            params["dust_correction"] = True

        print(f"  Derived params: blockDiam={params['blockDiam']} fixed_thresh={params['fixed_thresh']} dust={params['dust_correction']}")
        return params
    except Exception as e:
        print(f"  Could not parse {metadataPath}: {e}")
        return params


def _extractFirstNumber(text, keyword):
    """Find first number after keyword in metadata text."""
    m = re.search(fr"{keyword}[^0-9]*([0-9]+\.?[0-9]*)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


if __name__ == "__main__":
    batchRun(
        configPath="/home/smellick/ImageLibrary/experiment_config.json",
        replicateCsv="/home/smellick/ImageLibrary/ReplicatePositions.csv"
    )
