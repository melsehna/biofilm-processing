# biofilm-processing

[![CI](https://github.com/melsehna/biofilm-processing/actions/workflows/ci.yml/badge.svg)](https://github.com/melsehna/biofilm-processing/actions/workflows/ci.yml)

**Biofilm phenotyping from brightfield timelapse microscopy of 96-well plates.**

biofilm-processing takes the per-well image stacks produced by a Cytation5 microscope and turns them into clean biomass curves, mask overlay videos, tracked colony labels, and per-colony / whole-image feature tables — all driven from a desktop GUI.

> **New here? Skip straight to [Quick start](#quick-start).** Installation takes ~10 minutes and you don't need any prior Python or command-line experience.

---

## Table of contents

- [Quick start](#quick-start)
  - [1. Install Miniconda (one-time)](#1-install-miniconda-one-time)
  - [2. Download biofilm-processing](#2-download-biofilm-processing)
  - [3. Install biofilm-processing](#3-install-biofilm-processing)
  - [4. Make a desktop shortcut](#4-make-a-desktop-shortcut-optional-but-recommended)
- [Using the GUI](#using-the-gui)
- [What gets produced](#what-gets-produced)
- [Updating biofilm-processing](#updating-biofilm-processing)
- [Troubleshooting](#troubleshooting)
- [Advanced: command line + Python API](#advanced-command-line--python-api)
- [Authors & license](#authors--license)

---

## Quick start

### 1. Install Miniconda (one-time)

Miniconda gives you an isolated Python environment so biofilm-processing's dependencies don't conflict with anything else on your computer. **If you already have Anaconda or Miniconda installed, skip to step 2.**

- **Windows / macOS / Linux:** download and run the installer from <https://www.anaconda.com/download/success> (pick "Miniconda" — it's smaller than Anaconda and works exactly the same way).
- Accept all defaults during install.
- After install, open the **"Anaconda Prompt"** (Windows) or your normal **Terminal** (macOS / Linux). All commands below should be typed there.

### 2. Download biofilm-processing

Pick **one** option:

**Option A — easiest, no Git needed:**
1. Go to <https://github.com/melsehna/biofilm-processing>
2. Click the green **"Code"** button → **"Download ZIP"**
3. Unzip it somewhere you'll remember (e.g. your Desktop or Documents folder)
4. Rename the unzipped folder to `biofilm-processing` if it isn't already

**Option B — if you have Git:**
```bash
git clone https://github.com/melsehna/biofilm-processing.git
```

### 3. Install biofilm-processing

In your terminal, navigate **into** the folder you just downloaded and run the install commands. Copy these one block at a time:

```bash
cd biofilm-processing
conda create -n biofilm-processing python=3.11 -y
conda activate biofilm-processing
pip install -e .
```

What these do:
- `cd biofilm-processing` — moves into the project folder. If you put it on your Desktop, you may need `cd Desktop/biofilm-processing` first.
- `conda create -n biofilm-processing python=3.11 -y` — makes a new Python environment named `biofilm-processing`.
- `conda activate biofilm-processing` — switches into that environment. **You'll need to do this every time you open a fresh terminal**, unless you use the desktop shortcut (step 4).
- `pip install -e .` — installs biofilm-processing and all its dependencies (numpy, scipy, scikit-image, opencv, mahotas, PySide6, etc.).

Once that's finished, launch the app:

```bash
biofilm-processing-gui
```

> **Windows users — `mahotas` install error?** `mahotas` is a C library and needs a compiler. Easiest fix: before running `pip install -e .`, run `conda install -c conda-forge mahotas -y`. If you'd rather use the Microsoft compiler, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and check the "Desktop development with C++" box during setup.

> **Reproducing published results?** The command above installs *current* dependency versions, which may differ from the ones a paper was produced with. To reproduce the exact validated stack, create the environment from the pinned manifest instead:
> ```bash
> conda env create -f environment.yml
> conda activate biofilm-processing
> pip install -e . --no-deps
> ```
> See [Reproducibility & containers](#reproducibility--containers) for details.

### 4. Make a desktop shortcut (optional but recommended)

If you don't want to open a terminal every time you launch biofilm-processing:

```bash
python scripts/installDesktopShortcut.py
```

This detects your conda environment and creates a clickable icon on your desktop:

| Platform | What it makes |
|---|---|
| Linux   | `.desktop` file on Desktop + entry in app menu |
| macOS   | `biofilm-processing.app` bundle on Desktop |
| Windows | `.bat` launcher + `.lnk` shortcut on Desktop |

Double-click the icon to launch — no terminal needed.

---

## Using the GUI

The app has six tabs. You'll usually move through them left to right.

| Tab | What you do |
|---|---|
| **Setup** | Point at a folder of plates, pick which plates and magnifications to process, choose where output goes |
| **Parameters** | Choose what to compute (biomass, overlays, tracking, features) and tune preprocessing settings |
| **Preview** | See live previews of preprocessing and segmentation for any plate / well / frame |
| **Conditions** | Label the 96-well grid with experimental conditions (mutants, media, etc.) |
| **Test Well** | Run the full pipeline on a single well for a quick sanity check before committing to a batch |
| **Run** | Hit go. Watch the progress bar and live log as the batch runs |

### Setup tab

1. Click **Browse** and pick the **root directory** that contains your plate folders. Plate folders are auto-discovered.
2. Tick which plates to include.
3. **Magnifications** are auto-detected from each plate. Tick one or more (e.g. just 10x), or leave them all checked.
4. Set the **output directory** where results should be written.

### Parameters tab

**Analyses** — pick what you want to compute. Dependencies are enforced automatically (e.g. enabling colony features auto-enables colony tracking).

| Analysis | Produces |
|---|---|
| Biofilm biomass | Preprocessing + registration + binary masks + biomass curve (always on) |
| Mask overlay videos | MP4 with cyan mask overlay on processed frames |
| Whole-image texture features | Haralick moments, intensity stats, entropy per frame |
| Colony tracking | Connected-component tracking across frames |
| Colony-level features | Per-colony geometry, intensity, spatial features |

**Preprocessing parameters:**

| Parameter | Default | Description |
|---|---|---|
| Block diameter | 101 | Kernel size for local contrast normalization (must be odd). Larger values smooth more background. |
| Fixed threshold | 0.04 | Binary mask threshold on the normalized image. Lower = more sensitive. |
| Dust correction | on | Removes pixels that appear at t=0 but disappear later (likely dust, not biofilm). |

**Performance** — number of parallel workers. Capped automatically at 75% of CPU cores so the rest of your computer stays responsive.

**Saved outputs (advanced)** — toggle whether to keep intermediate files (registered raw stacks, processed images, binary masks). Turn off to save disk space if you only care about the final CSVs.

### Preview tab

Two rows of side-by-side panels for visual QC:

- **Top row** (live from current parameters): raw image, preprocessed image, mask overlay
- **Bottom row** (colony view): colony segmentation, tracked labels (loaded from a previous run if available), colony overlay on the raw image

Use the dropdowns and slider to flip through plates, wells, and frames. Click **Refresh** after a run to reload results.

### Conditions tab

A 96-well grid for labeling wells (e.g. mutant names, media types).

- Click individual wells to select them
- Click-and-drag to paint
- Click row headers (A–H) or column headers (1–12) to select whole rows / columns
- **Select All** / **Clear** for bulk operations
- Type a name and click **Save condition** to assign it to the selected wells
- Click an existing condition in the list to highlight its wells

### Test Well tab

Run the full pipeline on a single well end-to-end with a progress bar. Use this to sanity-check parameters before kicking off a multi-plate batch.

### Run tab

Click **Start**, watch the progress bar and live log. **Stop** safely cancels mid-batch.

### Saving your setup

Use **Save / Load configuration** to dump all settings to a JSON file. If your root directory already contains an `experiment_config.json`, the app loads it automatically when you open that folder.

---

## What gets produced

For each well (e.g. `A1`), the pipeline can write:

| File | Description |
|---|---|
| `A1_registered_raw.tif` | Drift-corrected raw stack |
| `A1_processed.tif` | Contrast-normalized stack |
| `A1_masks.npz` | Binary segmentation masks |
| `A1_overlay.mp4` | Mask overlay video (with optional text label) |
| `A1_trackedLabels_allFrames_*.npz` | Tracked colony label stack |
| `A1_colonyFeatures_*.csv` | One row per colony per frame |
| `A1_wellColonyFeatures_*.csv` | One row per frame (well-level colony aggregates) |
| `A1_wholeImage_*.csv` | One row per frame (whole-image texture features) |

Per-plate aggregates land in `numericalData/`:

| File | Description |
|---|---|
| `{mag}X_BF.csv` | Biomass timeseries for all wells at that magnification |
| `{mag}X_wholeImage.csv` | Whole-image features for all wells |
| `{mag}X_colonyFeatures.csv` | Per-colony features for all wells |
| `{mag}X_colonyAgg.csv` | Well-level colony aggregates |

Run-level master CSVs (across all plates) land at the output root:

| File | Description |
|---|---|
| `master_frame_features.csv` | One row per (drawer, plate, well, mag, frame) |
| `master_colony_features.csv` | One row per (drawer, plate, well, mag, frame, colony) |

---

## Updating biofilm-processing

When a new version is available:

```bash
conda activate biofilm-processing
cd /path/to/biofilm-processing
git pull           # or: re-download the ZIP and replace the folder
pip install -e .   # picks up any new dependencies
```

Your desktop shortcut keeps working — no need to re-make it.

---

## Troubleshooting

**`biofilm-processing-gui: command not found`** — you forgot to activate the environment. Run `conda activate biofilm-processing` first.

**The GUI won't launch / closes immediately** — open a terminal, activate the env, and run `biofilm-processing-gui` from there. Any error message will print to the terminal so you can see what went wrong.

**`mahotas` install fails on Windows** — see the note in [step 3](#3-install-biofilm-processing) about installing it via conda or installing the C++ Build Tools.

**Output files are 0 KB on a network drive** — biofilm-processing writes overlays to a local temp file first and then moves them, but make sure your network mount supports normal file moves. Avoid `cp -p` style metadata-preserving copies on SMB / NFS mounts that don't support them.

**Magnification not detected** — biofilm-processing reads magnification from the TIFF metadata embedded by the Cytation. If the files have been re-saved or stripped of metadata, detection will fail loudly rather than guess. Re-export from the Cytation if needed.

**Colony tracking looks fragmented** — try increasing **min colony area** (Parameters tab) or **propagation radius**. biofilm-processing already uses a frozen-snapshot strategy during stalled growth to avoid edge-noise fragmentation, but very thin or low-contrast biofilms may still need tuning.

---

## Advanced: command line + Python API

For automated pipelines, SLURM submissions, or scripting from Python.

### Process a single well from Python

```python
import numpy as np
import tifffile
from multiWellAnalysis.processing.analysis_main import timelapseProcessing

stack = tifffile.imread('path/to/A1.tif').astype(np.float64)

# Ensure (H, W, T) axis order
if stack.shape[0] < stack.shape[1]:
    stack = np.transpose(stack, (1, 2, 0))

masks, biomass, odMean = timelapseProcessing(
    images=stack,
    blockDiameter=101,
    ntimepoints=stack.shape[2],
    shiftThresh=50,
    fixedThresh=0.04,
    dustCorrection=True,
    outdir='path/to/output',
    filename='A1',
    imageRecords=None,
    fftStride=1,            # register every frame (frame-to-frame); default since v0.5.0
    downsample=4,
    label='mutantName  plateName-A1',
)
```

### Batch processing (multiple plates)

Use the headless runner — the same pipeline the GUI Run tab drives:

```bash
biofilm-processing-run experiment_config.json --output-dir /path/to/output --workers 40

# or entirely from flags, with no config file
biofilm-processing-run --plates /path/to/plateA /path/to/plateB \
    -o /path/to/output --mag _03 --workers 40 \
    --whole-image --colony-tracking --colony-feats
```

See `scripts/examples/` for a SLURM submission and a per-dataset run template.

### Regenerate overlay videos

Without rerunning the pipeline (e.g. after adding mutant labels):

```bash
python scripts/regenOverlays.py /path/to/plate/directory                       # all wells
python scripts/regenOverlays.py /path/to/plate/directory --wells A1_03 B5_03   # specific wells
python scripts/regenOverlays.py /path/to/plate/directory --index mutants.csv   # with labels
python scripts/regenOverlays.py /path/to/plate/directory --fps 6               # custom fps
```

### Colony tracking (single well)

```python
from multiWellAnalysis.colony.runTrackingGUI import trackAndSave

raw_stack = tifffile.imread('output/processedImages/A1_registered_raw.tif')
mask_data = np.load('output/processedImages/A1_masks.npz')

npz_path = trackAndSave(
    raw_stack, mask_data['masks'],
    outdir='output/processedImages',
    plateId='Plate_1',
    wellId='A1',
    biomass=biomass,
)
```

### Colony feature extraction (single well)

```python
from multiWellAnalysis.colony.runColonyFeatsGUI import extractAndSave

data = np.load(npz_path)
colony_df, well_df = extractAndSave(
    rawStack=raw_stack,
    labelStack=data['labels'],
    frames=data['frames'],
    plateId='Plate_1',
    wellId='A1',
    wasTracked=bool(data['wasTracked']),
    trackedLabelsPath=npz_path,
    rawPath='output/processedImages/A1_registered_raw.tif',
    outdir='output/processedImages',
)
```

### Whole-image texture features (single well)

```python
from multiWellAnalysis.wholeImage.runWholeImageGUI import extractWholeImageFeatures

status = extractWholeImageFeatures(
    processedPath='output/processedImages/A1_processed.tif',
    plateId='Plate_1',
    wellId='A1',
    outDir='output/processedImages',
)
```

### Batch CLI tools

```bash
# Colony feature extraction across plates
python -m multiWellAnalysis.colony.runColFeatsCLI \
    --index processed_index.csv \
    --outRoot /path/to/output \
    --nProc 16

# Whole-image features across plates
python -m multiWellAnalysis.wholeImage.runWholeImage \
    --index processed_index.csv \
    --outdir /path/to/output \
    --workers 32
```

### Project structure

```
src/multiWellAnalysis/
    gui/               PySide6 GUI (app.py is the entry point)
    cli/               Headless entry points (biofilm-processing-run / -test-well)
    processing/        Core pipeline: preprocessing, registration, segmentation, overlay, master CSVs
    colony/            Colony tracking + per-colony feature extraction
    wholeImage/        Whole-image (Haralick) texture features
    analysis/          UMAP embedding of the run-level feature table
scripts/
    installDesktopShortcut.py    Desktop shortcut creator (Linux/macOS/Windows)
    regenOverlays.py             CLI: regenerate overlay videos
    regenOverlaysFromIndex.py    CLI: bulk regen across many plates
```

---

## Reproducibility & containers

Reproducibility is layered — pick the tier that matches your need:

| Tier | Artifact | Guarantees | Use when |
|---|---|---|---|
| Ranges | `pyproject.toml` | Installs and runs | Day-to-day use, `pip install -e .` |
| Exact env | `environment.yml` | Same Python package versions the results used | Reproducing a paper's analysis |
| Full system | `Dockerfile` (→ image) | Same versions **and** OS libraries | Archival, HPC, sharing with reviewers |

**Why the exact env matters.** The numerically-sensitive libraries (`numpy`, `scipy`, `scikit-image`, `mahotas`) can change segmentation counts and feature values across versions — `skimage.measure.label`, morphology, `distance_transform_edt`, and the Haralick implementation are not guaranteed stable release-to-release. `environment.yml` pins the exact versions (from conda-forge, the channel the numerics were validated on) so results are stable. It is pinned to Python 3.9.23, the interpreter the published results were produced on; the general install above uses a newer Python for convenience.

**Why a container, not just a lockfile.** A `pip freeze` captures Python packages but not the system libraries `opencv` and Qt depend on (`libGL`, `libEGL`, `ffmpeg`) — the usual source of "works on my machine, `ImportError: libGL.so.1` on the cluster." The `Dockerfile` bakes those in and builds on a micromamba/conda-forge base so the in-container numerics match `environment.yml`. It targets the **headless CLI** (`biofilm-processing-run` / `-test-well`) — the GUI is not containerized (Qt GUIs need X forwarding; the reproducible science is the headless pipeline).

```bash
# Build a multi-arch image (x86 HPC + Apple Silicon) and push
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/melsehna/biofilm-processing:0.5.0 --push .

# Run locally
docker run --rm -v /path/to/data:/data -v /path/to/output:/out \
  ghcr.io/melsehna/biofilm-processing:0.5.0 \
  biofilm-processing-run --plates /data/plateA -o /out --mag _03 --workers 8

# Run on an HPC cluster via Apptainer/Singularity (no root, no Docker daemon)
apptainer pull biofilm.sif docker://ghcr.io/melsehna/biofilm-processing:0.5.0
apptainer exec --bind /path/to/data,/path/to/output biofilm.sif \
  biofilm-processing-run --plates /path/to/data/plateA -o /path/to/output --mag _03 --workers 40
```

For an archival, fully-solved lock (all transitive deps, per-platform), generate `conda-lock.yml` from `environment.yml` with [`conda-lock`](https://github.com/conda/conda-lock) and commit it alongside — then a release image tagged to a Zenodo DOI gives a citable frozen artifact.

---

## Authors & license

**Authors:** Seh Na Mellick, Jojo Prentice, Andrew Bridges
CMU Ray and Stephanie Lane Computational Biology Department · CMU Department of Biological Sciences

**License:** [MIT](LICENSE)
