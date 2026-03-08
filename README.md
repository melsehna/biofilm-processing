# Phenotypr

High-throughput biofilm phenotyping from automated brightfield timelapse microscopy. Processes 96-well plate image data through registration, segmentation, colony tracking, and feature extraction.

## Background

Biofilms grown in multi-well plates are imaged over time using brightfield microscopy, producing per-well TIFF stacks. Phenotypr automates the analysis pipeline:

1. **Preprocessing** — Local contrast normalization removes illumination artifacts (vignetting, uneven lighting) so that downstream thresholding reflects actual biomass, not optics.

2. **Registration** — Phase-correlation drift correction aligns frames across the timelapse, compensating for stage drift or plate movement between timepoints.

3. **Segmentation** — Thresholding on the preprocessed, blurred image produces binary masks of biomass. Optional dust correction removes persistent bright artifacts that are not biological.

4. **Colony tracking** — Connected-component labeling at a seed frame propagates labels forward in time using distance-based assignment. This links the same colony across frames, enabling growth-rate and morphology measurements at the single-colony level.

5. **Colony feature extraction** — Per-colony geometry (area, circularity, eccentricity), intensity statistics, spatial features (centroid offset, nearest-neighbor distances), and background intensity are computed for every tracked colony at every frame. Well-level aggregates summarize colony populations per well.

6. **Whole-image feature extraction** — Haralick texture moments (via Mahotas), intensity statistics, and entropy are extracted from each preprocessed frame as a global texture fingerprint of the well.

### Data conventions

- **Input**: A root directory containing plate subdirectories, each with per-well TIFF files.
  - Multi-frame stacks: `A1.tif` (single file, shape `(T, H, W)` or `(H, W, T)`)
  - Single-frame series: `A1_001.tif`, `A1_002.tif`, ...
- **Well IDs**: Parsed from filenames via regex `^[A-H]\d{1,2}` (96-well layout: rows A-H, columns 1-12).
- **Output**: All results are written to a user-specified output directory, mirroring the plate folder structure. The original data is never modified.

## Installation

```bash
# 1. Clone the repository
git clone <repo-url> biofilm-processing
cd biofilm-processing

# 2. Create and activate a conda environment
conda create -n phenotypr python=3.11 -y
conda activate phenotypr

# 3. Install the package (pulls all dependencies)
pip install -e .
```

### Dependencies

Installed automatically by `pip install -e .`:

numpy, scipy, scikit-image, mahotas, pandas, matplotlib, tifffile, imageio, imageio-ffmpeg, PySide6

## Usage

### GUI

```bash
phenotypr-gui
```

Or equivalently:

```bash
python -m multiWellAnalysis.gui.app
```

The GUI has five tabs:

| Tab | Purpose |
|---|---|
| **Setup** | Select root directory (input data), output directory, and which plates to process |
| **Parameters** | Configure preprocessing (block diameter, threshold, FFT stride, downsample, dust correction), choose outputs to save, enable feature extraction |
| **Preview** | Live preview of raw, preprocessed, and mask overlay for any plate/well/frame. Shows active parameter values in titles. |
| **Conditions** | 96-well grid for labeling experimental conditions per well |
| **Run** | Start/stop processing with per-plate and per-well progress bars and a live log |

#### Output toggles

In the Parameters tab, you choose which outputs to save and which analyses to run:

- **Registered raw stacks** (.tif) — drift-corrected raw images
- **Processed images** (.tif) — contrast-normalized images
- **Binary masks** (.npz) — segmentation masks
- **Mask overlay videos** (.mp4) — visualization of segmentation over time

#### Feature extraction toggles

- **Whole-image texture features** — requires processed stacks (auto-enabled)
- **Colony tracking** — requires registered raw stacks and masks (auto-enabled)
- **Colony-level feature extraction** — requires colony tracking (auto-enabled)

Dependencies between toggles are enforced automatically. For example, enabling colony features will also enable colony tracking and the registered raw stacks it depends on.

#### Configuration

Use the **Save/Load configuration** buttons at the bottom of the window to save or load all settings (parameters, selected plates, output directory, conditions) as a JSON file. The save dialog lets you choose the file location.

If the root directory already contains an `experiment_config.json`, it is auto-loaded when you browse to that directory.

### CLI / scripting

The processing modules are standalone and can be used without the GUI.

#### Preprocessing + segmentation (single well)

```python
import numpy as np
import tifffile
from multiWellAnalysis.processing.analysis_main import timelapse_processing

stack = tifffile.imread('path/to/A1.tif').astype(np.float64)

# Ensure (H, W, T) axis order
if stack.shape[0] < stack.shape[1]:
    stack = np.transpose(stack, (1, 2, 0))

masks, biomass, od_mean = timelapse_processing(
    images=stack,
    block_diameter=101,
    ntimepoints=stack.shape[2],
    shift_thresh=50,
    fixed_thresh=0.04,
    dust_correction=True,
    outdir='path/to/output',     # creates processedImages/ here
    filename='A1',
    image_records=None,
    fftStride=6,
    downsample=4,
)
```

#### Colony tracking (single well)

```python
from multiWellAnalysis.colony.runTrackingGUI import trackAndSave

raw_stack = tifffile.imread('output/processedImages/A1_registered_raw.tif')
mask_data = np.load('output/processedImages/A1_masks.npz')
mask_stack = mask_data['masks']

npz_path = trackAndSave(
    raw_stack, mask_stack,
    outdir='output/processedImages',
    plateId='Plate_1',
    wellId='A1',
    biomass=biomass,   # from timelapse_processing
)
```

#### Colony feature extraction (single well)

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

#### Whole-image texture features (single well)

```python
from multiWellAnalysis.wholeImage.runWholeImageGUI import extractWholeImageFeatures

status = extractWholeImageFeatures(
    processedPath='output/processedImages/A1_processed.tif',
    plateId='Plate_1',
    wellId='A1',
    outDir='output/processedImages',
)
```

#### Batch processing (CLI, legacy)

The original batch runner processes all plates listed in an experiment config:

```python
from multiWellAnalysis.processing.batch_runner import batch_run

batch_run(
    config_path='path/to/experiment_config.json',
    replicate_csv='path/to/ReplicatePositions.csv',
)
```

There are also standalone CLI scripts for colony tracking and feature extraction at scale:

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

## Output files

For each well (e.g., `A1`), the pipeline can produce:

| File | Description |
|---|---|
| `A1_registered_raw.tif` | Drift-corrected raw stack |
| `A1_processed.tif` | Contrast-normalized stack |
| `A1_masks.npz` | Binary segmentation masks (key: `masks`) |
| `A1_overlay.mp4` | Mask overlay video |
| `A1_trackedLabels_allFrames_*.npz` | Colony label stack with tracking metadata |
| `A1_colonyFeatures_*.csv` | Per-colony features (one row per colony per frame) |
| `A1_wellColonyFeatures_*.csv` | Well-level colony aggregates (one row per frame) |
| `A1_wholeImage_*.csv` | Whole-image texture features (one row per frame) |

## Project structure

```
src/multiWellAnalysis/
    gui/                          # PySide6 GUI
        app.py                    # Main window, entry point
        state.py                  # Centralized state (AppState)
        tabs/
            setup.py              # Tab 1: input/output directory selection
            parameters.py         # Tab 2: processing parameters
            preview.py            # Tab 3: live image preview
            conditions.py         # Tab 4: 96-well condition assignment
            runGUI.py             # Tab 5: pipeline execution (active)
            run.py                # Tab 5: original version (unused)
    processing/                   # Core image analysis (GUI-independent)
        analysis_main.py          # timelapse_processing()
        preprocessing.py          # normalize_local_contrast(), mean_filter()
        registration.py           # Phase-correlation drift correction
        segmentation.py           # Binary masking + dust correction
        batch_runner.py           # CLI batch processing
    colony/                       # Colony tracking & features
        runTrackingGUI.py         # GUI-adapted colony tracking
        runColonyFeatsGUI.py      # GUI-adapted colony feature extraction
        runTrackingMpTraining.py  # Original batch tracking script
        runColonyFeatsTrackedMP.py # Original batch feature script
        runColFeatsCLI.py         # CLI colony feature extraction
        colonyFeatsMicrons.py     # Per-colony feature functions
        wellAggMicrons.py         # Well-level aggregation
        segmentation.py           # Colony-level segmentation
    wholeImage/                   # Whole-image texture features
        runWholeImageGUI.py       # GUI-adapted whole-image features
        runWholeImage.py          # Original batch script
        extractWholeImageFeats.py # Mahotas feature extraction
    intensity/                    # Per-pixel intensity features
```

## License

TBD
