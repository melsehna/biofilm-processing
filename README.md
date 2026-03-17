# Biofilm processing for bulk Cytation5 brightfield experiments

High-throughput biofilm phenotyping from automated brightfield timelapse microscopy. Processes 96-well plate image data through registration, segmentation, colony tracking, and feature extraction.

## Installation

```bash
git clone https://github.com/melsehna/biofilm-processing.git biofilm-processing
cd biofilm-processing
conda create -n phenotypr python=3.11 -y
conda activate phenotypr
pip install -e .
```

Dependencies (installed automatically): numpy, scipy, scikit-image, opencv-python, mahotas, pandas, matplotlib, tifffile, imageio, imageio-ffmpeg, PySide6

### Desktop shortcut

Create a clickable shortcut so you can launch the GUI without a terminal:

```bash
python scripts/install-desktop-shortcut.py
```

This works on Linux, macOS, and Windows. It detects your active conda or virtualenv and bakes the activation into the shortcut.

| Platform | What it creates |
|---|---|
| Linux | `.desktop` file on Desktop + app menu |
| macOS | `Phenotypr.app` bundle on Desktop |
| Windows | `.bat` launcher + `.lnk` shortcut on Desktop |

## What it does

Biofilms grown in multi-well plates are imaged over time using brightfield microscopy, producing per-well TIFF stacks. This pipeline automates:

1. **Preprocessing** -- Local contrast normalization removes illumination artifacts (vignetting, uneven lighting) so that downstream thresholding reflects actual biomass, not optics.
2. **Registration** -- Phase-correlation drift correction aligns frames across the timelapse, compensating for stage drift or plate movement between timepoints.
3. **Segmentation** -- Thresholding on the preprocessed image produces binary masks of biomass. Optional dust correction removes persistent bright artifacts that are not biological.
4. **Overlay generation** -- MP4 videos with cyan mask overlay on the processed frames, with optional text labels (mutant, plate, well).
5. **Colony tracking** -- Connected-component labeling at a seed frame propagates labels forward in time using distance-based assignment, linking the same colony across frames.
6. **Colony feature extraction** -- Per-colony geometry (area, circularity, eccentricity), intensity statistics, spatial features, and background intensity for every tracked colony at every frame.
7. **Whole-image feature extraction** -- Haralick texture moments, intensity statistics, and entropy extracted from each preprocessed frame as a global texture fingerprint.

## Input data

- A **root directory** containing plate subdirectories, each with per-well TIFF files
- Multi-frame stacks: `A1.tif` (single file, shape `(T, H, W)` or `(H, W, T)`)
- Single-frame series: `A1_001.tif`, `A1_002.tif`, ...
- Well IDs are parsed from filenames via regex `^[A-H]\d{1,2}` (96-well layout: rows A-H, columns 1-12)
- For plates with multiple magnifications, filenames encode the magnification step: `A1_03_1_1_Bright Field_001.tif`

## GUI

```bash
phenotypr-gui
# or: python -m multiWellAnalysis.gui.app
```

The GUI has five tabs:

| Tab | Purpose |
|---|---|
| **Setup** | Select root directory, output directory, plates, and magnifications |
| **Parameters** | Choose analyses to run, configure preprocessing, set worker count |
| **Preview** | Live preview of raw, preprocessed, and mask overlay for any plate/well/frame |
| **Conditions** | 96-well grid for labeling experimental conditions per well |
| **Run** | Start/stop processing with per-plate and per-well progress bars and a live log |

### Setup tab

1. **Browse** to the root directory containing plate folders. Plates are auto-discovered.
2. Check/uncheck plates to include or exclude them.
3. **Magnifications** are auto-detected from filenames. Check one or more magnifications to process (e.g., just 10x), or leave all checked to process everything. For plates without magnification suffixes, this section stays empty and all wells are processed.
4. Set the **output directory** where results will be written.

### Parameters tab

**Analysis** -- choose what to compute. Dependencies are enforced automatically (e.g., enabling colony features auto-enables colony tracking).

| Analysis | What it does |
|---|---|
| Biofilm biomass | Preprocessing + registration + masking + biomass curve (always on) |
| Mask overlay videos | MP4 with cyan mask overlay on processed frames |
| Whole-image texture features | Haralick moments, intensity stats, entropy per frame |
| Colony tracking | Connected-component tracking across frames |
| Colony-level feature extraction | Per-colony geometry, intensity, spatial features |

**Preprocessing parameters:**

| Parameter | Default | Description |
|---|---|---|
| Block diameter | 101 | Kernel size for local contrast normalization (must be odd). Larger values smooth more background. |
| Fixed threshold | 0.04 | Binary mask threshold on the normalized image. Lower = more sensitive. |
| Shift threshold | 50 | Maximum allowed registration shift in pixels. Frames with larger drift are skipped. |
| FFT stride | 6 | Compute registration shifts every N frames (interpolate between). Higher = faster. |
| Downsample | 4 | Downsample factor for FFT registration. Higher = faster but less precise. |
| Dust correction | on | Remove pixels that appear at t=0 but disappear later (likely dust, not biofilm). |

**Performance** -- set the number of workers for parallel operations (registration, I/O). Hard-capped at 75% of CPU cores.

**Saved outputs (advanced)** -- toggle whether to keep intermediate files on disk (registered raw stacks, processed images, binary masks). Useful for saving disk space if you only need the final CSV results.

### Conditions tab

A 96-well grid for labeling experimental conditions (e.g., mutant names, media types).

- **Click** individual wells to select them
- **Click-drag** across wells to paint a selection
- **Click row headers** (A-H) to select/deselect an entire row
- **Click column headers** (1-12) to select/deselect an entire column
- **Select All / Clear** buttons for bulk operations
- Type a condition name and click **Save condition** to assign selected wells
- Click a saved condition in the list to highlight its wells on the grid

### Configuration

Use **Save/Load configuration** buttons to save or load all settings as JSON. If the root directory already contains an `experiment_config.json`, it is auto-loaded.

## CLI / scripting

### Process a single plate

```bash
python scripts/runSinglePlate.py /path/to/plate/directory \
    -o /path/to/output \
    -m _03                   # magnification suffix (optional, default: all)
```

### Process a single well (Python)

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
    outdir='path/to/output',
    filename='A1',
    image_records=None,
    fftStride=6,
    downsample=4,
    label='mutantName  plateName-A1',  # optional text label on overlay
)
```

### Batch processing

```python
from multiWellAnalysis.processing.batch_runner import batch_run

batch_run(
    config_path='experiment_config.json',
    replicate_csv='ReplicatePositions.csv',
)
```

### Regenerate overlay videos

Regenerate overlays without rerunning the full pipeline (e.g., after adding mutant labels):

```bash
# All wells in a plate
python scripts/regenOverlays.py /path/to/plate/directory

# Specific wells
python scripts/regenOverlays.py /path/to/plate/directory --wells A1_03 B5_03

# With mutant labels from an index CSV
python scripts/regenOverlays.py /path/to/plate/directory --index /path/to/index.csv

# Custom fps
python scripts/regenOverlays.py /path/to/plate/directory --fps 6
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

## Output files

For each well (e.g., `A1`), the pipeline can produce:

| File | Description |
|---|---|
| `A1_registered_raw.tif` | Drift-corrected raw stack |
| `A1_processed.tif` | Contrast-normalized stack |
| `A1_masks.npz` | Binary segmentation masks (key: `masks`) |
| `A1_overlay.mp4` | Mask overlay video with optional text label |
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
            setup.py              # Tab 1: directory + plate + magnification selection
            parameters.py         # Tab 2: analysis, preprocessing, workers
            preview.py            # Tab 3: live image preview
            conditions.py         # Tab 4: 96-well condition assignment
            runGUI.py             # Tab 5: pipeline execution
    processing/                   # Core image analysis
        analysis_main.py          # timelapse_processing() — single-well pipeline
        preprocessing.py          # normalize_local_contrast(), mean_filter()
        registration.py           # Phase-correlation drift correction
        segmentation.py           # Binary masking + dust correction
        overlay.py                # Overlay video generation (cv2 VideoWriter)
        batch_runner.py           # Multi-plate batch processing with magnification discovery
        io_utils.py               # Threaded image I/O
        helpers.py                # Utility functions
        plotting.py               # Plate-level summary plots
        plotting_tools.py         # Diagnostic panels (peak frame, biomass curves)
        pipeline.py               # High-level Pipeline entry point
    colony/                       # Colony tracking & features
        runTrackingGUI.py         # Colony tracking
        runColonyFeatsGUI.py      # Colony feature extraction
        runColFeatsCLI.py         # CLI colony feature extraction
        colonyFeatsMicrons.py     # Per-colony feature functions
        wellAggMicrons.py         # Well-level aggregation
        segmentation.py           # Colony-level segmentation
    wholeImage/                   # Whole-image texture features
        runWholeImageGUI.py       # Whole-image feature extraction
        runWholeImage.py          # Batch CLI runner
        extractWholeImageFeats.py # Mahotas feature extraction
    intensity/                    # Per-pixel intensity features
scripts/
    install-desktop-shortcut.py   # Create desktop shortcut (Linux/macOS/Windows)
    runSinglePlate.py             # CLI: process one plate with magnification filtering
    regenOverlays.py              # CLI: regenerate overlay videos from existing data
    reimaging/                    # Reimaging dataset scripts
    training/                     # Training dataset scripts
    transposons/                  # Transposon library scripts
```

## Authors

Seh Na Mellick, Jojo Prentice, Andrew Bridges
CMU Ray and Stephanie Lane Computational Biology Department
CMU Department of Biological Sciences

## License

TBD
