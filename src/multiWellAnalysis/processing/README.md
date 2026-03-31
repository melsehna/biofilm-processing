# processing

Core image analysis pipeline for brightfield timelapse biofilm data. Handles preprocessing, registration, segmentation, and overlay video generation.

## Modules

| Module | Purpose |
|---|---|
| `analysis_main.py` | `timelapseProcessing()` — full single-well pipeline |
| `preprocessing.py` | `normalizeLocalContrast()`, `meanFilter()` |
| `registration.py` | Two-pass in-place phase-correlation drift correction |
| `segmentation.py` | `computeMaskInplace()`, `dustCorrectInplace()` |
| `overlay.py` | `writeOverlayVideo()` — MP4 generation with mask overlay |
| `batch_runner.py` | Multi-plate batch processing with magnification discovery |
| `master_csv.py` | `assembleMasterCsvs()`, `assemblePlateNumericalData()` |
| `io_utils.py` | Threaded TIFF I/O (`readImagesInplace()`, `saveStack()`) |
| `helpers.py` | Utility functions (`roundOdd()`, `calculateStats()`) |
| `plotting.py` | Plate-level summary plots |
| `plotting_tools.py` | Diagnostic panels (peak frame, biomass curves) |
| `pipeline.py` | High-level `Pipeline` entry point |

## Pipeline stages

```
Raw .tif files
    |
[1] Scale to [0, 1] float32
    |
[2] Normalize local contrast (downsample 8x -> Gaussian blur -> upsample -> subtract)
    |  + post-normalization Gaussian blur (sigma=2)
    |
[3] Register frames (two-pass phase correlation, in-place, threaded)
    |
[4] Crop NaN borders
    |
[5] Binary mask (threshold on normalized image)
    |  + optional dust correction
    |
[6] Biomass curve (mean of masked region per frame)
    |  with OD via -log10 if Imin/Imax reference images provided
    |
[7] Save stacks (.tif), masks (.npz)
    |
[8] Overlay video (.mp4) via cv2.VideoWriter
```

## Usage

### Single well

```python
import numpy as np
import tifffile
from multiWellAnalysis.processing.analysis_main import timelapseProcessing

stack = tifffile.imread('plate/A1_03_1_1_Bright Field_001.tif')

# Ensure (H, W, T)
if stack.shape[0] < stack.shape[1]:
    stack = np.transpose(stack, (1, 2, 0))

masks, biomass, odMean = timelapseProcessing(
    images=stack.astype(np.float64),
    blockDiameter=101,
    ntimepoints=stack.shape[2],
    shiftThresh=50,
    fixedThresh=0.014,
    dustCorrection=True,
    outdir='/path/to/output',
    filename='A1_03',
    imageRecords=None,
    fftStride=6,
    downsample=4,
    label='lipA  Plate1-A1',  # optional text on overlay
)
```

### Batch (multiple plates)

```python
from multiWellAnalysis.processing.batch_runner import batchRun

batchRun(
    configPath='experiment_config.json',
    replicateCsv='ReplicatePositions.csv',
)
```

### Single plate with magnification filtering

```python
from multiWellAnalysis.processing.batch_runner import runPlate
from multiWellAnalysis.processing.helpers import roundOdd

params = {
    'blockDiam': roundOdd(101),
    'fixed_thresh': 0.014,
    'shift_thresh': 50,
    'dust_correction': True,
    'Imin': None,
    'Imax': None,
}

mutantMap = {'A1': 'lipA', 'B2': 'WT', ...}

df = runPlate('/path/to/plate', mutantMap, params)
```

### Overlay video only

```python
from multiWellAnalysis.processing.overlay import writeOverlayVideo

# displayStack: (H, W, T) float32 in [0, 1]
# masks: (H, W, T) bool
writeOverlayVideo(
    displayStack, masks, 'output/A1_overlay.mp4',
    fps=2,
    label='lipA  Plate1-A1',
)
```

### Preprocessing only

```python
from multiWellAnalysis.processing.preprocessing import (
    normalizeLocalContrast,
    normalizeLocalContrastOutput,
)

# For segmentation input (background - image):
norm = normalizeLocalContrast(frame, blockDiameter=101)

# For visualization (image - background + midpoint, clipped to [0, 1]):
vis = normalizeLocalContrastOutput(stack, blockDiameter=101, fpMean=0.5)
```

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `blockDiameter` | 101 | Local contrast kernel size (odd). Larger = smoother background estimate. |
| `fixedThresh` | 0.014 (CLI) / 0.04 (GUI) | Mask threshold on normalized image. |
| `shiftThresh` | 50 | Max registration shift (px). Larger shifts are rejected. |
| `fftStride` | 6 | Compute shifts every N frames. |
| `downsample` | 4 | Downsample factor for FFT. |
| `dustCorrection` | True | Kill pixels present at t=0 that vanish later. |
| `skipOverlay` | False | Skip MP4 generation (saves ~30% runtime). |

## Magnification handling

`batch_runner` automatically groups wells by magnification:

1. **From `protocol.csv`** (preferred): maps imaging steps to magnification labels
2. **From filenames**: parses suffix `WELL_MAGSUFFIX_...` (e.g., `_01`=4x, `_02`=4x, `_03`=10x, `_04`=20x)

Each magnification group is processed independently, producing separate output CSVs (`{mag}_BF_biomass.csv`, `{mag}_BF_timeseries.csv`).

## Output files

Per well:

| File | Description |
|---|---|
| `{well}_processed.tif` | Normalized + blurred + registered + cropped |
| `{well}_registered_raw.tif` | Raw registered + cropped (for OD calculation) |
| `{well}_masks.npz` | Binary masks (key: `masks`) |
| `{well}_overlay.mp4` | Cyan mask overlay video |

Per magnification group:

| File | Description |
|---|---|
| `{mag}_BF_biomass.csv` | Wide format: columns = wells, rows = timepoints |
| `{mag}_BF_timeseries.csv` | Long format: plate, well, mag, mutant, frame, biomass, od_mean |
