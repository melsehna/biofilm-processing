# INFO.md — using the biofilm-processing pipeline

Practical reference for running the pipeline and understanding how data flow through
it: commands, parameters, file types, datatypes, and **which representation of the
data is used for which stage** (signal vs masks vs display vs feature input). Reflects
**v0.5.0**. For deeper internals see `CLAUDE.md`; for known artifacts see `ISSUES.md`.

---

## 1. What it does

High-throughput biofilm phenotyping for brightfield timelapse microscopy of multi-well
plates (Cytation5). Per-well TIFF stacks → registration → segmentation → colony
tracking → feature extraction, plus a PySide6 GUI and CLI tools.

## 2. Install & commands

```bash
pip install -e .                 # editable install (also defines the GUI entry point)

# GUI
phenotypr-gui                    # or: python -m multiWellAnalysis.gui.app

# CLI — single plate
python scripts/runSinglePlate.py /path/to/plate -o /path/to/output -m _03
#   -m/--mag <suffix>     restrict to one magnification suffix (e.g. _03)
#   --block-diam 101      local-contrast kernel (odd)
#   --fixed-thresh 0.014  mask threshold (CLI default 0.014; GUI default 0.04)
#   --skip-overlay        skip MP4 generation;  --force  reprocess existing

# Regenerate overlays from existing processed data
python scripts/regenOverlays.py /path/to/plate --wells A1_03 --fps 6
python scripts/regenOverlaysFromIndex.py index.csv --outdir /mnt/... --fps 8 --workers 40
sbatch scripts/regenOverlays_reimaging.sbatch     # SLURM (40 CPU, 200G, 24h)

# UMAP (optional extra: pip install -e .[umap])
python scripts/runUmap.py ...

pytest                           # smoke test (module imports)
```

## 3. Pipeline stages (in order)

1. **Load** per-frame raw TIFFs → stack `(H, W, T)`.
2. **Bit-depth scale** → float32 in `[0, 1]` (`_toBitDepthScaled`).
3. **Local-contrast normalization** (high-pass) → `normBlur` stack (for registration/mask).
4. **Registration** — phase-correlation, **frame-to-frame** (`fftStride=1`); warp with
   `BORDER_CONSTANT`/NaN.
5. **Crop** NaN borders (`cropStack`) to the region valid in every frame.
6. **Threshold** the normalized stack → binary masks; **dust correction**.
7. **Biomass** curve.
8. **Render + save** display stacks, masks, overlay; optionally tracking + features.

## 4. Data: file types & datatypes

**Input (raw):** per-frame TIFFs, **uint16**, Cytation5. Filename pattern
`{well}{suffix}_1_1_Bright Field_NNN.tif` (e.g. `B5_03_1_1_Bright Field_001.tif`).
- `{well}` = `A1`–`H12` (96-well; 6/12/24/48/96/384 supported).
- `{suffix}` (`_02/_03/_04/_05`) is an acquisition-slot index — **magnification is
  resolved from TIFF metadata, NOT the suffix** (the same suffix maps to different
  objectives on different microscopes). See §7.
- Acquisition parameters (objective, px→µm) live in the `ImageDescription` XML tag.

**Internal axis order:** `(H, W, T)`; stacks transposed on load if needed.

## 5. Output files (under `<outputRoot>/<plate>/`)

```
processedImages/
  index.csv                       per-well paths + pxToUm/objective metadata
  run_params.json                 serialized params + _pipelineVersion stamp (§8)
  {well}_registered_raw.tif       phase-corrected raw, float32 (H,W,T)   <- SIGNAL
  {well}_processed.tif            adaptive-fpMean display render, float32 <- DISPLAY ONLY
  {well}_processed_fpHalf.tif     fixed-fpMean(0.5) render, float32       <- FEATURE INPUT
  {well}_masks.npz                key 'masks', bool (H,W,T)
  {well}_trackedLabels_allFrames_trackingVec_v3.npz   int labels + arrays
  {well}_overlay.mp4
  {well}_biomass.csv
  {well}_wholeImageFeatures.csv
  {well}_perColonyFeatures.csv
  {well}_wellColonyFeatures.csv
numericalData/<mag>X_*.csv         per-magnification aggregates
master_frame_features.csv          run-level, one row per (plate, well, frame)
master_colony_features.csv         run-level, one row per (plate, well, frame, colony)
```

## 6. Which representation feeds which stage (important)

The pipeline keeps **three** versions of each well's stack, used for different purposes:

| Representation | What it is | Used for |
|---|---|---|
| **`_registered_raw.tif`** (signal) | phase-corrected, bit-depth-scaled raw, float32 `[0,1]` | **biomass** = `mean((1−raw)·mask)`; the source for both render renderings; (geometry comes from masks/labels, not this) |
| **`_processed_fpHalf.tif`** (fixed render) | `clip(raw − localMean + 0.5, 0, 1)` — fixed background gray | **THE feature input**: whole-image (Haralick, intensity, entropy) **and** colony intensity features. **Required** — workers error if missing. |
| **`_processed.tif`** (adaptive render) | same but `fpMean = 0.5·(max+min)` per stack | **display/overlay ONLY** — retired as a feature input (its per-stack offset drifts batch-to-batch). Overlay MP4 is built from it. |
| **`_masks.npz`** | binary segmentation | biomass; seed/footprint for colony tracking |
| **`_trackedLabels…npz`** | per-colony integer labels over time | colony geometry/intensity/spatial features |

**Rule of thumb:** *signal* (`registered_raw`) → biomass; *fixed render* (`fpHalf`) →
all intensity/texture features; *adaptive render* (`processed`) → human-facing display
only. This separation (added in v0.5.0) is what keeps features comparable across
batches — see `ISSUES.md`.

### Feature extraction inputs in detail
- **Whole-image** (`runWholeImageGUI` → `extractWholeImageFeats`): reads
  `_processed_fpHalf.tif`. Per frame: intensity stats (mean/std/median/MAD/IQR/skew/
  kurtosis, percentiles, entropy) + 13 Haralick texture features. Columns `whole_*`.
- **Colony** (`runColonyFeatsGUI`): geometry from the tracked-label NPZ (converted to µm
  via `pxToUm` from metadata — fails loudly if missing); **intensity** read from
  `_processed_fpHalf.tif`. Outputs per-colony and well-aggregate CSVs.
- **Biomass** (`_biomass.csv`): from `registered_raw` × mask; also read by tracking for
  seed-frame detection.

## 7. Magnification detection (metadata-driven)

`probePlateMeta(plateDir)` reads the Cytation TIFF header XML
(`<ObjectiveSize>`, `<ImageWidthMicrons>`, `<PixelWidth>`) → `objective` and
`pxToUm = ImageWidthMicrons / PixelWidth`. Stored per-plate in `state.plateMeta` and
`experiment_config.json`. Colony µm features require `pxToUm` and fail loudly without
it (never silently default).

## 8. Provenance / version stamping (v0.5.0)

Every `run_params.json` and `experiment_config.json` carries a `_pipelineVersion`
record (package version + git branch/commit/dirty) from `buildRecord()`. On resume,
the GUI warns if existing outputs came from a different pipeline version. The stamp is
ignored by the resume param-match (a version bump records provenance without forcing
reprocessing). This closes the "which code version produced these features?" gap.

## 9. Key parameters (current defaults)

| Parameter | Default | Purpose |
|---|---|---|
| `blockDiam` | 101 | local-contrast kernel (odd) |
| `fixedThresh` | 0.04 (GUI) / 0.014 (CLI) | binary mask threshold on normalized image |
| `shiftThresh` | 50 | max accepted registration shift (px) — rejects spurious phase-corr peaks |
| `fftStride` | **1** | register every frame (frame-to-frame). >1 strides keyframes (legacy) |
| `downsample` | 4 | decimation factor for phase correlation |
| `dustCorrection` | True | remove pixels ON in frame 0 but OFF later |
| `saveFpHalf` | **True** | write the fixed render — required for features; leave on |
| `minColonyAreaPx` | 200 | min connected-component size (**pixels**) to label a colony |
| `propRadiusPx` | 25 | colony label-propagation radius (**pixels**) |
| `workers` | 8 | parallel workers (hard-capped at 75% of cores) |

Per-magnification overrides via `magParams` (e.g. `{'_03': {'fixedThresh': 0.02}}`) —
merged over globals by suffix at run time. Can override `blockDiam`, `fixedThresh`,
`dustCorrection`, `minColonyAreaPx`, `propRadiusPx`.

Feature stages are opt-in: `wholeImageFeats`, `colonyTracking`, `colonyFeats` (GUI
checkboxes / state flags).

## 10. Resume & config

- **Resume:** before processing a plate, `run_params.json` is loaded; if the tracked
  params match, wells with an existing `_processed.tif` are skipped and `index.csv` is
  reused for downstream stages. Extra/unknown keys in the saved file are ignored.
- **Config:** the GUI persists all state to `experiment_config.json` (auto-loaded from
  the root dir), including `plateMeta`, conditions, and the per-mag overrides.

## 11. Network mounts / staging

- Output often targets NFS/SMB (`/mnt/...`). Overlays are written to a local temp file
  then moved (avoids corrupt partials). Use `shutil.copyfile` (not `copy2`) to mounts.
- Raw plate dirs are typically **read-only** — never write sidecars into them; per-plate
  metadata persists in `experiment_config.json`.
- **NAS mirror** (`nasMirrorEnabled`/`nasMirrorDir`): process to a fast local `outputDir`,
  then rsync each finished plate to the NAS and delete the local copy (batched transfers
  beat per-file SMB writes). Auto-staging under `~/biofilm-staging-<ts>/` when `outputDir`
  is empty or on a NAS.

## 12. Conventions

- **camelCase** throughout (functions, variables).
- Magnification suffixes are per-plate; always resolve via metadata.
- Masks: `.npz` key `'masks'` (bool). Colony labels: `.npz` keys `labels/frames/wasTracked`.
- Run tests with `pytest` (import smoke test only; no unit tests for pipeline logic).
