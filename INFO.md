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
biofilm-processing-gui                    # or: python -m multiWellAnalysis.gui.app

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
  {well}_processed.tif            fixed-fpMean(0.5) render, float32        <- DISPLAY + FEATURE INPUT
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
| **`_registered_raw.tif`** (signal) | phase-corrected, bit-depth-scaled raw, float32 `[0,1]` | **biomass** = `mean(OD·mask)` where `OD = −log10(I_t / mean(frames 0–4))` (frame-0..4 blank → batch-invariant absorbance; falls back to `Imin`/`Imax` flat-field if provided); the source for the render; (geometry comes from masks/labels) |
| **`_processed.tif`** (fixed render) | `clip(raw − localMean + 0.5, 0, 1)` — fixed background gray; the **sole** render (the per-stack adaptive fpMean was retired because it drifts batch-to-batch) | **THE feature input** (whole-image Haralick/intensity/entropy + colony intensity) **and** the `_overlay.mp4` source. |
| **`_masks.npz`** | binary segmentation | biomass; seed/footprint for colony tracking |
| **`_trackedLabels…npz`** | per-colony integer labels over time | colony geometry/intensity/spatial features |

**Rule of thumb:** *signal* (`registered_raw`) → biomass; *fixed render* (`_processed.tif`)
→ all intensity/texture features **and** the overlay. The per-stack adaptive render was
retired (it drifted batch-to-batch), so there is a single render — which is what keeps
features comparable across batches. See `ISSUES.md`.

**Known limitation (planktonic blind spot).** The local-contrast normalization is a
high-pass, so it removes the slow background *level*; features on that render (plus the
mask-restricted biomass) are blind to a uniform medium darkening from planktonic cells.
Whole-image Haralick keeps background *texture* but not *level*. A dispersal mutant thus
reads only as "less biofilm." **Biomass now uses the frame-0..4-blank OD** (`mean(OD·mask)`,
batch-invariant); the complementary **`planktonic` = `mean(OD·¬mask)`** measure and
OD-based texture features are the remaining additive steps. See `ISSUES.md` Phase 5.

### Feature extraction inputs in detail
- **Whole-image** (`runWholeImageGUI` → `extractWholeImageFeats`): reads
  `_processed.tif`. Per frame: intensity stats (mean/std/median/MAD/IQR/skew/
  kurtosis, percentiles, entropy) + 13 Haralick texture features. Columns `whole_*`.
- **Colony** (`runColonyFeatsGUI`): geometry from the tracked-label NPZ (converted to µm
  via `pxToUm` from metadata — fails loudly if missing); **intensity** read from
  `_processed.tif`. Outputs per-colony and well-aggregate CSVs.
- **Biomass** (`_biomass.csv` → `numericalData/<mag>X_BF.csv`): `mean(OD·mask)` per frame,
  OD computed from `registered_raw` against the frames 0–4 blank (batch-invariant; see
  ISSUES.md Phase 5). Also read by tracking for seed-frame detection (threshold 0.005
  still valid — OD starts ~0 and crosses at essentially the same frame).

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
- **Disk gate (per plate):** output accumulates to a *full plate* before the per-plate
  delete frees it, so before writing each plate the run estimates its output (~5× the raw
  uint16 input — dominated by the two float32 stacks) and checks the target filesystem has
  that much free plus a reserve (`max(15%, 5 GB)`). If a plate won't fit it **aborts before
  writing** (no mid-stage `ENOSPC`/corrupt partial TIFFs); already-finished plates are kept
  and master CSVs still assemble, so you can free space and resume. Fails open (warns, does
  not block) if the size or free space can't be determined. Startup also logs a one-time
  free-space warning at <20 GB.

## 12. Conventions

- **camelCase** throughout (functions, variables).
- Magnification suffixes are per-plate; always resolve via metadata.
- Masks: `.npz` key `'masks'` (bool). Colony labels: `.npz` keys `labels/frames/wasTracked`.
- Run tests with `pytest` (import smoke test only; no unit tests for pipeline logic).
