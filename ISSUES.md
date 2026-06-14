# ISSUES.md — Cross-Session Batch Artifacts

Audit of `src/` (image processing + feature extraction) for sources of **batch
artifacts**: why data collected in a later session does not look like / does not
align with data collected earlier, even for biologically equivalent samples.

**Root theme:** the pipeline has **no cross-session photometric anchor**. There is
no flat-field / illumination correction step. The Beer-Lambert OD path exists
(`processing/analysis_main.py:155-159`) but the GUI never passes `Imin`/`Imax`, so
processing always falls into the `1.0 - rawCropped` branch (`analysis_main.py:161`).
Consequently every absolute-intensity quantity is free to drift with lamp age,
exposure, gain, and focus between sessions. Layered on top of that, the *rendering
provenance* feeding feature extraction is **inconsistent** (adaptive vs. fixed vs.
raw fallback), which is the single most likely reason newer plates don't line up
with older ones.

Issues are grouped into **(A) photometric / rendering drift** (the dominant cause)
and **(B) thresholds & metadata that silently retune per session**.

Severity legend: **CRITICAL** (systematic, affects most intensity/texture features),
**HIGH**, **MEDIUM** (conditional or structural), **LOW**.

---

## HEADLINE STATUS (2026-06-08) — read this first

The investigation below ranged across many candidate causes (photometric anchor,
saturation, thresholds). **The dominant, confirmed cause is narrower than the
original "Root theme" above suggests:** it is **processing-version drift in the
`_processed.tif` rendering** (Issues 2/3), i.e. different pipeline versions rendered
the display stack differently, and intensity/texture features (esp. `whole_haralick_*`)
read that rendering.

**Proof (Phase 3, see below):** rendering the clean-deletion and reimaging batches
from the same `registered_raw` through one fixed-fpMean (`fpHalf`) render + the same
extraction collapses the WT↔WT haralick batch effect from **z ≈ 50 → z ≈ 2**. The
old atlas render compressed `haralick_12` against a ceiling (std ~0.01), and the
analysis-side StandardScaler amplified any cross-batch delta into ~50σ. This is
**fixable processing-version drift, not irreducible acquisition optics** (which
contradicts the earlier `biofilm-analysis/cleanDeletions_hand/ISSUES.md` conclusion).

**Status:**
- ✅ **Version stamping** implemented (`buildRecord` → `run_params.json` +
  `experiment_config.json`; resume warns on version mismatch). Prevents *future*
  untraceable drift.
- ✅ **The render-consistency fix** implemented: the adaptive render is **retired** —
  `_processed.tif` is now the **sole fixed-fpMean(0.5) render** (no separate `_fpHalf`
  file, no `saveFpHalf` flag); features read it directly; frame-to-frame registration +
  NaN crop landed; biomass is now frame-0..4-blank OD. Changes feature values → the
  atlas must be reprocessed.
- ⬜ **Reprocess** atlas + clean-del (+ training) through the pinned render — the step
  that actually closes the effect (the validation run on the matched panel confirmed
  convergence; the full atlas reprocess + scaler refit is the remaining compute).
- ⏸️ **De-prioritized** (investigated, shown *not* dominant): photometric anchor
  (Issue 1), saturation (Issue 8), relative thresholds (Issues 5/6).

Outstanding work is tracked in **`TODO.md`**.

---

## Mechanisms — what each contributor is, what it computes, how it becomes a batch effect

The *why* behind the inventory: for each contributor, what it is, what it is
computing, and how an acquisition/version difference turns into a **feature**
difference for *identical biology*. Issue numbers cross-reference the A/B/C inventory
below; `(TODO)` items are detailed in the 2026-04-30 processing audit in `TODO.md`.

### Rendering & texture

**fpMean — adaptive vs fixed `fpHalf` (Issues 2/3).**
- *What:* the processed image is `clip(img − localMean(img) + fpMean, 0, 1)`. The
  `img − localMean` term is a high-pass residual that carries the biology (centered
  near 0); `fpMean` is one additive constant setting the background gray level — it has
  no biological meaning.
- *Computing:* adaptive `fpMean = 0.5·(max+min)` of each well's own stack (drifts
  well-to-well / batch-to-batch); fixed `fpHalf = 0.5` always (pinned).
- *Effect:* the biology term is identical between the two; only the constant differs,
  but features read the *rendered* image, so it leaks in three ways: (1) intensity-location
  features (`mean`, `median`, percentiles) shift by ~fpMean; (2) it places the image in a
  different uint8 band → different Haralick quantization; (3) the `clip` makes it
  *nonlinear* — a low/high fpMean clips more of the residual at 0/1, an asymmetric per-well
  distortion. NOTE: the historical old↔new gap is larger than adaptive-vs-fpHalf — the
  *old atlas render added ~0 offset* (near-zero-centered, ~[−0.05, 0.21]); that
  render-formula change is the dominant version drift.

**GLCM gray-levels keyed to `max(img)` (`extractWholeImageFeats.py:62`).**
- *What:* Haralick features summarize a Gray-Level Co-occurrence Matrix (how often value
  *i* is adjacent to *j*). `mahotas.features.haralick(img)` is called with no `levels=`,
  so mahotas sizes the GLCM to `0..max(img)`.
- *Computing:* the GLCM dimension/resolution tracks the image's max gray value.
- *Effect:* a render placing the image in a narrow low band (old atlas → uint8 max ≈ 53)
  → small, coarse GLCM → texture features saturate with tiny variance — **the std ≈ 0.01
  "ceiling compression" that let the StandardScaler amplify a modest delta into ~50σ**.
  Same biology, different band → different texture. This is the feature-extraction-level
  root of the render sensitivity.

**`img_as_ubyte` (`extractWholeImageFeats.py:38`).**
- *What:* fixed `[0,1]→[0,255]` map (×255, round) — not adaptive.
- *Computing:* quantizes to 256 levels.
- *Effect:* harmless alone, but it is the *conduit* — where fpMean/render places the image
  in [0,1] decides where it lands in [0,255], and thus the GLCM.

**No flat-field / photometric anchor (Issue 1).**
- *What:* an absence — nothing corrects lamp/exposure/gain; the Beer-Lambert OD path
  exists (`analysis_main.py:155-159`) but is never fed `Imin`/`Imax`.
- *Computing:* raw values carry whatever the optics produced.
- *Effect:* exposure/gain is **multiplicative** — a dimmer/brighter session scales every
  pixel *and* the residual amplitude → intensity and texture drift. fpHalf (additive)
  cannot fix it; only dividing by background or OD would.

### Registration & preprocessing

**Registration `BORDER_REFLECT` (TODO A).**
- *What:* the empty edge created when a frame is shifted to align it is filled by mirroring
  interior pixels instead of NaN; this also defeats `cropStack` (which only trims NaN
  borders → no-ops).
- *Computing:* fabricates biofilm-like content at the drifted edges.
- *Effect:* fake texture/biomass enters mask/biomass/texture at edges, proportional to how
  much each well drifted → drift-dependent batch effect.

**`fftStride` + silently dropped shifts (TODO C/F).**
- *What:* a real phase-correlation shift is computed only every `fftStride`-th frame
  (default 6) and reused between; any shift > `shiftThresh` (50px) is discarded with no log.
- *Computing:* 5/6 frames inherit a stale shift; large real drifts are ignored.
- *Effect:* residual mis-registration smears edges (Haralick is edge/sharpness-sensitive)
  and computes whole-image stats over a slightly shifting field of view.

**`blockDiameter` + per-mag `magParams` overrides.**
- *What:* the local-contrast kernel size; can be overridden per magnification.
- *Computing:* sets the spatial scale of the high-pass `localMean`.
- *Effect:* if defaults/overrides differ across versions or magnifications, the whole
  normalization (hence render, threshold, texture) differs.

### Segmentation & thresholds

**Fixed mask threshold (Issue 5).**
- *What:* `mask = normalized > fixedThresh` (0.04).
- *Computing:* decides which pixels are biofilm.
- *Effect:* the residual amplitude depends on contrast/illumination, so a fixed cut catches
  a different boundary across batches → mask size → biomass, colony area/count, all geometry
  shift.

**Pixel-unit segmentation/tracking (`minColonyAreaPx=200`, `propRadiusPx=25`).**
- *What:* drop colonies < 200 **px**, link within 25 **px** — decisions made in pixels,
  before the µm conversion (`runTrackingGUI.py:29-30`, `segmentation.py:36`).
- *Computing:* filters/links colonies by pixel size.
- *Effect:* at different magnifications/pixel sizes, 200px = a different *physical* area →
  a different set of colonies survives → colony count, density, and every aggregate shift.

**Absolute biomass thresholds + growth filter (Issue 6).**
- *What:* seed frame and well inclusion gated at biomass ≥ 0.005; biomass =
  `mean((1−raw)·mask)`.
- *Computing:* selects which frames are tracked and which wells enter the dataset.
- *Effect:* biomass scale is render/threshold/exposure-dependent, so the absolute cut admits
  a *different population* of wells/frames across batches — a **composition** batch effect
  (which data exists), invisible in per-feature comparisons.

**Dust correction frame-0 assumption (TODO D).**
- *What:* a pixel masked at t=0 and ever off later is zeroed across all frames.
- *Computing:* removes persistent specks present at the start.
- *Effect:* misses dust arriving mid-run, and erases pre-existing biofilm if imaging starts
  after formation — both depend on acquisition timing.

### Metadata & environment

**`pxToUm` (Issue 7).**
- *What:* pixel→micron factor from TIFF metadata; geometry features × it.
- *Computing:* scales `*_um` features.
- *Effect:* if objective/scope/metadata differs, µm features shift proportionally while
  pixel-space features don't (a telltale signature).

**Library-version drift.**
- *What:* mahotas / skimage / numpy / opencv / scipy versions at processing time (current:
  mahotas 1.4.18, skimage 0.24.0, numpy 2.0.2, cv2 4.12.0, scipy 1.13.1).
- *Computing:* the *same code* can compute slightly different haralick/regionprops/entropy
  across library versions.
- *Effect:* silent value drift across processing dates. The version stamp records the git
  commit but **not** these — extend `buildRecord()` to capture key dependency versions.

**Bit-depth scaling heuristic (Issue 4).**
- *What:* float inputs scaled by an `amax>1.5` heuristic instead of declared bit depth.
- *Computing:* infers 8/16-bit from the observed max.
- *Effect:* a float stack whose true range is 16-bit but maxes ≤255 is over-scaled 256×.
  Rare (the uint16 path guards it) but catastrophic when triggered.

### Acquisition (raw data — not fixable downstream)

**Sensor saturation / clipping (Issue 8).** Bright low-biomass wells rail at the sensor
ceiling in early frames; clipped values are unrecoverable, and a flat-topped field starves
the local-contrast residual.

**Cadence / timelapse length + camera restart (Issue 9).** Different frame counts and a
mid-run restart misalign frame-indexed (`_t<frame>`) features across campaigns.

---

## A. Photometric & rendering drift

### Issue 1 — No cross-session photometric normalization (no flat-field) — **HIGH (umbrella)**

There is no flat-field or illumination-correction step anywhere in the pipeline.

- `processing/analysis_main.py:155-161` — OD/Beer-Lambert branch requires `Imin`/`Imax`
  reference images; the GUI run path never supplies them, so biomass is always
  `np.nanmean((1.0 - rawCropped) * masks, axis=(0, 1))` — an **absolute** quantity.
- The local-contrast normalization (`preprocessing.py:normalizeLocalContrast`,
  `blurred − img`) is a high-pass: it removes slow illumination **gradients** but
  **not** the overall brightness/exposure level.

**Mechanism:** any change in lamp output, exposure, or gain between sessions flows
straight into every absolute-intensity feature and into biomass. The high-pass
protects *shape/texture geometry* but not *intensity level*.

**Affected outputs:** biomass curve, all colony/background intensity features,
whole-image intensity stats — i.e. most of `master_frame_features.csv` and
`master_colony_features.csv`, hence the UMAP.

**Detection:** plot `bgMeanIntensity` (a raw-illumination proxy) for
biologically-equivalent control wells across the two eras. A clean level shift with
unchanged spread ⇒ photometric drift.

**Remediation direction:** add a per-session background-normalization anchor (e.g.
normalize each frame's background mode/percentile to a fixed reference), or capture
and apply `Imin`/`Imax` flat-field references so the OD path is used.

---

### Issue 2 — Adaptive `fpMean` bakes each stack's own dynamic range into `_processed.tif` — **CRITICAL**

`processing/analysis_main.py:171`
```python
fpMean = 0.5 * (np.nanmax(rawCropped) + np.nanmin(rawCropped))
```

This midpoint is computed **per-well, per-session** from the stack's own min/max,
then added into the saved `_processed.tif` and the overlay video
(`analysis_main.py:172-186`). Two wells with identical biology but different
illumination/dynamic range render at different gray levels.

**Mechanism:** the additive offset has no biological meaning — it is purely a
display-centering choice — but it enters every intensity-based feature computed off
`_processed.tif`. `img_as_ubyte` (see Issue 3) then quantizes those shifted values,
so Haralick graycomatrix bins shift too.

**Affected outputs:** `_processed.tif`, `_overlay.mp4`, and any feature read from the
adaptive rendering.

**Mitigation present:** `_processed_fpHalf.tif` uses fixed `fpMean = 0.5`
(`analysis_main.py:199-210`), eliminating the rendering-centering drift. But it is
only written when `saveFpHalf=True`, and feature workers only use it when the file
exists (see Issue 3).

**Cross-references:** CLAUDE.md "fpMean policy (2026-05-23)";
`JULIA_REFERENCE_COMPARISON.md` in microTyper-Vision.

---

### Issue 3 — Feature inputs read from *inconsistent* renderings depending on what is on disk — **CRITICAL (most likely "old vs new" smoking gun)**

Feature workers pick their input file by existence check, not by a single guaranteed
product:

- Whole-image: `gui/tabs/run.py:238-239`
  ```python
  fpHalfPath = os.path.join(outdir, f'{wellId}_processed_fpHalf.tif')
  inputPath = fpHalfPath if os.path.exists(fpHalfPath) else row['processed']  # adaptive fallback
  ```
- Colony / background intensity: `gui/tabs/run.py:278-279`
  ```python
  fpHalfPath = os.path.join(outdir, f'{wellId}_processed_fpHalf.tif')
  intensityPath = fpHalfPath if os.path.exists(fpHalfPath) else rawPath  # _registered_raw.tif fallback
  ```

`saveFpHalf` only recently became default `True` in `gui/state.py:34`, **but several
call sites still default it to `False` when the key is absent**: `run.py:138`,
`run.py:1144`, `parameters.py:90`, `parameters.py:581`.

**Mechanism / why old ≠ new:**
- Plates processed *before* the fpHalf era have only adaptive `_processed.tif`.
  Their whole-image features were computed on the **adaptive** rendering (Issue 2
  drift); their colony features fell back to **`_registered_raw.tif`** (fully
  uncorrected raw intensities).
- Plates processed *with* fpHalf compute whole-image features on the **fixed-0.5**
  rendering and colony features on **fpHalf**.

So earlier and later plates are literally measured off **different image products**.
That alone produces a systematic batch shift in every intensity, percentile,
entropy, and Haralick feature — independent of biology.

**Affected outputs:** all of `_wholeImageFeatures.csv`, `_perColonyFeatures.csv`,
`_wellColonyFeatures.csv` → master CSVs → UMAP.

**Detection (check this first):** for a sample of "earlier" vs "later" plates,
confirm whether `_processed_fpHalf.tif` exists per well. Mixed presence across eras
explains a systematic offset by itself.

**Remediation direction:** require a single canonical feature input (fpHalf) and
fail loudly if missing rather than silently falling back; regenerate fpHalf for
legacy plates before comparing across eras.

---

### Issue 4 — `_toBitDepthScaled` infers bit depth from one stack's max for float inputs — **MEDIUM (conditional, catastrophic when triggered)**

`processing/analysis_main.py:27-34`
```python
if np.issubdtype(arr.dtype, np.integer):
    return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)   # safe
arr = arr.astype(np.float32, copy=False)
amax = float(arr.max()) if arr.size else 0.0
if amax > 1.5:
    bitDepthScale = 255.0 if amax <= 255.0 else 65535.0             # content-dependent
    return arr / bitDepthScale
return arr
```

Integer inputs are safe. But a **float** stack whose true range is 12/16-bit yet
happens to max out ≤ 255 is divided by 255 instead of 65535 — a **256× scale error**
that depends on the data's content, not its true bit depth.

**Mechanism / why old ≠ new:** if any session exported float TIFFs (or a different
bit depth) while others exported integers, the float sessions can land on a different
photometric axis than the integer sessions.

**Affected outputs:** everything downstream — biomass, OD, all intensity features.

**Detection:** check the dtype (and value range) of the raw TIFFs across the two
eras.

**Remediation direction:** resolve bit depth from TIFF metadata rather than the
observed max; reject/flag ambiguous float stacks.

---

## B. Thresholds & metadata that retune per session

### Issue 5 — Fixed mask threshold applied to the normalized image — **HIGH (propagates everywhere)**

`processing/analysis_main.py:143` → `processing/segmentation.py`:
```python
masks[:] = stack > float(fixedThresh)   # fixedThresh default 0.04
```

The normalized residual magnitude depends on contrast/illumination, so a fixed cut
yields different masks across sessions.

**Mechanism:** mask differences cascade into biomass, dust correction, colony
seeding, and **every geometry feature**. The high-pass reduces but does not remove
sensitivity to background/contrast level.

**Affected outputs:** `_masks.npz`, biomass, all colony geometry, colony counts,
tracking.

**Remediation direction:** consider an adaptive/relative threshold (e.g. percentile
of the normalized residual) or per-session calibration of `fixedThresh`.

---

### Issue 6 — Absolute biomass seed threshold gates which frames exist — **MEDIUM (structural, easy to miss)**

`colony/runTrackingGUI.py` — `BIOMASS_THRESHOLD = 0.005`, applied in `findSeedFrame`
to `biomass = nanmean((1 - rawCropped) * masks)` (`analysis_main.py:161`).

**Mechanism:** because `1 - raw` is exposure-dependent (Issue 1), a dimmer or
brighter session crosses 0.005 at a different frame → different `seedFrame` →
different tracked-frame set. This does **not** shift a column value; it changes
**which rows exist** in the output, which is invisible until you compare
distributions or frame counts.

**Affected outputs:** `trackedFrames`/`seedFrame` in the labels NPZ → which frames
feed feature extraction.

**Detection:** compare seed-frame index and tracked-frame counts for equivalent
wells across eras.

**Remediation direction:** derive the seed threshold from a normalized biomass or a
per-session baseline rather than an absolute constant.

---

### Issue 7 — `pxToUm` scales ~14 µm features and varies by scope/objective/metadata — **MEDIUM (only if acquisition setup differs)**

`colony/colonyFeatsMicrons.py` — `area_um2`, `perimeter_um`, major/minor axis
lengths, `distanceToCenter_um`, `nnDistance*_um`, `mstEdge*_um` all multiply by
`pxToUm` / `pxToUm²`. The value comes from per-plate TIFF metadata
(`gui/tabs/run.py:290-308`, stored in `experiment_config.json`).

**Mechanism / why old ≠ new:** the lab runs multiple Cytation5s where the same
filename suffix can map to different objectives. If "earlier" and "later" data came
off different microscopes — or metadata parsed/rounded differently — all µm-scaled
geometry shifts proportionally while pixel-space features stay put. That divergence
(µm features shift, pixel features don't) is itself a diagnostic signature.

**Affected outputs:** all `*_um` / `*_um2` colony geometry and spatial features.

**Detection:** confirm objective + `pxToUm` recorded in each plate's
`experiment_config.json` / `index.csv` match across the two eras.

**Remediation direction:** already partly handled (metadata-driven, fails loudly if
missing); add a cross-plate consistency check / warning when `pxToUm` differs for the
same nominal objective.

---

## C. Acquisition-level artifacts (raw data, not fixable downstream)

### Issue 8 — Sensor saturation on low-biomass (bright) wells from insufficient exposure headroom — **CRITICAL (unrecoverable)**

Discovered and characterized in Phase 2 (see results below). NOT a global
over-exposure and NOT a save-format bug — both were investigated and ruled out:

- Capture + display settings are **byte-for-byte identical** to 2024 training
  (CameraGain=4, LEDIntensity=3, BrightnessLevel=50, ContrastLevel=33,
  SaturationLevel=65504). So it is not "they cranked the exposure."
- A quantization-gap test (mid-range integer histogram) shows the 2025 pixels are
  **genuine sensor integers with no periodic comb** — there is no baked-in
  multiplicative save-scaling. The unclipped 2025 well B5 has the same
  integer-population structure as training.

**Actual mechanism:** the 2025 exposure lacks **headroom for the brightest wells**.
A bright, high-transmission well (low biomass) exceeds the sensor full well and rails
at the SaturationLevel (65504): in the 250513 Drawer5 plate, ~86% of such a well's
pixels sit in the top 5 counts (65500–65504). Wells with enough biomass to absorb
light stay below the ceiling and never clip. Once the bright field rails, the
transmission above the clip is flattened and **lost** — no anchor or normalization
recovers it; a flat-topped field also starves the local-contrast residual
(`raw − boxMean ≈ 0`) → weak masks on those frames.

**Spatial + temporal structure (it is biology-correlated, not random):** clipping
tracks **low biomass**. In Drawer5, columns 1–3 (growth-defect mutants — little
biofilm, bright wells) clip at frame 0; columns 4–6 (normal growers) do not — a clean
left/right split across all rows. It clears within ~4–8 frames as even the defect
wells accumulate a trace of material and drop a few counts below the ceiling.

**Why this is the opposite of a harmless-blank artifact:** the wells it corrupts are
the **growth-defect phenotype** — the biologically interesting low-biomass class — in
exactly their early frames, where a low-biomass mutant most resembles a bright empty
well. In 2024 (with headroom) the same phenotype recorded true intensities (~48000);
in 2025 it rails at 65504. So low-biomass-well intensity/whole-image features differ
between campaigns **purely from clipping** — a phenotype-specific old↔new artifact.

**Masked in QC:** invisible in the overlay/processed videos, because the display
normalization re-centers any flat-bright field to mid-gray. "Video looks fine" and
"raw is railed" are both true.

**Interaction with the anchor:** the chosen photometric anchor estimates background
from *early frames* — exactly where low-biomass wells are clipped. The anchor must
estimate from the earliest **unclipped** frame per well and refuse (flag, exclude)
when none exists.

**Affected outputs:** early-frame intensity / whole-image / background features of
low-biomass wells; masks/biomass on clipped frames. Caveat: characterized on **one**
reimaging plate (250513 Drawer5, 18 wells at 10x); raw data for other reimaging plates
is not on the mounts (only overlays). Generality needs confirmation.

**Remediation direction:** (1) detect clipped well-frames at ingest (fraction of
pixels ≥ SaturationLevel above a threshold) and mark their intensity features
invalid / exclude from cross-campaign comparison; (2) root fix is upstream —
acquisition with enough headroom (lower LED/gain/integration) that the brightest
low-biomass wells stay off the ceiling.

### Issue 9 — Timelapse length / cadence mismatch across campaigns — **MEDIUM (breaks frame-indexed features)**

2025 reimaging has **17 frames**; 2024 training has **31**. The wide-table pivot for
UMAP (`analysis/wide_table.py`) keys features as `<feature>_t<frame>`. Different
lengths/cadence misalign those columns across campaigns independent of photometry: the
same wall-clock biology lands in different `t<frame>` bins, or columns simply don't
exist for one campaign.

**Affected outputs:** the collapsed wide table and every UMAP/model built on
frame-indexed features that mixes campaigns.

**Remediation direction:** align on real time (acquisition interval from metadata)
rather than frame index, or resample to a common time grid, before pivoting.

## Corrections to common assumptions

- **`img_as_ubyte` is NOT an adaptive per-image stretch.** In
  `wholeImage/extractWholeImageFeats.py:38`, for a float [0,1] image it is a *fixed*
  `round(img * 255)`. So the uint8 cast and Haralick quantization are **stable as
  long as the input rendering is stable**. The whole-image batch fragility is
  entirely upstream (Issues 2 + 3), not the cast. Fixing input provenance fixes
  Haralick for free.
- **`fractalDimension` (`extractWholeImageFeats.py:13`) is defined but not called**
  by `extractFrameFeats`, so its hard-coded `0.9 * z.max()` threshold is not an
  active artifact in the GUI pipeline (but would be if that path were re-enabled).

---

## Diagnostic checklist (where to look first)

1. **Inventory file products.** For "earlier" vs "later" plates, check whether
   `_processed_fpHalf.tif` exists per well. Mixed presence ⇒ Issue 3 alone explains a
   systematic offset.
2. **Plot a session-invariant control.** Overlay distributions of `bgMeanIntensity`
   and `whole_meanIntensity` for believed-equivalent wells. Level shift, unchanged
   spread ⇒ photometric drift (Issues 1–3). Change in *which frames appear* ⇒
   Issue 6.
3. **Confirm scope/objective + `pxToUm`** match across eras (Issue 7).
4. **Check raw TIFF dtype/range** across eras for the Issue 4 trap.

## Priority ordering

1. Issue 3 (inconsistent rendering provenance) — most likely direct cause of old ≠ new.
2. Issue 2 (adaptive fpMean) — the drift Issue 3 exposes.
3. Issue 1 (no flat-field) — underlying reason fpHalf alone is insufficient.
4. Issue 5 (fixed mask threshold) — broad geometry propagation.
5. Issue 4 (bit-depth heuristic) — rare but catastrophic.
6. Issue 6 (absolute seed threshold) — structural, hard to spot.
7. Issue 7 (pxToUm) — only if acquisition setup differs across eras.

---

# Remediation Plan

## Decisions on record

Three strategic questions drove the plan (answered 2026-06-05):

1. **Raw TIFF data for earlier plates is fully available** → we can recompute every
   plate to one consistent standard rather than only correcting frozen outputs.
2. **Blank-field / empty-well references are not reliably available** → the
   photometric anchor must be **in-silico** (Option B), not flat-field/Beer-Lambert
   OD (Option A). The disabled `Imin`/`Imax` path stays disabled.
3. **Downstream models / UMAPs trained on the old features must keep working** →
   we **version, not mutate**. The current feature set is frozen as **v1**; the
   anchored recipe becomes **v2**.

## Strategy: version, don't mutate (v1 frozen, v2 anchored)

- **v1** = the existing feature set and CSVs. Left untouched and reproducible so
  current models/UMAPs keep running.
- **v2** = the anchored recipe below, computed for **every** plate (old and new —
  raw data exists for all). Because v2 reprocesses all eras through one recipe +
  one photometric anchor, old and new finally land in the same space. That is where
  the "old vs new" mismatch is actually resolved.
- Downstream consumers migrate v1 → v2 on their own schedule. v2 outputs go to a
  new schema version / new paths; nothing overwrites v1 in place.

## The v2 recipe

| # | Change | Issues addressed | Risk |
|---|---|---|---|
| 1 | fpHalf-only feature input; remove silent fallbacks at `run.py:239` (whole-image) and `run.py:279` (colony); error loudly if `_processed_fpHalf.tif` is missing | 2, 3 | Low |
| 2 | Unify `saveFpHalf` default to `True` across `run.py:138`, `run.py:1144`, `parameters.py:90`, `parameters.py:581` (single source of truth) | 2, 3 | Low |
| 3 | Resolve bit depth from TIFF `BitsPerSample` metadata; assert instead of guessing from `amax > 1.5` (`analysis_main.py:30-33`) | 4 | Low |
| 4 | **In-silico photometric anchor** — new normalization stage in `analysis_main.py` (design below) | 1, 2-residual | **Needs validation** |
| 5 | Keep `fixedThresh` / `BIOMASS_THRESHOLD` fixed initially, now applied to anchored input; revisit only with validation | 5, 6 | Deferred |
| 6 | Warn when same nominal objective yields different `pxToUm` across plates in a run | 7 | Low |

Adaptive `_processed.tif` survives **only** as the legacy overlay rendering, never
as a feature source.

## Why fpHalf alone is not enough (motivates change #4)

The display rendering is `clip(raw − boxMean(raw) + fpMean)`. The subtraction removes
an **additive** offset; fixing `fpMean = 0.5` nails the background gray level. But
exposure / gain / lamp drift is **multiplicative**: if exposure doubles, `raw` and
`boxMean` both double, so the residual `raw − boxMean` doubles too. fpHalf therefore
stabilizes rendering-centering (and Haralick bins) but leaves intensity-feature
amplitude riding on session gain. The photometric anchor (#4) is what removes that.

## In-silico photometric anchor — design

**Principle:** multiplicative gain cancels under division. Divide each frame by its
**background (empty-agar) level** `B`; background-relative transmission is a
blank-free Beer-Lambert with the in-frame agar acting as the blank. This unifies with
biomass: `1 − raw/B` becomes an absorbance anchored to agar — exactly what the
disabled `Imin`/`Imax` OD path computed.

**CRITICAL pitfall (must not violate):** estimate `B` from **background pixels only**
(bright agar mode / upper percentile) — **never** the global frame mean. As a biofilm
grows it darkens the field and the global mean drops; that drop *is the biomass signal
we measure*. Normalizing by global mean would regress out the signal. Agar brightness
stays ~constant (only colony pixels darken), so a mode / high-percentile estimate is
stable across the timelapse and safe.

**Open design choices to validate before committing code:**
- **Scope of `B`:** per-session vs per-well vs per-frame. Drift is mostly per-session;
  per-well also catches vignetting; per-frame risks instability once the field
  confluences and background pixels grow scarce. Leaning: estimate from early frames
  (frame 0 mostly empty), apply per-well — pending stability measurement.
- **Estimator:** histogram mode vs fixed high percentile (e.g. p90). Mode is more
  principled; percentile more robust at small N.

## Phased sequencing

> **Phase 2 runs first** — pressure-test the anchor on Drawer 7 before any code is
> committed (per decision 2026-06-05).

- **Phase 2 — anchor pressure-test (NEXT, no code commits):** prototype the background
  estimator on the Drawer 7 plate (B2 stall case, frames ~18, 10x/20x). Measure:
  (a) background-mode/percentile stability across the full timelapse, including
  confluent late frames; (b) old↔new control-well alignment under candidate
  estimators; (c) per-session vs per-well vs per-frame behavior. Decide estimator +
  scope from evidence.
- **Phase 1 — recipe fixes (low-risk, can land once anchor design is settled):**
  v2 recipe changes #1, #2, #3, #6 + the legacy `_processed_fpHalf.tif` backfill
  script (regenerate from `_registered_raw.tif`, no re-registration). Removes the
  recipe mismatch independent of the anchor.
- **Phase 3 — reprocess + validate:** compute v2 for all plates; verify shared
  controls overlay across eras; v1 left frozen.
- **Phase 4 — thresholds (only if needed):** relative `fixedThresh` (Otsu /
  percentile / k·MAD-above-background) and relative seed detection
  (fraction-of-plateau / first-derivative), gated on Phase 3 residuals and validated
  against the Drawer 7 stall cases.

## Validation assets

- **Drawer 7 plate:** `/mnt/bridgeslab/Good imaging data/Multi-phenotype training/
  241011_183053_...Drawer7` — B2 frame 18 at 10x and 20x is a known stall case.
- **Shared control wells** imaged across the two eras: the primary old↔new alignment
  check (distributions of `bgMeanIntensity`, `whole_meanIntensity` should overlay
  after anchoring).

## Phase 2 results — anchor pressure-test (2026-06-07)

Probe: `phase2_anchor/anchor_probe.py` on Drawer 7 wells **B5, B9** at 10x (`_03`)
and 20x (`_04`), comparing sessions **241011 (Oct-11)** and **241017 (Oct-17)** —
the same drawer imaged 6 days apart. Per-frame background-level estimators (histogram
mode, p90/95/99, mean) measured on the bit-depth-scaled raw stacks.

**Finding 1 — per-frame anchoring fails (confirms the pitfall).** Frames 0–6 are
flat; from frame ~7 every estimator — including the mode — slides down as the biofilm
confluences (B9 10x mode 0.78 → 0.64). Background pixel fraction collapses 0.37 →
0.10. A per-frame divisor would regress out biomass signal. **Rejected.**

**Finding 2 — early-frame per-well scalar anchor is stable and safe.** Frames 0–4 are
flat to <0.5% within a session and agree across sessions. A single per-well constant
only rescales the stack, so it cannot remove temporal/spatial biology by
construction. **This is the chosen anchor.** Estimator = histogram mode (p95 gives
the same answer to ~0.1%; either is acceptable). Background reference ≈ 0.78 in
[0,1], consistent across wells and magnifications.

**Finding 3 — true illumination drift on this pair is ~1%.** Measured on empty-agar
early frames (frames 0–4, biology-independent), Oct-11 → Oct-17 background drift is
−0.5% to −1.1% across all wells/mags/estimators (frame-0-only: −0.4% to −1.1%). The
larger apparent drift seen when averaging over all frames was biology (Oct-17 is
6-days-older, denser biofilm), not illumination.

**Implication.** This session pair barely stresses the anchor — the multiplicative
drift it removes is ~1% here. So (a) the anchor *design* is validated but its
*impact* is not yet demonstrated on this pair; we need a higher-drift pair (different
microscope, months apart, or across a lamp change) to prove it removes meaningful
drift. (b) For closely-spaced sessions on a stable scope, the "old vs new" mismatch
is likely dominated by the recipe/provenance (Issues 2, 3) and threshold (Issues 5,
6) artifacts rather than photometric gain — which would raise the priority of
Phase 1 relative to the anchor. Pending confirmation on a known-divergent pair.

### Phase 2 part 2 — cross-campaign (2024 training vs 2025 reimaging) (2026-06-07)

Probe: `phase2_anchor/cross_campaign_probe.py`. Added the 2025 reimaging campaign
(250513 Drawer5, May-2025, single-10x BF_YL protocol) — the only reimaging plate with
raw TIFFs on the mounts; all others under `vcReimaging/` are overlays-only. Same
dtype/objective/pxToUm as training (Issue 4 not triggered). Early-frame background
mode per well at 10x:

| session | bg mode | within-session spread |
|---|---|---|
| 2024 train Oct-11 | 0.785 | 0.037 |
| 2024 train Oct-17 | 0.777 | 0.025 |
| 2025 reimg May-13 | 0.954 | **0.274** |

Nominal drift 2024→2025 ≈ **+21.5%**, but the huge spread exposed the real cause:
**saturation**, not smooth gain. 5/6 of the 2025 wells have 94–99% of pixels pinned
at the sensor ceiling in early frames; the background mode rails at ~1.0. See
**Issue 8**. Saturation self-corrects over the timelapse (B2: 99%→14%→0%). The 2025
plate also has only 17 frames vs 31 (see **Issue 9**).

**Revised Phase 2 verdict:**
- The early-frame per-well background anchor is **sound for well-exposed data**
  (training; within-2024 drift ~1%) but **fails on saturated sessions** because the
  anchor frames are clipped. It must estimate from the earliest *unsaturated* frame
  and refuse when none exists.
- The dominant 2024↔2025 intensity mismatch is **sensor saturation on low-biomass
  (bright) wells from insufficient exposure headroom (Issue 8)** — NOT a settings
  crank and NOT a save bug (both ruled out: identical capture/display metadata; no
  quantization comb). Genuine, unrecoverable clipping; re-ranks above the anchor for
  the training-vs-reimaging case. Clipping is biology-correlated (tracks low biomass:
  growth-defect mutants in cols 1–3 clip; normal growers in cols 4–6 do not) and
  time-limited (clears in ~4–8 frames).
- Confidence caveat: one reimaging plate / 18 wells at 10x. Confirm generality once
  more reimaging raw data is locatable.

### Phase 3 — root cause confirmed: rendering version drift (2026-06-08)

Pivoted from the photometric/saturation hypotheses (which the data did not support
as dominant) to the user's pipeline-version-drift hypothesis, using
`biofilm-analysis` as the source of old-processing reference features
(`reimagingIndex.csv` → `registered_raw` paths + `_wholeImage_mahotas_v2.csv`).

**Test A — extraction code is unchanged.** Running *current* `extractWholeImageFeats`
on the *stored* atlas `_processed.tif` reproduces the stored haralick **exactly**
(`max|Δ| = 0.00000`). The feature-extraction code did not drift.

**Test B — the render drifted catastrophically.** The stored atlas `_processed.tif`
is a **near-zero-centered** local-contrast residual (range ~[−0.02, 0.09]); the
*current* pipeline renders a **0.5-centered, fpMean-offset, clipped** display
(~[0.45, 0.68]). Re-rendering the same `registered_raw` through the current pipeline
moves `haralick_0` 0.73→0.003 and `haralick_12` 0.82→0.03 — a 100× shift. Haralick
runs `img_as_ubyte` on `_processed.tif`, so the two renders quantize into entirely
different graycomatrix regimes.

**Reprocess-consistency test — the fix works.** Rendering BOTH the clean-deletion
and reimaging batches from `registered_raw` through one **fixed-fpMean (`fpHalf`)**
render + same extraction, the WT↔WT haralick batch effect collapses:

| feature | reim-WT | cleanDel-WT | z (reim SD) | stored z |
|---|---|---|---|---|
| haralick_0 | 0.011 | 0.011 | **−0.05** | — |
| haralick_2 | 0.289 | 0.670 | **2.56** | — |
| haralick_12 | 0.345 | 0.716 | **2.28** | **≈ −50** |

`z ≈ 50 → ≈ 2`. Two effects: the raw gap shrank (0.51→0.37) **and** the reim-WT
`haralick_12` std went 0.01→0.16 — i.e. the old render had compressed haralick
against a ceiling, manufacturing the tiny std that let the scaler amplify a modest
delta into 50σ. Script: `phase2_anchor/reprocess_consistency_test.py`.

Caveats: n=5 wells/batch, one clean-del plate, frames matched by index not real time
(camera restart adds ~1h offset); the residual ~2σ is approximate (true optics /
timing / small-n). The 50σ→2σ collapse is the robust result.

**Conclusion.** The dominant clean-del↔atlas batch effect is **fixable
processing-version drift in the render**, not acquisition optics. Fix = one pinned
render (`fpHalf`) + reprocess; version stamp prevents recurrence.

### Phase 4 — planned: validate fpHalf on training raw, then retire adaptive fpMean

Before flipping the pipeline to fpHalf-only, validate the fixed render on the raw
**training** data at `/mnt/bridgeslab/Good imaging data/Multi-phenotype training/`:

1. Generate `fpHalf` `_processed.tif` renderings from the raw training stacks and
   **visually QC** them — confirm the processed images look normal/good (not washed
   out, biology clearly visible) across magnifications.
2. Extract whole-image (+ colony) features under **fpHalf vs adaptive fpMean** on the
   same wells and **compare** — quantify how much each feature shifts, confirm fpHalf
   is well-behaved (no degenerate ceiling compression like the old atlas render) and
   that biological signal is preserved.
3. If both look good, **retire adaptive fpMean**: make `fpHalf` the sole `_processed.tif`
   rendering and feature input, regenerate overlays from it, drop the `_fpHalf` suffix.
   (This is the migration already foreshadowed in CLAUDE.md "fpMean policy".)
4. **End-to-end cross-batch validation — process from RAW.** Run the full fixed
   pipeline (registration → fpHalf render → feature extraction) starting from the
   **raw TIFFs** — not the `registered_raw` shortcut the Phase 3 test used — for
   matched biology in two batch pairs, then compare cross-batch to confirm the batch
   effect is actually gone end-to-end:
   - **reimaging ↔ clean-deletions** (WT control + gene pairs BioD↔bioD, ManA↔manA,
     PdhE2↔pdhE2)
   - **training ↔ reimaging** (shared controls / overlapping strains, e.g. WT)
   Going from raw also exercises any **registration/preprocessing** version drift, not
   just the render (Phase 3 held registration constant by starting from `registered_raw`).
   Success = matched biology overlaps in feature distributions / UMAP across batches.

### Phase 4 P0 results — fpHalf validated on training raw (2026-06-13)

`phase2_anchor/training_fpHalf_validate.py` on 5 training wells (241011 Drawer7, 10x),
rendering each from raw through BOTH adaptive and fixed fpHalf.

- **QC: fpHalf renders look normal** (montage `phase2_anchor/training_fpHalf_qc/`) —
  visually indistinguishable from adaptive, biology clearly visible in mature frames,
  no washout / over-clipping.
- **`haralick_0/2/12` + `entropy` identical** adaptive vs fpHalf (mean|Δ|=0.0000, same
  cross-well spread) — they measure *relative* structure, invariant to the additive
  fpMean offset when neither render clips.
- **`meanIntensity` drifts with adaptive** (cross-well SD 5.25, tracking each well's
  fpMean 0.564–0.618) and is **pinned by fpHalf** (SD 0.0001; every well at 127.5 =
  0.5×255). fpHalf removes exactly this intensity-location drift.

**Refined mechanism (updates the headline).** The historical 50σ haralick batch effect
was NOT adaptive-vs-fixed fpMean — it was the OLD render **clipping** (fpMean ≈ 0 →
negative residuals clipped to 0 → compressed GLCM). Both *modern* renders center the
residual mid-range, never clip, and are texture-equivalent. Therefore:
- The texture-batch fix is **reprocessing old data onto the modern non-clipping
  render** (the atlas used the old clipping render) — a *consistency* fix satisfied by
  either modern render.
- fpHalf's *additional* benefit is pinning the intensity-LOCATION features
  (`meanIntensity`, percentiles, `bg*`) that adaptive lets drift well-to-well.

**Verdict: GO** — fpHalf is safe (identical texture, normal renders) and strictly
better for intensity consistency. Proceed to retire adaptive + reprocess. Caveat: these
5 wells did not trigger clipping under either render, so the edge case where adaptive
mis-centers and clips (and fpHalf would not) is not exercised — but that only
strengthens the case for fpHalf.

### Phase 5 — planned: frame-0 OD signal path (the real signal-preserving fix)

Resolves the architecture critique (features computed on a *display render*, not a
signal), Issue 1 (no flat-field), and a newly-identified **planktonic blind spot**.

**Planktonic blind spot (finding, 2026-06-14).** The local-contrast normalization is a
high-pass (subtract a ~`blockDiameter`-blurred copy), so it removes the slow/large-scale
background **level**. Features are computed on that render, and biomass is
mask-restricted (`mean((1−raw)·mask)`), so the pipeline is **blind to a uniform medium
darkening from planktonic cells**. Whole-image Haralick *does* keep background **texture**
(it is whole-image and fine texture survives the high-pass) — but not the background
*level*. The Julia reference computed `planktonic = mean(OD·¬mask)`; the Python pipeline
dropped it. Consequence: a dispersal mutant reads only as "less biofilm" (biomass/area
drop); the planktonic increase is invisible, so "never grew" can't be distinguished from
"grew then dispersed."

**The fix — frame-0 OD, no calibration images needed.** Growth starts from a biofilm-free
well (biofilm never appears before ~frame 8), so **frame 0 — or a frames 0–4 average for
robustness — is a valid per-well, per-pixel blank.** Proper optical density is therefore
computable with NO `Imin`/`Imax` acquisition:

    OD_t(x,y) = −log10( I_t(x,y) / I_blank(x,y) )     I_blank = mean(frames 0..4)

(floor/`Imin`-subtract the denominator against dark/dead pixels).

Properties:
- **Batch-invariant** — a ratio against the well's own blank cancels exposure/gain
  (num & denom scale together) and illumination/vignetting (per-pixel flat-field). This
  is the exposure-invariance the fpHalf render only partially provides.
- **Level-preserving** — keeps absolute absorbance, so it captures biomass
  (`mean(OD·mask)`) AND planktonic (`mean(OD·¬mask)`) — the dispersal axis.
- **One representation for everything** — OD can feed intensity, biomass, planktonic, and
  Haralick (with fixed quantization) as a batch-invariant texture input.
- **Retroactive** — frame 0 exists for every well in every dataset (training, reimaging,
  clean-del), so OD is computable on all existing data without re-acquisition. Corrects
  the earlier "OD needs blanks we don't have" framing.

Caveats: requires good registration for the per-pixel ratio (the frame-to-frame + NaN-crop
fix provides it); guard frame-0 dark/dead pixels; frame 0/0–4 must be clean (in-focus,
dust-free — use the 0–4 average + dust correction); PSF/focus differences across
microscopes are spatial-frequency effects a per-pixel ratio does NOT remove (the residual
~2σ cross-scope texture difference would remain). Distinct from — and better than — the
spatial-background-percentile normalization prototyped earlier (which did not help); the
temporal frame-0 ratio is the right one. Additive: it does not disturb the current fpHalf
features.

**Status (2026-06-14):** the **biomass** half is implemented — the no-`Imin`/`Imax` path
now computes `biomass = mean(OD·mask)` with `OD = −log10(I_t / mean(frames 0–4))`
(`analysis_main.py`), so `numericalData/<mag>X_BF.csv` carries batch-invariant OD biomass.
Validated on a real well: OD ≈ 0 at frames 0–4, same trajectory shape as the old
`(1−raw)` measure, and the seed-frame threshold (0.005) fires within one frame of before.
**Remaining:** the `planktonic = mean(OD·¬mask)` measure (own CSV + `<mag>X_planktonic.csv`)
and OD-based intensity/Haralick texture features.
