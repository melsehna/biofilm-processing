# legacy — quarantined, unsupported modules

**Not part of the supported pipeline.** Nothing in the active codebase imports
these modules. They are kept for reference / backward compatibility and are not
maintained. Do not extend them — add new work to the supported paths
(`processing/`, `colony/`, `wholeImage/`, `gui/`, `cli/`).

Some modules here are **broken as-found** (missing symbols, hardcoded personal
paths) and are not guaranteed to import or run — e.g. `runIntensityFeatsMP.py`
imports a `loadProcessedStack` that was never defined, so it `ImportError`ed
before quarantine too. They are archived, not fixed.

| Module | Why it's here |
|---|---|
| `batch_runner.py` | Legacy, divergent biomass-only multi-plate runner (writes a different output layout, needs a `ReplicatePositions.csv`). Superseded by the `ProcessingWorker` path (`cli/run_pipeline.py`). |
| `pipeline.py` | Thin legacy wrapper (`Pipeline()`) over `batch_runner` + plotting, with hardcoded `/home/smellick/ImageLibrary/` paths. Was exported from `processing/__init__` but never called. |
| `feature_extraction.py` | Older standalone feature module; **not** called by the GUI/CLI pipeline (which uses `colony/colonyFeatsMicrons.py` + `wholeImage/extractWholeImageFeats.py`). |
| `verifyColFeats.py` | One-off colony-feature verification script with hardcoded `/mnt/data/...` paths. |
| `runGUI.py` | Legacy GUI Run-tab stub, not imported by `gui/app.py` (references functions that no longer exist). |
| `config.py` | Legacy GUI config-tab stub, not imported by `gui/app.py`. |
| `threshPrev.py` | Legacy GUI threshold-preview stub, not imported by `gui/app.py`. |

The still-supported legacy single-plate CLI `scripts/runSinglePlate.py` imports
`batch_runner` from here.

### Second quarantine pass (unreachable orphans)

Determined by import-reachability analysis: not reachable from any entry point
(`gui.app`, `cli.run_pipeline`, `cli.test_well`) or documented setup/run script,
and not required to install/run the package.

| Module | Origin | Note |
|---|---|---|
| `runTrackingMpReimaging.py`, `runTrackingMpTraining.py` | `colony/` | Multiprocess tracking batch scripts (hardcoded paths); run directly, never imported. |
| `runColonyFeatsTrackedMP.py` | `colony/` | Multiprocess colony-feature batch script. |
| `colonyFeats.py`, `wellAgg.py` | `colony/` | Pre-microns feature/aggregation code; superseded by `colonyFeatsMicrons.py` / `wellAggMicrons.py`. |
| `aggregateColonyFeats.py`, `makeTrackingGifs.py` | `colony/` | Standalone helpers, not wired in. |
| `intensityFeats.py`, `io_utils.py`, `runIntensityFeatsMP.py` | `intensity/` | The entire `intensity/` subpackage — a separate per-pixel-intensity feature path never wired into the GUI/CLI pipeline. Subpackage retired. |
| `plotting_tools.py` | `processing/` | Unused plotting helpers. |
| `buildTrainingProcIndex.py`, `buildReimagingProcIndex.py`, `12-31-runWholeImage.py` | `wholeImage/` | One-off index builders / dated script (`12-31-…` already had a broken import). |

The still-supported batch tools `colony/runColFeatsCLI.py` and
`wholeImage/runWholeImage.py` (README "Batch CLI tools") were **kept** — they are
documented, user-runnable, and import only supported modules.
