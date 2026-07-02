# legacy — quarantined, unsupported modules

**Not part of the supported pipeline.** Nothing in the active codebase imports
these modules. They are kept for reference / backward compatibility and are not
maintained. Do not extend them — add new work to the supported paths
(`processing/`, `colony/`, `wholeImage/`, `gui/`, `cli/`).

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

**Not yet quarantined (candidates, left in place):** the `colony/*MP*` and
`intensity/` multiprocess batch scripts — they have no importers but may still be
run directly, so they were left until confirmed retired.
