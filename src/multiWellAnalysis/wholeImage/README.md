# Whole-image phenotype extraction

This folder contains scripts for fast whole-image feature extraction
and phenotype embedding from processed timelapse microscopy data.

## Overview

Pipeline:

1. Extract whole-image features per frame (Mahotas-based)
2. Aggregate features per well replicate (mean + std)
3. Embed phenotypes using UMAP
4. Optionally combine with biomass trajectories

All whole-image features are computed **after frame 6** to avoid
early imaging artifacts.

## Scripts

### Feature extraction

- `extractWholeImageFeatures.py`
  - Fast, O(N) per-frame feature computation
  - Uses Mahotas Haralick + intensity statistics
  - Safe for large (~2000×2000) images

- `12-31-runWholeImage.py`
  - Parallel runner with per-well logging and checkpointing
  - Designed for cluster or multi-core execution

### UMAP analysis

- `umap_wholeImage_replicates.py`
  - One UMAP point per well replicate
  - Uses only morphology features

- `umap_combined_replicates.py`
  - Combined phenotype embedding:
    biomass trajectories + morphology

## Outputs

- Per-well CSVs: *_wholeImage.csv


## Notes

- Frame filtering: `frame > 6`
- NaNs handled by median imputation after feature aggregation
- Intended for exploratory phenotype clustering, not segmentation

## Status

Frozen version used for downstream analysis.



