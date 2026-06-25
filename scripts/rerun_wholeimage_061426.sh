#!/usr/bin/env bash
# Re-run the 18-plate multiphenotype dataset (2026-06-14) to add missing
# whole-image features, mirror completed plates to NAS, and delete local 3.3 TB.
#
# WHY: The original run (run_multiphenotype_061426.sh) had a bug where
# _wholeImageOneWell used `outdir` before defining it, so whole-image features
# were never written (0 _wholeImageFeatures.csv across all 1728 wells).
#
# HOW RESUME WORKS: The pipeline checks run_params.json (8 processing params
# only — not stage flags). Since blockDiam/fixedThresh/etc. match and all 1728
# _processed.tif exist, processing is skipped entirely. Only whole-image runs.
# Colony tracking + colony feats are already done — explicitly disabled here.
# NAS mirror rsyncs each plate after whole-image (size-only check, so large
# files already on NAS aren't retransferred) then deletes the local copy.
#
# Usage: bash scripts/rerun_wholeimage_061426.sh

set -euo pipefail
cd /home/smellick/biofilm-processing

INPUT="/mnt/bridgeslab/Good imaging data/Multi-phenotype training"
STAGING="/mnt/data/tmp/multiphenotype-data-061426/trainingData"
NAS="/mnt/phenotyper/Sehna/multiphenotype-data-061426/trainingData"

shopt -s nullglob
plates=( "$INPUT"/*_Discontinuous_Drawer* )
echo "Plates matched: ${#plates[@]}  (expect 18)"

biofilm-processing-run \
    --plates "${plates[@]}" \
    --output-dir    "$STAGING" \
    --nas-mirror-dir "$NAS" \
    --mag           _03 \
    --workers       16 \
    --block-diam    101 \
    --fixed-thresh  0.025 \
    --shift-thresh  250 \
    --fft-stride    1 \
    --downsample    1 \
    --whole-image \
    --no-colony-tracking \
    --no-colony-feats \
    --overlays \
    --processed-video \
    --min-colony-area 200 \
    --prop-radius   50
