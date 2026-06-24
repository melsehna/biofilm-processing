#!/usr/bin/env bash
#
# Headless full-pipeline run — Multi-phenotype training set, 10x only.
# Authored 2026-06-14. Reproducible record of exactly what was run.
#
# WHAT IT DOES (per well, via the GUI's canonical ProcessingWorker, run headless
# by cli/run_pipeline.py): registration -> segmentation -> mask-overlay MP4 +
# processed-video MP4 -> whole-image features -> colony tracking -> colony
# features -> per-mag numericalData CSVs -> run-level master CSVs.
#
# INPUT:  /mnt/bridgeslab/Good imaging data/Multi-phenotype training/
#         18 plate dirs matching *_Discontinuous_Drawer* (each has a nested
#         "<...>_Plate 1" subdir the runner auto-expands). Other dirs in that
#         folder (BF, Embeddings, Kleb training set, Reanalyzed, ...) are skipped
#         by the glob.
#
# MAGNIFICATION: _03 == 10x, verified from Cytation TIFF metadata
#         (_02=4x, _03=10x, _04=20x, _05=40x on this set).
#
# NAS MIRROR (speed on the CIFS NAS): process each plate to a LOCAL fast scratch
#         (--output-dir on /mnt/data, 22 TB ext4), then rsync that plate's
#         outputs to the NAS (--nas-mirror-dir) and delete the local copy.
#         Note: /mnt/bridgeslab is mounted READ-ONLY; the writable view of the
#         same server is /mnt/phenotyper, so the NAS dest lives there.
#
# PARAMS: blockDiam=101  fixedThresh=0.0250  fftStride=1  downsample=1
#         shiftThresh=250  minColonyArea=200px  propRadius=50px  workers=16
#         (machine: 56 cores / 256 GB; workers hard-capped at 75% of cores).
#
set -euo pipefail
cd /home/smellick/biofilm-processing

INPUT="/mnt/bridgeslab/Good imaging data/Multi-phenotype training"
STAGING="/mnt/data/tmp/multiphenotype-data-061426/trainingData"          # local scratch (outputDir)
NAS="/mnt/phenotyper/Sehna/multiphenotype-data-061426/trainingData"      # final NAS dest (nasMirrorDir)

mkdir -p "$STAGING"

# 18 drawer dirs; quoted glob preserves the spaces in the names.
shopt -s nullglob
plates=( "$INPUT"/*_Discontinuous_Drawer* )
echo "Plates matched: ${#plates[@]}"

python scripts/runHeadless.py \
    --plates "${plates[@]}" \
    --output-dir "$STAGING" \
    --nas-mirror-dir "$NAS" \
    --mag _03 \
    --workers 16 \
    --block-diam 101 \
    --fixed-thresh 0.0250 \
    --fft-stride 1 \
    --downsample 1 \
    --shift-thresh 250 \
    --min-colony-area 200 \
    --prop-radius 50 \
    --whole-image \
    --colony-tracking \
    --colony-feats \
    --overlays \
    --processed-video
