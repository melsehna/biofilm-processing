#!/usr/bin/env bash
#
# Headless full-pipeline run — "Project - Cluster" set (Jesse), 10x only.
# Authored 2026-06-27. Reproducible record of exactly what was run.
#
# WHAT IT DOES (per well, via the GUI's canonical ProcessingWorker, run headless
# by cli/run_pipeline.py): registration -> segmentation -> mask-overlay MP4 +
# processed-video MP4 -> whole-image features -> colony tracking -> colony
# features -> per-mag numericalData CSVs -> run-level master CSVs.
#
# INPUT:  /mnt/bridgeslab/Jesse/Project - Cluster/
#         8 experiment dirs (listed below). Each contains a single nested
#         "<timestamp>_Plate 1" subdir holding the per-well TIFFs. We pass those
#         plate subdirs DIRECTLY (not the experiment dirs) because one experiment
#         — 260512_hand_cleanDeletion_3 — also contains a *real nested duplicate*
#         of its plate one level deeper; targeting the direct "_Plate 1" child
#         avoids processing that plate twice.
#
# DATA:   8 plates, 10x. Wells: 96 (x5), 80 (x1), 64 (x2) ~= 688 total.
#         41 frames per well (uniform) — note this differs from the 31-frame
#         reimaging/training sets (matters for embedding nFrames later).
#
# MAGNIFICATION: _02 == 10x on this set, verified from Cytation TIFF metadata
#         (objective=10, pxToUm=0.6973). The folder name is not trusted; metadata is.
#
# NAS MIRROR (LEAN): process each plate to LOCAL fast scratch (--output-dir on
#         /mnt/data), then rsync that plate's outputs to the NAS (--nas-mirror-dir)
#         and delete the local copy. --nas-lean drops the heavy intermediates from
#         the NAS copy (registered_raw.tif, masks.npz, trackedLabels*.npz); the NAS
#         keeps numericalData/ + master_*.csv + per-well CSVs + index.csv,
#         <well>_processed.tif, <well>_processed.mp4, <well>_overlay.mp4.
#         /mnt/bridgeslab is READ-ONLY; the writable view is /mnt/phenotyper.
#
# PARAMS: blockDiam=101 fixedThresh=0.0250 fftStride=1 downsample=1 shiftThresh=250
#         minColonyArea=200px propRadius=50px workers=40  (same proven 10x set as
#         the reimaging run; 56-core/251GB box, cap 42).
#
# RUN MODE: detached via nohup, timestamped log next to this script, returns
#         immediately (survives SSH disconnect). Monitor with the printed tail -f.
#
# >>> EDIT/CONFIRM THESE TWO PATHS BEFORE RUNNING <<<
set -euo pipefail
cd /home/smellick/biofilm-processing

INPUT="/mnt/bridgeslab/Jesse/Project - Cluster"
STAGING="/mnt/data/tmp/cluster-data-062726/clusterData"          # local scratch (outputDir)
NAS="/mnt/phenotyper/Sehna/cluster-data-062726/clusterData"      # final NAS dest (nasMirrorDir)

# The 8 experiment dirs (user-specified). For each, take its DIRECT "<ts>_Plate N"
# child — this excludes the nested duplicate under 260512_hand_cleanDeletion_3.
EXPERIMENTS=(
    "260414_CmpdTreatment_V1"
    "260508_robot_cleanDeletion_1"
    "260509_robot_cleanDeletion_2"
    "260509_robot_cmpdTreatments"
    "260512_hand_cleanDeletion_3"
    "260521_hand_cleanDeletion_1"
    "260522_hand_cleanDeletion_2"
    "260522_hand_cmpdTreatment_1"
)

shopt -s nullglob
plates=()
for e in "${EXPERIMENTS[@]}"; do
    matched=0
    for pd in "$INPUT/$e"/*_Plate\ *; do
        [ -d "$pd" ] || continue
        plates+=( "$pd" ); matched=1
    done
    [ "$matched" = 0 ] && echo "WARNING: no '_Plate' subdir found under $e" >&2
done
echo "Plates matched: ${#plates[@]} (expected 8)"
printf '  %s\n' "${plates[@]}"

mkdir -p "$STAGING"

TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/smellick/biofilm-processing/scripts/run_cluster_${TS}.log"

nohup python scripts/runHeadless.py \
    --plates "${plates[@]}" \
    --output-dir "$STAGING" \
    --nas-mirror-dir "$NAS" \
    --nas-lean \
    --mag _02 \
    --workers 40 \
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
    --processed-video \
    < /dev/null > "$LOG" 2>&1 &

PID=$!
echo
echo "Run launched detached (survives logout)."
echo "  PID:      $PID"
echo "  Log:      $LOG"
echo "  Monitor:  tail -f \"$LOG\""
echo "  Stop:     kill $PID    # graceful — finishes the current well, then stops"
