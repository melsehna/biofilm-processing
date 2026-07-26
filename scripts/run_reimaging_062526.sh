#!/usr/bin/env bash
#
# Headless full-pipeline run — TN-Library "reimaging set", 10x only.
# Authored 2026-06-25. Reproducible record of exactly what was run.
#
# WHAT IT DOES (per well, via the GUI's canonical ProcessingWorker, run headless
# by cli/run_pipeline.py): registration -> segmentation -> mask-overlay MP4 +
# processed-video MP4 -> whole-image features -> colony tracking -> colony
# features -> per-mag numericalData CSVs -> run-level master CSVs.
#
# INPUT:  /mnt/bridgeslab/Good imaging data/TN-Library_imaging/10x_data/Results/Final_Re-imaging/
#         48 plate dirs matching "Plate*_Drawer* <date> <time>". The other dirs
#         in that folder (Analysis Code, Plots, Plots_Aligned, loose CSVs) are
#         skipped by the glob.
#
# MAGNIFICATION: _02 == 10x on THIS set, verified from Cytation TIFF metadata
#         (objective=10, pxToUm=0.6973). Note the suffix differs from the
#         Multi-phenotype set where 10x was _03 — slots are per-microscope, so
#         the folder name "10x_data" is NOT trusted; metadata is.
#
# NAS MIRROR (LEAN): process each plate to LOCAL fast scratch (--output-dir on
#         /mnt/data, 22 TB ext4), then rsync that plate's outputs to the NAS
#         (--nas-mirror-dir) and delete the local copy. --nas-lean drops the
#         heavy intermediates from the NAS copy (registered_raw.tif, masks.npz,
#         trackedLabels*.npz) — they are still written locally during processing
#         (colony/whole-image feature extraction needs them) and only excluded
#         from the mirror. The NAS therefore keeps exactly: numericalData/ +
#         master_*.csv + per-well biomass/feature CSVs + index.csv,
#         <well>_processed.tif, <well>_processed.mp4, <well>_overlay.mp4.
#         Trade-off: the NAS copy cannot re-run tracking/colony feats later.
#         /mnt/bridgeslab is READ-ONLY; the writable view of the same server is
#         /mnt/phenotyper, so the NAS dest lives there.
#
# PARAMS: blockDiam=101  fixedThresh=0.0250  fftStride=1  downsample=1
#         shiftThresh=250  minColonyArea=200px  propRadius=50px  workers=40
#         (machine: 56 cores / 251 GB; code hard-caps workers at 75% of cores =
#         42. Each worker is single-thread numpy — OMP/MKL/etc pinned to 1 — so
#         workers map ~1:1 to cores. 40 leaves a couple cores for the main proc,
#         per-plate rsync, and the OS. Likely ceiling is network-read bandwidth
#         off /mnt/bridgeslab, not CPU or RAM. Dial down if other jobs need room.)
#
# RUN MODE: launches detached via nohup, writes a timestamped log next to this
#         script, and returns immediately — survives SSH disconnect / closed
#         terminal. Monitor with the `tail -f` line printed on launch.
#
set -euo pipefail
cd /home/smellick/biofilm-processing

INPUT="/mnt/bridgeslab/Good imaging data/TN-Library_imaging/10x_data/Results/Final_Re-imaging"
STAGING="/mnt/data/tmp/multiphenotype-data-06-25-26/reimagingData"        # local scratch (outputDir)
NAS="/mnt/phenotyper/Sehna/multiphenotype-data-06-25-26/reimagingData"    # final NAS dest (nasMirrorDir)

mkdir -p "$STAGING"

# 48 plate dirs; quoted glob preserves the spaces in the names.
shopt -s nullglob
plates=( "$INPUT"/Plate*_Drawer* )
echo "Plates matched: ${#plates[@]}"

# Timestamped log next to this script; launch detached so the run survives an
# SSH disconnect / closed terminal. stdin from /dev/null so it can't block on a
# terminal read. nohup ignores SIGHUP; `&` returns the shell immediately.
TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/smellick/biofilm-processing/scripts/run_reimaging_${TS}.log"

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
