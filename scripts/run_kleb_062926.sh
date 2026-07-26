#!/usr/bin/env bash
#
# Headless full-pipeline run — KLEB training set (6 mutants), 4x.
# Authored 2026-06-29; corrected 2026-06-30 to the real data location.
#
# DATA: 6 plates, 4x (_02), 96 wells, 25 frames. One K. pneumoniae transposon
#   mutant per plate (96 well replicates). 250311 pilot.
#   /mnt/bridgeslab/Drew_Kleb/KLEB training set/
#     250311_..._Drawer1_NV_058  (WT)        250311_124651_Plate 1
#     250311_..._Drawer2_NV_059  (WzcQ395K)  250311_125358_Plate 1
#     250311_..._Drawer3_NV_064  (wcaJ)      250311_130104_Plate 1
#     250311_..._Drawer4_NV_065  (galU)      250311_130813_Plate 1
#     250311_..._Drawer5_NV_066  (waaL)      250311_131518_Plate 1
#     250311_..._Drawer6_NV_070  (mrkA)      250311_132226_Plate 1
#   Distinct nested plate names (no collision); pass the drawer dirs so output is
#   organized <drawer>/<plate>/. Mutant identity = the NV_XXX in the drawer name.
#
# MAGNIFICATION: _02 == 4x (objective 4, pxToUm 1.7440), metadata-verified. Single mag.
#
# PARAMS (4x): blockDiam=101 fixedThresh=0.04 fftStride=1 downsample=4
#   shiftThresh=250 minColonyArea=200px propRadius=25px  workers=32.
#
# NAS MIRROR (LEAN): process to LOCAL scratch, rsync each plate to the NAS, delete
#   local. Lean keeps processed.tif + processed.mp4 + overlay.mp4 + numerical data.
#
set -euo pipefail
cd /home/smellick/biofilm-processing

INPUT="/mnt/bridgeslab/Drew_Kleb/KLEB training set"
STAGING="/mnt/data/tmp/kleb-data-062926/klebData"          # local scratch (outputDir)
NAS="/mnt/phenotyper/Sehna/kleb-data-062926/klebData"      # final NAS dest

mkdir -p "$STAGING"

# The 6 mutant drawer dirs (250311_4x_..._DrawerN_NV_XXX).
shopt -s nullglob
plates=()
for d in "$INPUT"/250311_4x_*Drawer*; do [ -d "$d" ] && plates+=( "$d" ); done
echo "Plates matched: ${#plates[@]} (expected 6)"
printf '  %s\n' "${plates[@]##*/}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/smellick/biofilm-processing/scripts/run_kleb_${TS}.log"

nohup python scripts/runHeadless.py \
    --plates "${plates[@]}" \
    --output-dir "$STAGING" \
    --nas-mirror-dir "$NAS" \
    --nas-lean \
    --mag _02 \
    --workers 32 \
    --block-diam 101 \
    --fixed-thresh 0.04 \
    --fft-stride 1 \
    --downsample 4 \
    --shift-thresh 250 \
    --min-colony-area 200 \
    --prop-radius 25 \
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
echo "  Stop:     kill $PID"
