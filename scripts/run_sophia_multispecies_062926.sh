#!/usr/bin/env bash
#
# Headless full-pipeline run — Sophia multispecies LB sets (6-16 + 6-24 combined).
# Authored 2026-06-29. Reproducible record.
#
# DATA: two experiments, 3 plates each (6 total), 10x AND 4x, 96 wells, 24 frames.
#   6-16  /mnt/bridgeslab/Sophia/cytation/6-16_LBDilutes_and_BHI-TSB_EFvSA/LB_raw_images/<Drawer4,5,6>
#   6-24  /mnt/bridgeslab/Sophia/cytation/6-24_100v10_LB/raw/<Drawer6,7,8>
#   Each drawer holds a nested "<date>_Plate 1!PLATE_ID!_" dir (Cytation unfilled
#   template) with the TIFFs. Multispecies layout (8 species x 100%/10% LB) is in
#   ~/biofilm-analysis (uPULLI) control_multispecies_100v10LB_cytation_plates.xlsx.
#
# MAGNIFICATION: _04 = 10x (pxToUm 0.6973), _03 = 4x (pxToUm 1.744). Both processed
#   (--mag all). Verified from Cytation TIFF metadata (slot mapping differs from the
#   TN-Library sets, where 10x was _02).
#
# UNIQUE PLATE IDs (important): all 3 plates within an experiment share the SAME
#   nested folder name ("260616_Plate 1!PLATE_ID!_"), so passing them directly
#   would collide (output dirs + master CSV). We instead create uniquely-named
#   SYMLINKS (<date>_<DrawerN>) to each nested plate dir and pass those — the
#   resolver uses the path as given, so the symlink name becomes the plate id.
#
# PER-MAG PARAMS: one --mag all pass (required for lean NAS: each plate is synced+
#   deleted once). Per-mag blockDiam/fixedThresh/minColonyAreaPx/propRadiusPx come
#   from --config sophia_multispecies_config.json (magParams). TUNE per mag with
#   biofilm-processing-test-well, fill that JSON, THEN run. Registration params
#   (shift-thresh/fft-stride/downsample) are global below.
#
# >>> EDIT/CONFIRM the OUTPUT paths below, and fill sophia_multispecies_config.json
#     after tuning, before running. <<<
#
set -euo pipefail
cd /home/smellick/biofilm-processing

CYT="/mnt/bridgeslab/Sophia/cytation"
declare -A IMGDIRS=(
    ["6-16"]="$CYT/6-16_LBDilutes_and_BHI-TSB_EFvSA/LB_raw_images"
    ["6-24"]="$CYT/6-24_100v10_LB/raw"
)

LINKS="/mnt/data/tmp/sophia-multispecies-062926/inputLinks"     # writable dir for unique-name symlinks
STAGING="/mnt/data/tmp/sophia-multispecies-062926/staging"      # local scratch (outputDir)
# Final NAS dest. The user-named path /mnt/bridgeslab/phenotyper/Sehna/... is the
# READ-ONLY mount of this same share on this machine; /mnt/phenotyper/Sehna is the
# writable view of the identical server dir. Auto-created on first sync.
NAS="/mnt/phenotyper/Sehna/multispecies-data-062926"
CONFIG="/home/smellick/biofilm-processing/scripts/sophia_multispecies_config.json"

mkdir -p "$LINKS" "$STAGING"

# Build uniquely-named symlinks (<date>_<DrawerN>) -> each drawer's nested plate dir.
shopt -s nullglob
plates=()
for exp in "${!IMGDIRS[@]}"; do
    for drawer in "${IMGDIRS[$exp]}"/*_Drawer*; do
        [ -d "$drawer" ] || continue
        base=$(basename "$drawer")
        date=${base%%_*}                                   # 260616 / 260624
        dn=$(grep -oE 'Drawer[0-9]+' <<<"$base")            # Drawer4 ...
        name="${date}_${dn}"                                # 260616_Drawer4 (unique across both)
        nested=$(find "$drawer" -maxdepth 1 -mindepth 1 -type d -name '*_Plate *' | head -1)
        if [ -z "$nested" ]; then echo "WARNING: no nested _Plate dir in $drawer" >&2; continue; fi
        ln -sfn "$nested" "$LINKS/$name"
        plates+=("$LINKS/$name")
    done
done
echo "Plates (unique-named symlinks): ${#plates[@]} (expected 6)"
for p in "${plates[@]}"; do echo "  $(basename "$p") -> $(readlink "$p")"; done

TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/smellick/biofilm-processing/scripts/run_sophia_multispecies_${TS}.log"

nohup python scripts/runHeadless.py "$CONFIG" \
    --plates "${plates[@]}" \
    --output-dir "$STAGING" \
    --nas-mirror-dir "$NAS" \
    --nas-lean \
    --mag all \
    --workers 40 \
    --fft-stride 1 \
    --downsample 4 \
    --shift-thresh 250 \
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
