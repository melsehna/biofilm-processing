#!/usr/bin/env bash
#
# Regenerate MASKS + PER-COLONY LABELS for the "Project - Cluster" lean set
# (Jesse), for the BioImage Archive deposit. Derived from run_cluster_062726.sh
# — SAME plate resolution (direct nested "_Plate " children, avoiding the nested
# duplicate under 260512_hand_cleanDeletion_3) and SAME params, so masks and
# labels reproduce bit-exact.
#
# WHY: the original run used --nas-lean, dropping _masks.npz and
# _trackedLabels*.npz. Regenerate to LOCAL scratch, then place_masks.py --labels
# creates ONLY those files next to the existing _processed.tif (create-only).
#
# DIFFERENCES vs the original run script: output -> local scratch only; no
# --nas-* flags; --no-colony-feats --no-whole-image --no-overlays
# --no-processed-video (keep --colony-tracking). DRY-RUN by default (--go to run).
#
# PARAMS (10x, from run_cluster_062726.sh):
#   blockDiam=101 fixedThresh=0.0250 fftStride=1 downsample=1 shiftThresh=250
#   minColonyArea=200 propRadius=50
#
set -euo pipefail
cd /home/smellick/biofilm-processing

GO="${1:-}"
INPUT="/mnt/bridgeslab/Jesse/Project - Cluster"
REGEN="/mnt/data/tmp/regen/cluster"
NAS="/mnt/phenotyper/Sehna/cluster-data-062726/clusterData"

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

CMD=( python scripts/runHeadless.py
    --plates "${plates[@]}"
    --output-dir "$REGEN"
    --mag _02
    --workers 40
    --block-diam 101
    --fixed-thresh 0.0250
    --fft-stride 1
    --downsample 1
    --shift-thresh 250
    --min-colony-area 200
    --prop-radius 50
    --colony-tracking
    --no-colony-feats
    --no-whole-image
    --no-overlays
    --no-processed-video )

echo; echo "REGEN output -> $REGEN  (NAS untouched here)"
echo "Placement (separate step, after this completes):"
echo "  python scripts/place_masks.py --src-root \"$REGEN\" --nas-root \"$NAS\" --labels           # dry-run"
echo "  python scripts/place_masks.py --src-root \"$REGEN\" --nas-root \"$NAS\" --labels --apply   # create masks+labels"
echo; echo "COMMAND:"; printf '  %q' "${CMD[@]}"; echo

if [ "$GO" != "--go" ]; then
    echo; echo "[DRY-RUN] not launching. Re-run with:  $0 --go"
    exit 0
fi

mkdir -p "$REGEN"
TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/smellick/biofilm-processing/scripts/regen_masks_cluster_${TS}.log"
nohup "${CMD[@]}" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo; echo "Regen launched detached. PID: $PID  Log: $LOG"
echo "  Monitor: tail -f \"$LOG\"   Stop: kill $PID"
