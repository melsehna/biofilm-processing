#!/usr/bin/env bash
#
# Regenerate MASKS + PER-COLONY LABELS for the TN-Library reimaging lean set,
# for the BioImage Archive deposit. Derived from run_reimaging_062526.sh — SAME
# plate resolution and SAME processing + tracking params, so masks and labels
# reproduce bit-exact (proven on the training set with prop_radius=50).
#
# WHY: the original run used --nas-lean, dropping _masks.npz and
# _trackedLabels*.npz from the NAS copy. We regenerate to LOCAL scratch, then
# scripts/place_masks.py --labels creates ONLY those files next to the existing
# _processed.tif on the NAS (create-only, never overwrites).
#
# DIFFERENCES vs the original run script:
#   * output -> local scratch only; NO --nas-mirror-dir / --nas-lean
#   * --no-colony-feats --no-whole-image --no-overlays --no-processed-video
#     (keep --colony-tracking: we DO want the per-colony labels here)
#   * DRY-RUN by default: prints the plan; launches only with `--go`
#
# PARAMS (10x, from run_reimaging_062526.sh):
#   blockDiam=101 fixedThresh=0.0250 fftStride=1 downsample=1 shiftThresh=250
#   minColonyArea=200 propRadius=50
#
set -euo pipefail
cd /home/smellick/biofilm-processing

GO="${1:-}"
INPUT="/mnt/bridgeslab/Good imaging data/TN-Library_imaging/10x_data/Results/Final_Re-imaging"
REGEN="/mnt/data/tmp/regen/reimaging"
NAS="/mnt/phenotyper/Sehna/multiphenotype-data-06-25-26/reimagingData"

shopt -s nullglob
plates=( "$INPUT"/Plate*_Drawer* )
echo "Plates matched: ${#plates[@]} (expected 48)"

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
LOG="/home/smellick/biofilm-processing/scripts/regen_masks_reimaging_${TS}.log"
nohup "${CMD[@]}" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo; echo "Regen launched detached. PID: $PID  Log: $LOG"
echo "  Monitor: tail -f \"$LOG\"   Stop: kill $PID"
