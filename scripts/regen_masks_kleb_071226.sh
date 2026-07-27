#!/usr/bin/env bash
#
# Regenerate MASKS ONLY (no colony tracking) for the KLEB lean set, for the
# BioImage Archive deposit. Derived from run_kleb_062926.sh — SAME plate
# resolution and SAME processing params, so masks reproduce bit-exact.
#
# WHY: the original kleb run used --nas-lean, which dropped _masks.npz from the
# NAS copy. Masks are deterministic given identical params (proven on the
# training set: 0/114M pixels differ). We regenerate to LOCAL scratch, then
# scripts/place_masks.py creates ONLY the _masks.npz next to the existing
# _processed.tif on the NAS (create-only, never overwrites). Per-colony labels
# are intentionally NOT regenerated for kleb.
#
# DIFFERENCES vs the original run script:
#   * output -> local scratch only; NO --nas-mirror-dir / --nas-lean
#   * --no-colony-tracking --no-colony-feats --no-whole-image --no-overlays
#     --no-processed-video  (we only need _masks.npz; skip the expensive tail)
#   * DRY-RUN by default: prints the plan; launches only with `--go`
#
# PARAMS (4x, from run_kleb_062926.sh + its run_params.json):
#   blockDiam=101 fixedThresh=0.04 fftStride=1 downsample=4 shiftThresh=250
#
set -euo pipefail
cd /home/smellick/biofilm-processing

GO="${1:-}"
INPUT="/mnt/bridgeslab/Drew_Kleb/KLEB training set"
REGEN="/mnt/data/tmp/regen/kleb"                       # local scratch (this run only)
NAS="/mnt/phenotyper/Sehna/kleb-data-062926/klebData"  # placement target (place_masks.py, separate step)

# The 6 mutant drawer dirs — identical resolution to the original run.
shopt -s nullglob
plates=()
for d in "$INPUT"/250311_4x_*Drawer*; do [ -d "$d" ] && plates+=( "$d" ); done
echo "Plates matched: ${#plates[@]} (expected 6)"
printf '  %s\n' "${plates[@]##*/}"

CMD=( python scripts/runHeadless.py
    --plates "${plates[@]}"
    --output-dir "$REGEN"
    --mag _02
    --workers 32
    --block-diam 101
    --fixed-thresh 0.04
    --fft-stride 1
    --downsample 4
    --shift-thresh 250
    --no-colony-tracking
    --no-colony-feats
    --no-whole-image
    --no-overlays
    --no-processed-video )

echo; echo "REGEN output -> $REGEN  (NAS untouched here)"
echo "Placement (separate step, after this completes):"
echo "  python scripts/place_masks.py --src-root \"$REGEN\" --nas-root \"$NAS\"           # dry-run"
echo "  python scripts/place_masks.py --src-root \"$REGEN\" --nas-root \"$NAS\" --apply   # create masks"
echo; echo "COMMAND:"; printf '  %q' "${CMD[@]}"; echo

if [ "$GO" != "--go" ]; then
    echo; echo "[DRY-RUN] not launching. Re-run with:  $0 --go"
    exit 0
fi

mkdir -p "$REGEN"
TS=$(date +%Y%m%d_%H%M%S)
LOG="/home/smellick/biofilm-processing/scripts/regen_masks_kleb_${TS}.log"
nohup "${CMD[@]}" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo; echo "Regen launched detached. PID: $PID  Log: $LOG"
echo "  Monitor: tail -f \"$LOG\"   Stop: kill $PID"
