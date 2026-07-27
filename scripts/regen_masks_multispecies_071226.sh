#!/usr/bin/env bash
#
# Regenerate MASKS ONLY (no colony tracking) for Sophia's multispecies LB lean
# set (6-16 + 6-24), for the BioImage Archive deposit. Derived from
# run_sophia_multispecies_062926.sh — SAME unique-name symlink resolution and
# SAME per-mag params (via sophia_multispecies_config.json), so masks reproduce
# bit-exact. Per-colony labels intentionally NOT regenerated.
#
# WHY: the original run used --nas-lean, dropping _masks.npz. Regenerate to LOCAL
# scratch, then place_masks.py creates ONLY the _masks.npz next to the existing
# _processed.tif on the NAS (create-only, never overwrites).
#
# DIFFERENCES vs the original run script: output -> local scratch only; no
# --nas-* flags; --no-colony-tracking --no-colony-feats --no-whole-image
# --no-overlays --no-processed-video. DRY-RUN by default (--go to launch).
#
# PARAMS: --mag all; per-mag blockDiam/fixedThresh/dust from the config
#   (_03=4x thr0.04, _04=10x thr0.03). Global: fftStride=1 downsample=4 shift=250.
#
set -euo pipefail
cd /home/smellick/biofilm-processing

GO="${1:-}"
CYT="/mnt/bridgeslab/Sophia/cytation"
declare -A IMGDIRS=(
    ["6-16"]="$CYT/6-16_LBDilutes_and_BHI-TSB_EFvSA/LB_raw_images"
    ["6-24"]="$CYT/6-24_100v10_LB/raw"
)
LINKS="/mnt/data/tmp/regen/multispecies/inputLinks"
REGEN="/mnt/data/tmp/regen/multispecies/staging"
NAS="/mnt/phenotyper/Sehna/multispecies-data-062926"
CONFIG="/home/smellick/biofilm-processing/scripts/sophia_multispecies_config.json"

mkdir -p "$LINKS"

# Uniquely-named symlinks (<date>_<DrawerN>) -> each drawer's nested plate dir —
# identical to the original run (the nested plate dirs share a name, hence links).
shopt -s nullglob
plates=()
for exp in "${!IMGDIRS[@]}"; do
    for drawer in "${IMGDIRS[$exp]}"/*_Drawer*; do
        [ -d "$drawer" ] || continue
        base=$(basename "$drawer")
        date=${base%%_*}
        dn=$(grep -oE 'Drawer[0-9]+' <<<"$base")
        name="${date}_${dn}"
        nested=$(find "$drawer" -maxdepth 1 -mindepth 1 -type d -name '*_Plate *' | head -1)
        if [ -z "$nested" ]; then echo "WARNING: no nested _Plate dir in $drawer" >&2; continue; fi
        ln -sfn "$nested" "$LINKS/$name"
        plates+=("$LINKS/$name")
    done
done
echo "Plates (unique-named symlinks): ${#plates[@]} (expected 6)"
for p in "${plates[@]}"; do echo "  $(basename "$p") -> $(readlink "$p")"; done

CMD=( python scripts/runHeadless.py "$CONFIG"
    --plates "${plates[@]}"
    --output-dir "$REGEN"
    --mag all
    --workers 40
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
LOG="/home/smellick/biofilm-processing/scripts/regen_masks_multispecies_${TS}.log"
nohup "${CMD[@]}" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo; echo "Regen launched detached. PID: $PID  Log: $LOG"
echo "  Monitor: tail -f \"$LOG\"   Stop: kill $PID"
