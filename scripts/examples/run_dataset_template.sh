#!/usr/bin/env bash
# Template for a per-dataset run record. Copy to scripts/run_<dataset>_<MMDDYY>.sh
# and fill in the header — it is the source of truth for what the run did.
#
# DATA: <N plates, ~N wells, N frames, raw location>
# MAGNIFICATION: <_03 == 10x, verified from Cytation TIFF metadata — objective
#   slots are per-microscope, never trust the folder name>
# PARAMS: <blockDiam=101 fixedThresh=0.04 fftStride=1 workers=24>
# OUTPUTS: <what was written, and where>

set -euo pipefail
cd "$(dirname "$0")/.."

INPUT="/path/to/raw/plates"
OUT="/path/to/fast/local/scratch"
LOG="$(dirname "$0")/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT"

# nullglob + quoting: plate dir names contain spaces
shopt -s nullglob
plates=("$INPUT"/*"_Plate"*)
echo "Matched ${#plates[@]} plates (expected: <N>)"
if [ "${#plates[@]}" -eq 0 ]; then
    echo "No plates matched under $INPUT" >&2
    exit 1
fi

# If nested Cytation dirs collide (every drawer holding a "..._Plate 1"), pass the
# nested child directly or symlink to unique names — the path is used as plate id.

# nohup so the run survives disconnect; `kill <PID>` stops it cooperatively
nohup python scripts/runHeadless.py \
    --plates "${plates[@]}" \
    --output-dir "$OUT" \
    --mag _03 \
    --workers 24 \
    --block-diam 101 \
    --fixed-thresh 0.04 \
    --whole-image \
    --colony-tracking \
    --colony-feats \
    --overlays \
    < /dev/null > "$LOG" 2>&1 &

echo "Started PID $! — log: $LOG"

# To mirror to a network share, process to local scratch (above) and add
# --nas-mirror-dir <share>. Adding --nas-lean also drops _registered_raw.tif,
# _masks.npz and _trackedLabels*.npz from the mirror — halves the footprint, but
# the mirror can no longer re-run tracking or colony features.
