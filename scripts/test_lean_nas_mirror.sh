#!/usr/bin/env bash
# Test the lean NAS-mirror mode on 3 wells (A1, A2, A3 at 10x) from Drawer1.
#
# Creates a symlinked test plate dir so only 3 wells run instead of all 96.
# Stages locally at /mnt/data/tmp/lean_test_staging/, mirrors to
# /mnt/phenotyper/Sehna/.lean_test/ with --nas-lean, then prints what landed
# on the NAS and confirms the excluded intermediates are absent.
#
# Usage:  bash scripts/test_lean_nas_mirror.sh

set -euo pipefail

REAL_PLATE="/mnt/bridgeslab/Good imaging data/Multi-phenotype training/241010_105227_4x_10x_20x_40x_Discontinuous_Drawer1 10-Oct-2024 10-48-54/241010_105227_Plate 1"
TEST_PLATE_DIR="/mnt/data/tmp/lean_test_plate"
TEST_PLATE="$TEST_PLATE_DIR/241010_105227_Plate_1_test3wells"
LOCAL_STAGING="/mnt/data/tmp/lean_test_staging"
NAS_MIRROR="/mnt/phenotyper/Sehna/.lean_test"
WELLS="A1 A2 A3"

# ── 1. Build symlinked test plate ──────────────────────────────────────────────
echo "=== Building symlinked test plate: $TEST_PLATE ==="
rm -rf "$TEST_PLATE"
mkdir -p "$TEST_PLATE"
for well in $WELLS; do
    count=0
    while IFS= read -r f; do
        ln -sf "$f" "$TEST_PLATE/$(basename "$f")"
        (( count++ )) || true
    done < <(find "$REAL_PLATE" -maxdepth 1 -name "${well}_03_*")
    echo "  $well: $count frames linked"
done

# ── 2. Clean up any previous staging run ──────────────────────────────────────
PLATE_NAME=$(basename "$TEST_PLATE")
rm -rf "$LOCAL_STAGING/$PLATE_NAME"
mkdir -p "$LOCAL_STAGING"

# ── 3. Run pipeline ────────────────────────────────────────────────────────────
echo ""
echo "=== Running pipeline (lean NAS mirror) ==="
biofilm-processing-run \
    --plates "$TEST_PLATE" \
    --output-dir "$LOCAL_STAGING" \
    --mag _03 \
    --workers 4 \
    --block-diam 101 \
    --fixed-thresh 0.025 \
    --fft-stride 1 \
    --downsample 1 \
    --shift-thresh 250 \
    --whole-image \
    --colony-tracking \
    --colony-feats \
    --overlays \
    --processed-video \
    --min-colony-area 200 \
    --prop-radius 50 \
    --nas-mirror-dir "$NAS_MIRROR" \
    --nas-lean

# ── 4. Verify NAS artifacts ────────────────────────────────────────────────────
echo ""
echo "=== Verify NAS artifacts ==="
NAS_PROC="$NAS_MIRROR/$PLATE_NAME/processedImages"

count_nas() {
    # Use find so the glob in $1 expands correctly regardless of shell quoting.
    local dir pat
    dir=$(dirname "$1"); pat=$(basename "$1")
    find "$dir" -maxdepth 1 -name "$pat" 2>/dev/null | wc -l
}

echo "KEEP (expect 3 each):"
printf "  processed tifs:     %d\n" "$(count_nas "$NAS_PROC/*_processed.tif")"
printf "  processed fpHalf:   %d\n" "$(count_nas "$NAS_PROC/*_processed_fpHalf.tif")"
printf "  overlay mp4s:       %d\n" "$(count_nas "$NAS_PROC/*_overlay.mp4")"
printf "  processed mp4s:     %d\n" "$(count_nas "$NAS_PROC/*_processed.mp4")"
printf "  biomass csvs:       %d\n" "$(count_nas "$NAS_PROC/*_biomass.csv")"
printf "  colony feat csvs:   %d\n" "$(count_nas "$NAS_PROC/*_perColonyFeatures.csv")"
printf "  well colony csvs:   %d\n" "$(count_nas "$NAS_PROC/*_wellColonyFeatures.csv")"
printf "  whole image csvs:   %d\n" "$(count_nas "$NAS_PROC/*_wholeImageFeatures.csv")"
printf "  index.csv:          %d\n" "$(count_nas "$NAS_PROC/index.csv")"
printf "  run_params.json:    %d\n" "$(count_nas "$NAS_PROC/run_params.json")"
echo ""
echo "SKIP (lean mode — expect 0 each):"
printf "  registered_raw tif: %d\n" "$(count_nas "$NAS_PROC/*_registered_raw.tif")"
printf "  masks npz:          %d\n" "$(count_nas "$NAS_PROC/*_masks.npz")"
printf "  tracked labels npz: %d\n" "$(count_nas "$NAS_PROC/*_trackedLabels*.npz")"
echo ""
echo "NUMERICAL DATA (expect > 0):"
NAS_NUM="$NAS_MIRROR/$PLATE_NAME/numericalData"
printf "  numericalData dir:  %s\n" "$(ls "$NAS_NUM" 2>/dev/null | tr '\n' '  ' || echo '(missing)')"
echo ""
echo "LOCAL staging still present? (should be gone after NAS sync):"
if ls "$LOCAL_STAGING/$PLATE_NAME" 2>/dev/null; then
    echo "  PRESENT — cleanup failed or NAS sync failed"
else
    echo "  gone (correct)"
fi
echo ""
echo "MASTER CSV at NAS root:"
ls "$NAS_MIRROR"/master_*.csv 2>/dev/null || echo "  (none)"
