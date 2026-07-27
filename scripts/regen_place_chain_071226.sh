#!/usr/bin/env bash
#
# Chained regenerate -> place -> cleanup for the two masks+LABELS lean sets
# (reimaging, then cluster). Runs each dataset end-to-end, sequentially, in one
# job — no per-dataset kickoffs. Masks + per-colony labels, bit-exact params.
#
# Per dataset:  generate (masks + tracking) to LOCAL scratch  ->  place_masks.py
# --labels --apply (create-only, audited)  ->  delete scratch (only if placement
# succeeded). A failure in one dataset skips its placement, keeps its scratch,
# and moves on — it never touches the NAS beyond create-only placement.
#
# NOTE: run AFTER multispecies finishes (avoids 80-worker oversubscription on a
# 56-core box). NAS is only ever written by place_masks.py (create-only).
#
set -uo pipefail
cd /home/smellick/biofilm-processing
source ~/anaconda3/etc/profile.d/conda.sh; conda activate phenotypr2

PLACE=scripts/place_masks.py

run_one() {
    local name="$1"; local regen="$2"; local nasroot="$3"; shift 3
    local -a plates=( "$@" )
    echo "══════════════════ $name : ${#plates[@]} plates ══════════════════"
    rm -rf "$regen"; mkdir -p "$regen"

    python scripts/runHeadless.py --plates "${plates[@]}" \
        --output-dir "$regen" --mag _02 --workers 40 \
        --block-diam 101 --fixed-thresh 0.0250 --fft-stride 1 --downsample 1 --shift-thresh 250 \
        --min-colony-area 200 --prop-radius 50 \
        --colony-tracking --no-colony-feats --no-whole-image --no-overlays --no-processed-video
    local rc=$?
    if [ $rc -ne 0 ]; then echo "!! $name generation FAILED (rc=$rc); skipping placement, keeping scratch."; return 1; fi

    local nm nt
    nm=$(find "$regen" -name "*_masks.npz" | wc -l)
    nt=$(find "$regen" -name "*_trackedLabels*.npz" | wc -l)
    echo "$name generated: masks=$nm  labels=$nt"

    echo "--- $name placement DRY-RUN ---"
    python "$PLACE" --src-root "$regen" --nas-root "$nasroot" --labels
    echo "--- $name placement APPLY ---"
    if python "$PLACE" --src-root "$regen" --nas-root "$nasroot" --labels --apply; then
        echo "$name placed OK; freeing scratch."
        rm -rf "$regen"
    else
        echo "!! $name placement did not complete cleanly; keeping scratch $regen for inspection."
        return 1
    fi
}

# ---- reimaging (48 plates) ----
RE_IN="/mnt/bridgeslab/Good imaging data/TN-Library_imaging/10x_data/Results/Final_Re-imaging"
shopt -s nullglob
re_plates=( "$RE_IN"/Plate*_Drawer* )
run_one "reimaging" "/mnt/data/tmp/regen/reimaging" \
    "/mnt/phenotyper/Sehna/multiphenotype-data-06-25-26/reimagingData" "${re_plates[@]}"

# ---- cluster (8 plates: direct nested _Plate children) ----
CL_IN="/mnt/bridgeslab/Jesse/Project - Cluster"
CL_EXP=( "260414_CmpdTreatment_V1" "260508_robot_cleanDeletion_1" "260509_robot_cleanDeletion_2"
         "260509_robot_cmpdTreatments" "260512_hand_cleanDeletion_3" "260521_hand_cleanDeletion_1"
         "260522_hand_cleanDeletion_2" "260522_hand_cmpdTreatment_1" )
cl_plates=()
for e in "${CL_EXP[@]}"; do
    for pd in "$CL_IN/$e"/*_Plate\ *; do [ -d "$pd" ] && cl_plates+=( "$pd" ); done
done
run_one "cluster" "/mnt/data/tmp/regen/cluster" \
    "/mnt/phenotyper/Sehna/cluster-data-062726/clusterData" "${cl_plates[@]}"

echo "══════════════════ CHAIN COMPLETE ══════════════════"
