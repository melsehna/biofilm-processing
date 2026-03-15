#!/bin/bash
set -e

BASEDIR="/mnt/bridgeslab/Good imaging data/TN-Library_imaging"
OUTDIR="/mnt/data/transposonSet"
MAG="_03"
WORKERS=16

PLATES=(
  'TN-Plate01_4x_10x_20x_40x_Discontinuous_Drawer1 06-Nov-2024 14-57-44'
  'TN-Plate02_4x_10x_20x_40x_Discontinuous_Drawer2 25-Oct-2024 17-17-58'
  'TN-Plate03_4x_10x_20x_40x_Discontinuous_Drawer2 29-Oct-2024 14-46-08'
  'TN-Plate04_4x_10x_20x_40x_Discontinuous_Drawer3 29-Oct-2024 14-46-35'
  'TN-Plate07_4x_10x_20x_40x_Discontinuous_Drawer3 31-Oct-2024 13-36-05'
  'TN-Plate08_4x_10x_20x_40x_Discontinuous_Drawer1 04-Nov-2024 15-56-04'
  'TN-Plate09_4x_10x_20x_40x_Discontinuous_Drawer2 04-Nov-2024 15-56-34'
  'TN-Plate10_4x_10x_20x_40x_Discontinuous_Drawer3 04-Nov-2024 15-57-04'
  'TN-Plate11_4x_10x_20x_40x_Discontinuous_Drawer2 06-Nov-2024 14-58-16'
  'TN-Plate12_4x_10x_20x_40x_Discontinuous_Drawer3 06-Nov-2024 14-58-50'
  'TN-Plate13_4x_10x_20x_40x_Discontinuous_Drawer1 20-Nov-2024 16-25-11'
  'TN-Plate14_4x_10x_20x_40x_Discontinuous_Drawer2 20-Nov-2024 16-25-39'
  'TN-Plate15_4x_10x_20x_40x_Discontinuous_Drawer3 20-Nov-2024 16-47-34'
  'TN-Plate16_4x_10x_20x_40x_Discontinuous_Drawer1 11-Nov-2024 16-06-24'
  'TN-Plate17_4x_10x_20x_40x_Discontinuous_Drawer2 11-Nov-2024 16-06-55'
  'TN-Plate18_4x_10x_20x_40x_Discontinuous_Drawer3 11-Nov-2024 16-07-25'
  'TN-Plate19_4x_10x_20x_40x_Discontinuous_Drawer1 18-Nov-2024 11-46-38'
  'TN-Plate20_4x_10x_20x_40x_Discontinuous_Drawer2 18-Nov-2024 11-46-59'
  'TN-Plate21_4x_10x_20x_40x_Discontinuous_Drawer3 18-Nov-2024 12-47-49'
  'TN-Plate22_4x_10x_20x_40x_Discontinuous_Drawer1 12-Dec-2024 15-50-10'
  'TN-Plate23_4x_10x_20x_40x_Discontinuous_Drawer2 12-Dec-2024 15-50-38'
  'TN-Plate24_4x_10x_20x_40x_Discontinuous_Drawer2 19-Dec-2024 15-41-47'
  'TN-Plate25_4x_10x_20x_40x_Discontinuous_Drawer3 25-Nov-2024 15-53-17'
  'TN-Plate26_4x_10x_20x_40x_Discontinuous_Drawer4 25-Nov-2024 16-15-04'
  'TN-Plate27_4x_10x_20x_40x_Discontinuous_Drawer1 27-Nov-2024 13-22-12'
  'TN-Plate28_4x_10x_20x_40x_Discontinuous_Drawer2 27-Nov-2024 13-22-40'
  'TN-Plate29_4x_10x_20x_40x_Discontinuous_Drawer3 27-Nov-2024 13-23-07'
  'TN-Plate30_4x_10x_20x_40x_Discontinuous_Drawer1 04-Dec-2024 15-50-29'
  'TN-Plate31_4x_10x_20x_40x_Discontinuous_Drawer2 04-Dec-2024 15-50-59'
  'TN-Plate32_4x_10x_20x_40x_Discontinuous_Drawer3 04-Dec-2024 15-51-30'
  'TN-Plate33_4x_10x_20x_40x_Discontinuous_Drawer1 06-Dec-2024 16-04-05'
  'TN-Plate34_4x_10x_20x_40x_Discontinuous_Drawer2 06-Dec-2024 16-04-34'
)

mkdir -p "$OUTDIR"

cd ~/biofilm-processing/testProcPrototype

TOTAL=${#PLATES[@]}
for i in "${!PLATES[@]}"; do
  OUTER="${BASEDIR}/${PLATES[$i]}"
  PLATEDIR=$(find "$OUTER" -mindepth 1 -maxdepth 1 -type d -name '*Plate*' | head -1)

  if [ -z "$PLATEDIR" ]; then
    echo "[$((i+1))/$TOTAL] SKIP: no Plate subdir found in $OUTER"
    continue
  fi

  PLATE_NAME=$(echo "${PLATES[$i]}" | grep -oP 'TN-Plate\d+')
  echo ""
  echo "[$((i+1))/$TOTAL] $PLATE_NAME"
  echo "  Source: $PLATEDIR"
  echo "  Output: $OUTDIR"

  python runSinglePlateTest.py "$PLATEDIR" -o "$OUTDIR" -m "$MAG" -w "$WORKERS"
done

echo ""
echo "All plates complete. Output in: $OUTDIR"
