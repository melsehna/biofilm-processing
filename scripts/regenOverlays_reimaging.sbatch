#!/bin/bash
#SBATCH --job-name=regenOverlaysReimaging
#SBATCH --partition=work
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/smellick/biofilm-processing/scripts/regenOverlays_reimaging_%j.log

source ~/anaconda3/etc/profile.d/conda.sh
conda activate phenotypr2

python /home/smellick/biofilm-processing/scripts/regenOverlaysFromIndex.py \
    /mnt/data/indices/reimaging/reimagingIndex_withAllFeatFrameSummaries.csv \
    --outdir /mnt/phenotyper/vcReimaging \
    --fps 8 \
    --workers 40 \
    2>/dev/null

# Clean up any temp mp4 files left in /tmp
find /tmp -maxdepth 1 -name "*.mp4" -user "$USER" -delete 2>/dev/null
echo "Cleaned up temp mp4 files from /tmp"
