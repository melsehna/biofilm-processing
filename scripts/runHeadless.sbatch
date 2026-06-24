#!/bin/bash
#SBATCH --job-name=biofilmRun
#SBATCH --partition=work
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/smellick/biofilm-processing/scripts/biofilmRun_%j.log

# Full processing pipeline (registration → segmentation → tracking → features
# → master CSVs → UMAP), same as the GUI Run tab, with no display.
# Edit CONFIG / OUTDIR / plate paths for your run.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate phenotypr2

CONFIG=/home/smellick/ImageLibrary/experiment_config.json
OUTDIR=/scratch/$USER/biofilmRun_${SLURM_JOB_ID}

# scancel sends SIGTERM, which the runner traps for a cooperative stop.
biofilm-processing-run "$CONFIG" \
    --output-dir "$OUTDIR" \
    --mag _03 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --whole-image --colony-tracking --colony-feats

echo "Done. Outputs under $OUTDIR"
