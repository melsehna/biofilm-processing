#!/bin/bash
# Example SLURM submission. Copy and edit — it will not run unmodified.
#SBATCH --job-name=biofilmRun
#SBATCH --partition=CHANGEME_PARTITION
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=biofilmRun_%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biofilm-processing

CONFIG=/path/to/experiment_config.json
OUTDIR=/path/to/scratch/biofilmRun_${SLURM_JOB_ID}

# scancel sends SIGTERM, which the runner traps to stop between wells/stages
biofilm-processing-run "$CONFIG" \
    --output-dir "$OUTDIR" \
    --mag _03 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --whole-image --colony-tracking --colony-feats

# Or without a config file:
# biofilm-processing-run --plates /path/to/plateA /path/to/plateB \
#     --output-dir "$OUTDIR" --mag _03 --workers "$SLURM_CPUS_PER_TASK" \
#     --block-diam 101 --fixed-thresh 0.04 --whole-image --colony-tracking

echo "Done. Outputs under $OUTDIR"
