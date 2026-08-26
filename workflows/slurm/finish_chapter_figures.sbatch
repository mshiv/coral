#!/usr/bin/env bash
#SBATCH --job-name=coral-final-figures
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=coral-final-figures_%j.log
#SBATCH --error=coral-final-figures_%j.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
bash workflows/finish_chapter_figures.sh "$@"
