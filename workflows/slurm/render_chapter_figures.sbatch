#!/usr/bin/env bash
#SBATCH --job-name=coral-figures
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=coral-figures_%j.log
#SBATCH --error=coral-figures_%j.err
set -euo pipefail
# Submit from the CORAL root with the coral environment active.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
bash workflows/build_chapter_figures.sh "$@"
