#!/usr/bin/env bash
#SBATCH --job-name=coral-diagnostics
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=coral-diagnostics_%j.log
#SBATCH --error=coral-diagnostics_%j.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
bash workflows/build_chapter_diagnostics.sh "$@"
