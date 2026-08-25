#!/bin/bash
# Expanded, evaluation-only metrics for an existing checkpoint. Preserves the original
# holdout membership from its training report; it never fits or updates model weights.
#
#   sbatch workflows/slurm/04_evaluate_emulator.sbatch perkind
#   sbatch workflows/slurm/04_evaluate_emulator.sbatch combos
#   sbatch workflows/slurm/04_evaluate_emulator.sbatch siting_targeted
#   sbatch workflows/slurm/04_evaluate_emulator.sbatch perkind pp4_ood none ood_perkind
#
# Args: checkpoint stem, ensemble name, selection report (or none), output stem.
#SBATCH --job-name=coral-eval
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=coral-eval_%j.log

set -euo pipefail
export CORAL=/storage/home/hcoda1/6/smurugan9/data/dev/coral
export SCR=/storage/home/hcoda1/6/smurugan9/scratch/coral

STEM=${1:?usage: 04_evaluate_emulator.sbatch <checkpoint-stem> [ensemble] [selection-report|none] [output-stem]}
ENS=${2:-pp4_e01}
OUTDIR="$CORAL/reports/emulator/pp4_e01"
CKPT="$OUTDIR/emulator_unet_${STEM}.pt"
SELECT=${3:-$OUTDIR/emulator_unet_${STEM}.report.json}
OUTSTEM=${4:-$STEM}
MANIFEST="$SCR/runs/$ENS/manifest.json"
CACHE="$SCR/interim/grid_cache_pp4_e01"

module load anaconda3 cuda 2>/dev/null
source activate coral 2>/dev/null || conda activate coral 2>/dev/null
cd "$CORAL"

test -s "$CKPT" || { echo "missing checkpoint: $CKPT"; exit 1; }
test -s "$MANIFEST" || { echo "missing manifest: $MANIFEST"; exit 1; }
ARGS=()
if [ "$SELECT" != none ]; then
  test -s "$SELECT" || { echo "missing selection report: $SELECT"; exit 1; }
  ARGS+=(--selection-report "$SELECT")
fi

python -u -m coral.emulator.evaluate_checkpoint \
  --ckpt "$CKPT" \
  --manifest "$MANIFEST" \
  "${ARGS[@]}" \
  --cache "$CACHE" \
  --workers 8 \
  --threshold 0.10 \
  --cell-m 4 \
  --out "$OUTDIR/detailed_${OUTSTEM}.report.json" \
  --out-csv "$OUTDIR/detailed_${OUTSTEM}.csv"

echo "wrote detailed_${OUTSTEM}.report.json and .csv"
