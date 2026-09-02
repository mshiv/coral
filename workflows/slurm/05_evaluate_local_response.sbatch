#!/bin/bash
# Evaluation-only local and paired-response metrics for one retained U-Net checkpoint.
#
# Usage:
#   sbatch workflows/slurm/05_evaluate_local_response.sbatch perkind
#   sbatch workflows/slurm/05_evaluate_local_response.sbatch combos
#   sbatch workflows/slurm/05_evaluate_local_response.sbatch siting_targeted
#   sbatch workflows/slurm/05_evaluate_local_response.sbatch holdout_kind_floodwall
#   sbatch workflows/slurm/05_evaluate_local_response.sbatch holdout_kind_living_shoreline
#
# This does not train or alter the checkpoint.
#SBATCH --job-name=coral-local-response
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p cpu-small
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=coral-local-response_%j.log

set -euo pipefail

CORAL_ROOT=${CORAL_ROOT:-/storage/home/hcoda1/6/smurugan9/data/dev/coral}
SCRATCH_ROOT=${SCRATCH_ROOT:-/storage/scratch1/6/smurugan9/coral}
STEM=${1:?usage: 05_evaluate_local_response.sbatch <checkpoint-stem>}
ENSEMBLE=${2:-pp4_e01}

# The code checkout has historically lived under home while the retained checkpoints live on
# project storage.  Permit explicit override and otherwise select the first root containing both
# artifacts, so the job cannot silently evaluate a different or missing checkpoint copy.
if [ -n "${EMULATOR_REPORT_ROOT:-}" ]; then
  REPORT_ROOT="$EMULATOR_REPORT_ROOT"
else
  REPORT_ROOT=""
  for candidate in \
    "$CORAL_ROOT/reports/emulator/pp4_e01" \
    /storage/project/r-arobel3-0/smurugan9/dev/coral/reports/emulator/pp4_e01
  do
    if [ -s "$candidate/emulator_unet_${STEM}.pt" ] && \
       [ -s "$candidate/emulator_unet_${STEM}.report.json" ]; then
      REPORT_ROOT="$candidate"
      break
    fi
  done
fi
test -n "$REPORT_ROOT" || {
  echo "could not find checkpoint and report for $STEM; set EMULATOR_REPORT_ROOT" >&2
  exit 1
}

OUTDIR="$REPORT_ROOT/local_response"
CHECKPOINT="$REPORT_ROOT/emulator_unet_${STEM}.pt"
SELECTION="$REPORT_ROOT/emulator_unet_${STEM}.report.json"
MANIFEST="$SCRATCH_ROOT/runs/$ENSEMBLE/manifest.json"
CACHE="$SCRATCH_ROOT/interim/grid_cache_${ENSEMBLE}"

module load anaconda3 cuda 2>/dev/null
source activate coral 2>/dev/null || conda activate coral 2>/dev/null
cd "$CORAL_ROOT"
export PYTHONPATH="$CORAL_ROOT/src:${PYTHONPATH:-}"

test -s "$CHECKPOINT" || { echo "missing checkpoint: $CHECKPOINT"; exit 1; }
test -s "$SELECTION" || { echo "missing selection report: $SELECTION"; exit 1; }
test -s "$MANIFEST" || { echo "missing manifest: $MANIFEST"; exit 1; }
mkdir -p "$OUTDIR" "$CACHE"

EXTRA_ARGS=()
OUTPUT_SUFFIX=""
if [ -n "${LIMIT:-}" ]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
  OUTPUT_SUFFIX="_smoke${LIMIT}"
  echo "smoke test: limiting evaluation to $LIMIT members"
fi

python -u -m coral.analysis.emulator_local_response \
  --ckpt "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --selection-report "$SELECTION" \
  --cache "$CACHE" \
  --cell-m 4 \
  --radii-m 50 100 250 \
  --wet-threshold 0.10 \
  --response-threshold 0.01 \
  --analysis-elevation-cutoff 0.81 \
  "${EXTRA_ARGS[@]}" \
  --out-json "$OUTDIR/${STEM}${OUTPUT_SUFFIX}.json" \
  --out-csv "$OUTDIR/${STEM}${OUTPUT_SUFFIX}.csv"

echo "completed $STEM"
