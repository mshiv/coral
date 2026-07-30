#!/bin/bash
# Train the flood emulator on the finished 30 m ensemble.
#
# Run this after 01_prep_ensemble.sbatch and after the run array has produced .max files.
# NOTE: 
# It does not need every member: train_ensemble.py reports coverage and trains on whatever
# has finished. Run it at a few 100 members so the pipeline can be confirmed. 
#
#   sbatch workflows/slurm/02_train_emulator.sbatch                 # holdout sea levels
#   sbatch workflows/slurm/02_train_emulator.sbatch valid           # validate on valid30m
#   sbatch workflows/slurm/02_train_emulator.sbatch curve           # learning curve sweep
#SBATCH --job-name=coral-train
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=coral-train_%j.log

set -uo pipefail

export CORAL=/storage/home/hcoda1/6/smurugan9/data/dev/coral
export SCR=/storage/home/hcoda1/6/smurugan9/scratch/coral
export BASE=$SCR/runs/compound30m
TRAIN=$SCR/runs/train30m
VALID=$SCR/runs/valid30m
MODE=${1:-holdout}

log() { echo "[$(date +%H:%M:%S)] $*"; }
ok()  { echo "  [OK]   $*"; }
bad() { echo "  [FAIL] $*"; }

echo "============ CORAL emulator training, job $SLURM_JOB_ID on $(hostname) ============"
log "mode=$MODE"

module load anaconda3 cuda 2>/dev/null
source activate coral 2>/dev/null || conda activate coral 2>/dev/null
cd "$CORAL" || exit 1

echo; echo "=== preconditions ==="
[ -s "$TRAIN/manifest.json" ] && ok "training manifest" || { bad "run 01_prep_ensemble first"; exit 1; }
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { bad "torch missing: pip install -e '.[emulator]'"; exit 1; }
NDONE=$(ls "$TRAIN"/*/results_matthew_sav/res_matthew_sav.max 2>/dev/null | wc -l | tr -d ' ')
NPLAN=$(wc -l < "$TRAIN/run_dirs.txt")
log "$NDONE of $NPLAN members finished"
[ "$NDONE" -ge 50 ] && ok "enough members to train on" \
  || { bad "only $NDONE members done; let the array run further"; exit 1; }

OUTDIR=$CORAL/reports/emulator
mkdir -p "$OUTDIR"

echo; echo "=== training ==="
case "$MODE" in
  holdout)
    # Hold out whole sea levels. A random split would leak: 250 members share each level,
    # so near-duplicates of every held-out case would sit in the training set.
    python workflows/train_ensemble.py \
      --ensemble "$TRAIN" \
      --holdout-slr IntHigh2100 High2100 \
      --epochs 300 --patience 10 --workers 8 \
      --ckpt "$OUTDIR/emulator_unet_holdout.pt"
    ;;
  valid)
    # The stronger test: train on random placements, validate on the targeted ensemble.
    [ -s "$VALID/manifest.json" ] || { bad "no $VALID; build it with 01_prep_ensemble pinpoint_valid30m valid30m"; exit 1; }
    python workflows/train_ensemble.py \
      --ensemble "$TRAIN" --val-ensemble "$VALID" \
      --epochs 300 --patience 10 --workers 8 \
      --ckpt "$OUTDIR/emulator_unet_targeted.pt"
    ;;
  curve)
    # Does accuracy still improve with ensemble size? This is what decides whether to raise
    # n_per_kind from 50 to 100, which costs another 124 GB and about 12 hours of runs.
    python workflows/train_ensemble.py \
      --ensemble "$TRAIN" \
      --holdout-slr Int2070 \
      --subsample-per-kind 5 10 25 50 \
      --epochs 200 --patience 8 --workers 8 \
      --ckpt "$OUTDIR/emulator_unet_curve.pt"
    ;;
  *) bad "unknown mode $MODE (use holdout, valid, or curve)"; exit 1 ;;
esac
RC=$?

echo; echo "=== results ==="
if [ "$RC" -eq 0 ]; then
  ok "training finished"
  ls -la "$OUTDIR" | sed 's/^/  /'
  echo
  echo "Per-member held-out metrics are in the .report.json next to each checkpoint."
  echo "Turn a checkpoint into a depth raster:"
  echo "  python -m coral.emulator.inference --ckpt $OUTDIR/emulator_unet_holdout.pt \\"
  echo "      --run \$(head -1 $TRAIN/run_dirs.txt) --out reports/emulator/pred.asc"
else
  bad "training returned $RC; search this log for a traceback"
fi
exit $RC
