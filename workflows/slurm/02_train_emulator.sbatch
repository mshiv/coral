#!/bin/bash
# Train the flood emulator on the finished 30 m ensemble.
#
# Run this after 01_prep_ensemble.sbatch and after the run array has produced .max files.
# NOTE: 
# It does not need every member: train_ensemble.py reports coverage and trains on whatever
# has finished. Run it at a few 100 members so the pipeline can be confirmed. 
#
#   sbatch workflows/slurm/02_train_emulator.sbatch perkind         # stratified, the headline
#   sbatch workflows/slurm/02_train_emulator.sbatch combos          # singles -> portfolios
#   sbatch workflows/slurm/02_train_emulator.sbatch siting          # random -> targeted
#   sbatch workflows/slurm/02_train_emulator.sbatch                 # holdout sea levels
#   sbatch workflows/slurm/02_train_emulator.sbatch valid           # validate on valid30m
#   sbatch workflows/slurm/02_train_emulator.sbatch curve           # learning curve sweep
#   sbatch workflows/slurm/02_train_emulator.sbatch holdout-placement 20
#   sbatch workflows/slurm/02_train_emulator.sbatch curve-placement 20
#SBATCH --job-name=coral-train
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
# GPU jobs need an explicit partition; without one the job lands on a CPU partition and
# never gets a device. NOT gpu-v100: the V100 is compute capability 7.0 and this torch build
# ships kernels for >=7.5 only, so torch.cuda.is_available() returns True and then the first
# kernel launch fails. Usable here: rtx6000 (7.5), a100 (8.0), l40s (8.9), h100/h200 (9.0).
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
# 24 h, not 12. At about 0.2 s per member per epoch, a 1540-member split needs roughly five
# minutes an epoch, so a 300-epoch schedule is 27 hours and the 12 h limit truncated every run.
#SBATCH --time=24:00:00
#SBATCH --output=coral-train_%j.log

set -uo pipefail

export CORAL=/storage/home/hcoda1/6/smurugan9/data/dev/coral
export SCR=/storage/home/hcoda1/6/smurugan9/scratch/coral
export BASE=$SCR/runs/compound30m
# Ensemble to train on. Overridable so the corrected-physics ensemble can be trained without
# editing the script: train30m has neither marsh correction, train30m_full has both.
# Default is the current production ensemble. It used to default to train30m, which is the
# superseded 30 m design whose marsh members were built with an inert operator: a job launched
# without setting TRAIN trained on last month's data and reported a healthy-looking result.
# Override with TRAIN=... for the older ensembles.
TRAIN=${TRAIN:-$SCR/runs/pp4_e01}
VALID=${VALID:-$SCR/runs/valid30m}
# Decoded-grid cache is keyed on resolved paths, but keep ensembles in separate caches so a
# corrected and an uncorrected grid at the same path cannot collide.
CACHE_TAG=$(basename "$TRAIN")
MODE=${1:-holdout}

# stdout is a file here, so Python block-buffers it: four hours of training produced a log
# under 4 KB with no epoch lines in it, which reads exactly like a hung job. python -u below
# forces line buffering. Progress you cannot see is progress you will cancel.
log() { echo "[$(date +%H:%M:%S)] $*"; }
ok()  { echo "  [OK]   $*"; }
bad() { echo "  [FAIL] $*"; }

echo "============ CORAL emulator training, job $SLURM_JOB_ID on $(hostname) ============"
log "mode=$MODE"
log "ensemble=$TRAIN"

module load anaconda3 cuda 2>/dev/null
source activate coral 2>/dev/null || conda activate coral 2>/dev/null
cd "$CORAL" || exit 1

echo; echo "=== preconditions ==="
[ -s "$TRAIN/manifest.json" ] && ok "training manifest" || { bad "run 01_prep_ensemble first"; exit 1; }
# Hard stop rather than a warning: coral.emulator.train falls back to CPU when CUDA is
# absent, so a GPU that failed to attach would not error, it would train at roughly a
# hundredth of the speed and look healthy in the log. Checking on a login node proves nothing
# because login nodes have no GPU; this runs on the allocated compute node.
python - <<'PYCHK' || { bad "no usable GPU in this job. Check --gres=gpu:1 and the partition."; exit 1; }
import sys
try:
    import torch
except ImportError:
    print("  torch not installed: pip install -e '.[emulator]'"); sys.exit(1)
print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  CUDA not available on this node. Training would silently fall back to CPU and")
    print("  take weeks rather than hours, so this is a hard failure.")
    sys.exit(1)
print(f"  device: {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
# is_available() is not enough. On a V100 (compute capability 7.0) it returns True, and this
# torch build ships kernels for >=7.5 only, so the failure surfaces at the first kernel launch
# minutes into training with "no kernel image is available for execution on the device".
# Launching a real kernel here turns that into an immediate, legible precondition failure.
cc = torch.cuda.get_device_capability(0)
try:
    (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
    print(f"  kernel launch OK on compute capability {cc[0]}.{cc[1]}")
except Exception as e:
    print(f"  compute capability {cc[0]}.{cc[1]} has no kernels in this torch build: {e}")
    print("  Use rtx6000 (7.5), a100 (8.0), l40s (8.9) or h100 (9.0), not v100 (7.0).")
    sys.exit(1)
PYCHK
NDONE=$(ls "$TRAIN"/*/results_*/res_*.max 2>/dev/null | wc -l | tr -d ' ')
NPLAN=$(wc -l < "$TRAIN/run_dirs.txt")
log "$NDONE of $NPLAN members finished"
[ "$NDONE" -ge 50 ] && ok "enough members to train on" \
  || { bad "only $NDONE members done; let the array run further"; exit 1; }

OUTDIR=$CORAL/reports/emulator/$CACHE_TAG
mkdir -p "$OUTDIR"
# Decoded-grid cache on scratch. Parsing the ASCII grids costs about 35 min per epoch over
# 1506 members, which makes a 300-epoch run a week long. The first epoch fills this and the
# rest reload from it. Keyed on the resolved path, so grids an intervention did not edit are
# stored once across the whole ensemble rather than per member.
CACHE=$SCR/interim/grid_cache_$CACHE_TAG
mkdir -p "$CACHE"
log "grid cache: $CACHE ($(du -sh "$CACHE" 2>/dev/null | cut -f1) so far)"

echo; echo "=== training ==="
case "$MODE" in
  perkind)
    # The headline split. Holds out a fraction of members within every kind x sea level x
    # siting cell, so both sets contain all seven interventions at all eight levels in both
    # placement regimes. Answers "has it learned each intervention", which is the question
    # every other result is read against. $2 = holdout fraction, default 0.2.
    FRAC=${2:-0.2}
    log "stratified per-kind holdout: $FRAC of each kind x level x siting cell"
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" --holdout-per-kind "$FRAC" \
      --epochs 300 --patience 10 --workers 8 --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_perkind.pt" \
      --report "$OUTDIR/emulator_unet_perkind.report.json" ;;
  combos)
    # Train on single interventions, validate on the two-intervention members. This is what
    # decides whether the emulator can SEARCH adaptation space rather than only look up
    # configurations it has seen: portfolio space is far larger than could ever be simulated.
    log "portfolio holdout: single interventions train, two-intervention members validate"
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" --holdout-combos \
      --epochs 300 --patience 10 --workers 8 --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_combos.pt" \
      --report "$OUTDIR/emulator_unet_combos.report.json" ;;
  siting)
    # Train on random placements, validate on targeted ones: distribution shift in
    # intervention geometry. Weaker for de-pave by construction, because its suitability
    # driver is flat over a mostly dry impervious footprint and targeted placement there
    # degenerates to a sized random subset. Report that kind separately.
    HOLD=${2:-targeted}
    log "placement-regime holdout: validating on siting=$HOLD"
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" --holdout-siting "$HOLD" \
      --epochs 300 --patience 10 --workers 8 --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_siting_${HOLD}.pt" \
      --report "$OUTDIR/emulator_unet_siting_${HOLD}.report.json" ;;
  holdout)
    # Hold out whole sea levels. A random split would leak: 250 members share each level,
    # so near-duplicates of every held-out case would sit in the training set.
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" \
      --holdout-slr IntHigh2100 High2100 \
      --epochs 300 --patience 10 --workers 8 \
      --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_holdout.pt"
    ;;
  valid)
    # The stronger test: train on random placements, validate on the targeted ensemble.
    [ -s "$VALID/manifest.json" ] || { bad "no $VALID; build it with 01_prep_ensemble pinpoint_valid30m valid30m"; exit 1; }
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" --val-ensemble "$VALID" \
      --epochs 300 --patience 10 --workers 8 \
      --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_targeted.pt"
    ;;
  curve)
    # Does accuracy still improve with ensemble size? This is what decides whether to raise
    # n_per_kind from 50 to 100, which costs another 124 GB and about 12 hours of runs.
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" \
      --holdout-slr Int2070 \
      --subsample-per-kind 5 10 25 50 \
      --epochs 200 --patience 8 --workers 8 \
      --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_curve.pt"
    ;;
  curve50)
    # Just the n=50 point of the learning curve. The 12 h run finished 5, 10 and 25 before
    # timing out, and those checkpoints are already on disk; rerunning `curve` would spend
    # GPU hours recomputing them. --report is separate so the n<=25 report is not clobbered:
    # train_ensemble.py tags checkpoints with _n{N} but derives the report path from --ckpt.
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" \
      --holdout-slr Int2070 \
      --subsample-per-kind 50 \
      --epochs 200 --patience 8 --workers 8 \
      --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_curve.pt" \
      --report "$OUTDIR/emulator_unet_curve_n50.report.json"
    ;;
  holdout-slr)
    # Hold out one sea level (extrapolation test). $2 = the SLR label, e.g. High2100.
    HOLD=${2:-High2100}
    log "holdout SLR: $HOLD"
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" --holdout-slr "$HOLD" \
      --epochs 300 --patience 10 --workers 8 --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_holdout_slr_${HOLD}.pt" \
      --report "$OUTDIR/emulator_unet_holdout_slr_${HOLD}.report.json" ;;
  holdout-kind)
    # Hold out one intervention kind (generalisation test — the actual use case).
    # $2 = the kind, e.g. seawall. Trains on the other four, predicts the fifth.
    HOLD=${2:-seawall}
    log "holdout kind: $HOLD"
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" --holdout-kind "$HOLD" \
      --epochs 300 --patience 10 --workers 8 --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_holdout_kind_${HOLD}.pt" \
      --report "$OUTDIR/emulator_unet_holdout_kind_${HOLD}.report.json" ;;
  holdout-placement|holdout-realisation)
    # The production test: train on every kind and sea level, validate on unseen placements
    # of known kinds. $2 = realisation threshold: members with realisation index >= $2 go to
    # validation, the rest train. That is the highest (n_per_kind - $2) realisations per kind
    # per level. pp4_tide_decision has n_per_kind=25, so `holdout-placement 20` trains on 20
    # realisations per kind per level and validates on the highest 5 (indices 20-24). A user
    # in the tool draws a wall somewhere the model has not seen, so this is the split that
    # measures the deployment claim; holdout-slr (unseen forcing) and holdout-kind (unseen
    # type) are the complementary stress tests.
    THRESH=${2:-20}
    log "holdout placement: realisations >= $THRESH validate"
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" --holdout-realisation "$THRESH" \
      --epochs 300 --patience 10 --workers 8 --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_holdout_placement.pt" \
      --report "$OUTDIR/emulator_unet_holdout_placement.report.json" ;;
  curve-placement)
    # Learning curve for the production test: does accuracy still improve with more
    # realisations per kind when validation is unseen PLACEMENTS of known kinds?
    # $2 = realisation threshold (default 20), same semantics as holdout-placement: realisations
    # >= $2 validate and are never subsampled into a training point. Each subsample n keeps
    # realisations 0..n-1, so as long as n <= $2 the training pool stays disjoint from
    # validation. This assumes n_per_kind=25: subsamples 10, 15, 20 are the distinct points,
    # and the n=20 point is the same training data as `holdout-placement 20`, so it doubles as
    # that model's estimate at 200 epochs. A flat curve means 25/kind is plenty for the next
    # ensemble; a curve still falling at 20 says raise n_per_kind.
    THRESH=${2:-20}
    log "learning curve vs placement: realisations >= $THRESH validate, subsample 10/15/20"
    python -u workflows/train_ensemble.py \
      --ensemble "$TRAIN" \
      --holdout-realisation "$THRESH" \
      --subsample-per-kind 10 15 20 \
      --epochs 200 --patience 8 --workers 8 \
      --cache-dir "$CACHE" \
      --ckpt "$OUTDIR/emulator_unet_curve_placement.pt" \
      --report "$OUTDIR/emulator_unet_curve_placement.report.json" ;;
  *) bad "unknown mode $MODE (use perkind, combos, siting, holdout, holdout-slr, holdout-kind, holdout-placement, curve-placement, valid, curve, or curve50)"; exit 1 ;;
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
