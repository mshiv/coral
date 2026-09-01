#!/bin/bash
# Controlled objective ablation. Submit once; array task 0 trains the unchanged absolute-depth
# control and task 1 adds the paired intervention-response loss.
#
#   sbatch workflows/slurm/03_train_paired_ab.sbatch
#
# Optional overrides:
#   TRAIN=/path/to/ensemble SEED=23 DELTA_LAMBDA=1 sbatch ...
#SBATCH --job-name=coral-paired-ab
#SBATCH --account=gts-arobel3-atlas
#SBATCH --array=0-1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=coral-paired-ab_%A_%a.log

set -euo pipefail

export CORAL=${CORAL:-/storage/home/hcoda1/6/smurugan9/data/dev/coral}
export SCR=${SCR:-/storage/home/hcoda1/6/smurugan9/scratch/coral}
TRAIN=${TRAIN:-$SCR/runs/pp4_e01}
SEED=${SEED:-23}
DELTA_LAMBDA=${DELTA_LAMBDA:-1.0}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-0.02}
FRAC=${FRAC:-0.2}
EPOCHS=${EPOCHS:-300}
PATIENCE=${PATIENCE:-10}
WORKERS=${WORKERS:-8}
RUN_TAG=${RUN_TAG:-full}

module load anaconda3 cuda 2>/dev/null || true
source activate coral 2>/dev/null || conda activate coral
cd "$CORAL"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
x=torch.ones(4,device='cuda'); print(torch.__version__,torch.cuda.get_device_name(0),x.sum().item())
PY

case "$SLURM_ARRAY_TASK_ID" in
  0) OBJECTIVE=absolute ;;
  1) OBJECTIVE=paired ;;
  *) echo "unexpected array task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

TAG=$(basename "$TRAIN")
OUT="$CORAL/reports/emulator/${TAG}/paired_ab_${RUN_TAG}_seed${SEED}"
CACHE="$SCR/interim/grid_cache_${TAG}"
mkdir -p "$OUT" "$CACHE"

python -u workflows/train_paired_ab.py \
  --ensemble "$TRAIN" \
  --objective "$OBJECTIVE" \
  --holdout-per-kind "$FRAC" \
  --delta-lambda "$DELTA_LAMBDA" \
  --delta-train-threshold "$DELTA_THRESHOLD" \
  --epochs "$EPOCHS" --patience "$PATIENCE" --workers "$WORKERS" \
  --cache-dir "$CACHE" --seed "$SEED" --cell-m 4 \
  --ckpt "$OUT/unet_${OBJECTIVE}.pt" \
  --report "$OUT/unet_${OBJECTIVE}.report.json"
