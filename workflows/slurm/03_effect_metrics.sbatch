#!/bin/bash
# Intervention effect metrics over a finished ensemble: sign structure, benefit against adverse
# volume, area change at several depth thresholds, and whether intervention rankings survive
# sea-level rise.
#
#   sbatch workflows/slurm/03_effect_metrics.sbatch                    # pp4_e01 at 4 m
#   sbatch workflows/slurm/03_effect_metrics.sbatch pp4_e01 pp4_base 4
#
# CPU only. The cost is ASCII parsing: two grids per member across nearly two thousand members.
#SBATCH --job-name=coral-effect
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH -p cpu-small
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=coral-effect_%j.log

set -uo pipefail

export CORAL=/storage/home/hcoda1/6/smurugan9/data/dev/coral
export SCR=/storage/home/hcoda1/6/smurugan9/scratch/coral

ENS=${1:-pp4_e01}
BASE=${2:-pp4_base}
CELL=${3:-4}
# The community centre and radius that define the target zone. Benefit is counted inside it and
# adverse amplification outside, so "the intervention helped" is a statement about somewhere
# declared in advance rather than wherever it happened to help.
REF_LON=${4:--81.0903}
REF_LAT=${5:-31.9522}
RADIUS=${6:-2.0}

echo "============ CORAL effect metrics, job $SLURM_JOB_ID on $(hostname) ============"
echo "[$(date +%H:%M:%S)] ensemble=$SCR/runs/$ENS  base=$SCR/runs/$BASE  cell=${CELL} m"

module load anaconda3 2>/dev/null
source activate coral 2>/dev/null || conda activate coral 2>/dev/null
cd "$CORAL" || exit 1

[ -s "$SCR/runs/$ENS/manifest.json" ] || { echo "  [FAIL] no manifest in $SCR/runs/$ENS"; exit 1; }
DEM=$(ls "$SCR/runs/$BASE"/SUB_DEM_*.asc 2>/dev/null | head -1)
[ -n "$DEM" ] || { echo "  [FAIL] no SUB_DEM in $SCR/runs/$BASE"; exit 1; }
echo "  DEM $DEM"

mkdir -p reports/adapt
python -u -m coral.analysis.effect_metrics \
  --ens "$SCR/runs/$ENS" \
  --dem "$DEM" \
  --waterline 1.114 \
  --cell-m "$CELL" \
  --ref-point "$REF_LON" "$REF_LAT" \
  --focus-radius-km "$RADIUS" \
  --out-csv "reports/adapt/effect_metrics_${ENS}.csv"
RC=$?

echo
if [ "$RC" -eq 0 ]; then
  echo "  [OK] wrote reports/adapt/effect_metrics_${ENS}.csv"
  wc -l "reports/adapt/effect_metrics_${ENS}.csv"
else
  echo "  [FAIL] returned $RC"
fi
exit $RC
