#!/bin/bash
# Submit the five in-range intervention evaluations used by Chapter 4.
set -euo pipefail

cd "$(dirname "$0")/../.."
for stem in \
  perkind \
  combos \
  siting_targeted \
  holdout_kind_floodwall \
  holdout_kind_living_shoreline
do
  job=$(sbatch --parsable workflows/slurm/05_evaluate_local_response.sbatch "$stem")
  echo "$stem -> job $job"
done
