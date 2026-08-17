#!/bin/bash
#SBATCH --job-name=geoclaw
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=out.geoclaw.%j
#SBATCH --error=err.geoclaw.%j
#SBATCH --time=12:00:00

# Run one GeoClaw case.
#   sbatch run_geoclaw.sbatch <run_dir> [scenario.yaml]
#
# GeoClaw's AMR loop is OpenMP-only, so this is one node with threads. The cost is driven by
# the finest level and the area it covers: level 7 (~36 m) over the 22 x 61 km gauge box is
# far more than level 6 (~145 m) over the same box, because each extra level refines in x, y
# and t together.
#
# setrun.py reads the scenario for sea_level, amr_max and refine_box, so the level is changed
# by editing the config rather than the file. Pass the scenario explicitly here: the job runs
# from the run directory, and a relative default would resolve against that instead of the repo.
#
# The topography and the ATCF track are downloaded by setrun on the first `make .data`, which
# needs outbound network. Compute nodes usually have none, so run `make .data` on a login node
# before submitting this.

set -euo pipefail
module load gcc/12.3.0 netcdf-c/4.9.2-gszew36

RUN_DIR="${1:?usage: sbatch run_geoclaw.sbatch <run_dir> [scenario.yaml]}"
SCENARIO="${2:-$CORAL/configs/scenarios/savannah_matthew_compound.yaml}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CORAL_SCENARIO="$SCENARIO"

cd "$RUN_DIR"
echo "run      $PWD"
echo "scenario $CORAL_SCENARIO"
echo "threads  $OMP_NUM_THREADS"
python -c "
from coral import config; import os
c = config.load(os.environ['CORAL_SCENARIO'])
print(f'  sea_level {c.geoclaw.sea_level}  amr_max {c.geoclaw.amr_max}  refine_box {c.geoclaw.refine_box}')"
grep -vE '^#|^$' gauges.data | head -1

# .data is rebuilt here so the run matches the scenario as read at submit time, not whatever
# was last written interactively. It needs no network once scratch/ is populated.
make .data
time make .output

echo "frames written:"
ls _output/fort.q[0-9]* 2>/dev/null | wc -l
