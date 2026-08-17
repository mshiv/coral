#!/usr/bin/env bash
# Assemble the GeoClaw run directory on the HPC.
# Run from the repo root (~/scratch/savannah_matthew). Pass the path to your
# existing Matthew run as $1.
#
#   bash scripts/assemble_geoclaw_run.sh  ~/path/to/matthew_2016_MSLtide_0.81
#
# Result: ./geoclaw_run ready to `make .data && make .output`.
# templates/setrun.py has the 63 coupling gauges + 5 NOAA stations inlined and the Pin Point
# box at level 7. Observation gauges come from obs_gauges.txt in the run dir, if present.
# inlined, so nothing else needs editing.

set -euo pipefail
MATTHEW="${1:?usage: assemble_geoclaw_run.sh <path-to-existing-matthew-run>}"
DEST="geoclaw_run"

if [ ! -f "$MATTHEW/setrun.py" ]; then
    echo "ERROR: $MATTHEW doesn't look like a GeoClaw run (no setrun.py)"; exit 1
fi
if [ ! -f templates/setrun.py ]; then
    echo "ERROR: run from the repo root (templates/setrun.py not found)"; exit 1
fi

# 1. copy the base run WITHOUT the heavy/derived bits
rsync -a --exclude='_output' --exclude='_plots' --exclude='scratch' \
      --exclude='*.data' "$MATTHEW"/ "$DEST"/

# 2. reuse topo via symlink when the source case has it, so the ~80 MB of CRM and GEBCO is
#    not duplicated per run. When it does not, make an empty one: setrun downloads GEBCO, CRM
#    vol2 and the ATCF track into scratch/ on the first `make .data`, which needs a login node
#    because compute nodes usually have no outbound network.
if [ -d "$MATTHEW/scratch" ]; then
    ln -sfn "$(cd "$MATTHEW/scratch" && pwd)" "$DEST/scratch"
    echo "topo: symlinked $MATTHEW/scratch"
else
    mkdir -p "$DEST/scratch"
    echo "topo: no scratch/ in $MATTHEW -- setrun will download it on the first make .data"
fi

# 3. install the tracked setrun.py and kml2slu.py. kml2slu is here because the case copy uses
#    chained assignment, which writes to a temporary under pandas Copy-on-Write, so every
#    ruled-rectangle value came back None and RuledRectangle failed comparing None to None.
cp templates/setrun.py "$DEST/setrun.py"
cp templates/kml2slu.py "$DEST/kml2slu.py"

# 4. regenerate the .data files from the new setrun.py. setrun reads the scenario, so pass it
#    through: without it setrun falls back to a path under $CORAL, and if that is unset the
#    load fails rather than quietly using defaults.
: "${CORAL_SCENARIO:=$PWD/configs/scenarios/savannah_matthew_compound.yaml}"
export CORAL_SCENARIO
echo "scenario: $CORAL_SCENARIO"
( cd "$DEST" && make .data )

# 5. verify (gauge count is dynamic: coupling gauges + any dense obs gauges from
#    gen_boundary_points --obs-spacing-m; the header line of gauges.data reports it)
echo "=========================================================="
echo "gauges.data (gauge count reported below — coupling + optional dense obs gauges):"
head -1 "$DEST/gauges.data"
echo "regions: the L6 Pin Point box should appear in regions.data (and an L7 box if dense"
echo "         obs gauges were added — see the level-7 flagregion bbox printed by"
echo "         gen_boundary_points):"
grep -nE "^\s*[67]\s" "$DEST/regions.data" || \
    echo "  (check regions.data manually for the [6,6,...] / [7,7,...] Pin Point boxes)"
echo "=========================================================="
echo "Next:  cd $DEST && make .exe   # if xgeoclaw not built on this HPC"
echo "       cd $DEST && make .output  # or submit your SLURM job"
