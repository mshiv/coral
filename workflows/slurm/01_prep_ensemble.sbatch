#!/bin/bash
# Build the training ensemble run directories, then print the exact command to launch them.
# It writes the edited grids for every member. 
# Grids an intervention does not touch, and the .bdy for a
# given SLR level, are symlinked. 
#
#   sbatch workflows/slurm/01_prep_ensemble.sbatch                     # training, 1506 members
#   sbatch workflows/slurm/01_prep_ensemble.sbatch pinpoint_valid30m valid30m   # validation set
#SBATCH --job-name=coral-prep
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=coral-prep_%j.log

set -uo pipefail

export CORAL=/storage/home/hcoda1/6/smurugan9/data/dev/coral
export PROJ=/storage/home/hcoda1/6/smurugan9/data/projects/coral
export SCR=/storage/home/hcoda1/6/smurugan9/scratch/coral
export BASE=$SCR/runs/compound30m

SCENARIO=${1:-pinpoint_train30m}      # config name, without path or .yaml
OUTNAME=${2:-train30m}                # directory under $SCR/runs
OUT=$SCR/runs/$OUTNAME

log() { echo "[$(date +%H:%M:%S)] $*"; }
ok()  { echo "  [OK]   $*"; }
bad() { echo "  [FAIL] $*"; }

echo "============ CORAL ensemble prep, job $SLURM_JOB_ID on $(hostname) ============"
log "scenario=$SCENARIO  ->  $OUT"

module load anaconda3 gdal 2>/dev/null
source activate coral 2>/dev/null || conda activate coral 2>/dev/null
cd "$CORAL" || exit 1

echo; echo "=== preconditions ==="
[ -s "configs/scenarios/$SCENARIO.yaml" ] && ok "config found" || { bad "no configs/scenarios/$SCENARIO.yaml"; exit 1; }
[ -s "$BASE/savannah_ascii.par" ] && ok "base par" || { bad "run 00_setup first"; exit 1; }
grep -qE '^[[:space:]]*saveint' "$BASE/savannah_ascii.par" \
  && bad "saveint still set in the base par: each member will write about 3.5 GB. Run 00_setup." \
  || ok "no saveint, members write only .max/.mxe/.mass"
[ -s data/raw/nlcd_savannah.tif ] && ok "NLCD present" || { bad "NLCD missing, run 00_setup"; exit 1; }
log "scratch free before: $(df -h "$SCR" | awk 'NR==2{print $4}')"

# A stale ensemble from an older sweep would mix relative run_dirs and the wrong .par.
if [ -d "$OUT" ]; then
  log "removing existing $OUT so the sweep is regenerated cleanly"
  rm -rf "$OUT"
fi

echo; echo "=== building the ensemble ==="
if python -m coral.emulator.sweep --base "$BASE" --out "runs/$OUTNAME" \
     --config "configs/scenarios/$SCENARIO.yaml" --nlcd data/raw/nlcd_savannah.tif; then
  ok "sweep completed"
else
  bad "sweep failed"; exit 1
fi

echo; echo "=== verification ==="
N=$(wc -l < "$OUT/run_dirs.txt")
ok "$N members in run_dirs.txt"
FIRST=$(head -1 "$OUT/run_dirs.txt")
case "$FIRST" in
  /*) ok "run_dirs.txt holds absolute paths" ;;
  *)  bad "run_dirs.txt is relative, the array will fail to cd" ;;
esac
PAR=$(grep -E '^/.*lisflood[^/]*\s' "$OUT/run_array.sbatch" | tail -1 | awk '{print $NF}')
[ "$PAR" = "savannah_ascii.par" ] && ok "array runs $PAR" || bad "array runs $PAR, expected savannah_ascii.par"

# every input the par references must resolve inside the first member (symlink or real file)
echo "  checking inputs resolve in $(basename "$FIRST"):"
MISS=0
for f in $(grep -oE '[A-Za-z_]+_[A-Za-z0-9_]+\.(asc|bci|bdy|txt)' "$FIRST/savannah_ascii.par" | sort -u); do
  if [ -s "$FIRST/$f" ]; then echo "      ok   $f"; else echo "      MISS $f"; MISS=$((MISS+1)); fi
done
[ "$MISS" -eq 0 ] && ok "all par inputs resolve" || bad "$MISS input(s) missing"

# confirm the SLR offset actually reached the boundary files
echo "  SLR offsets applied to the shared .bdy files:"
python - "$BASE" "$OUT" <<'PY'
import sys, glob, os
base, out = sys.argv[1], sys.argv[2]
def first(p):
    """First data row's water level. Both fields must parse as numbers: the HVAR block header
    ("50250  seconds") also has two fields and its first one is numeric, so checking only the
    first field returns the row count for every file and makes every delta look like zero."""
    for l in open(p):
        s = l.split()
        if len(s) == 2:
            try:
                v, t = float(s[0]), float(s[1])
                return v
            except ValueError:
                pass
b = first(glob.glob(os.path.join(base, "*.bdy"))[0])
for p in sorted(glob.glob(os.path.join(out, "_shared", "*.bdy"))):
    print(f"      {os.path.basename(p):48s} delta {first(p)-b:+.4f} m")
PY

# ---------------------------------------------------------------- baseline snapshots
# The bulk ensemble runs without snapshots (the emulator needs only .max). The baseline
# members, one per SLR level, keep the full 30 min output so the SLR response through time,
# the flux fields, and animation frames survive. Their .par is a symlink to the base par, so
# a separate snapshot-enabled par goes in _shared and the baselines point at that instead.
echo; echo "=== baseline snapshot output ==="
SNAPPAR="$OUT/_shared/savannah_ascii_snapshots.par"
cp -f "$BASE/savannah_ascii.par" "$SNAPPAR"
# strip first, so the result has exactly one of each even if the base par still carries them
sed -i -E '/^[[:space:]]*(saveint|qoutput)\b/d' "$SNAPPAR"
printf 'saveint       1800\nqoutput\n' >> "$SNAPPAR"
NB=0
for d in "$OUT"/slr*_base; do
  [ -d "$d" ] || continue
  ln -sfn "$SNAPPAR" "$d/savannah_ascii.par"
  NB=$((NB+1))
done
ok "$NB baseline members linked to the snapshot par (about 3.8 GB each)"
grep -qE '^[[:space:]]*saveint' "$SNAPPAR" && ok "snapshot par has saveint 1800" || bad "snapshot par missing saveint"

echo; echo "=== footprint ==="
echo "  ensemble on disk now: $(du -sh "$OUT" | cut -f1)"
echo "  shared files: $(ls "$OUT/_shared" | wc -l | tr -d ' ') (one .bdy per SLR level, plus the snapshot par)"
echo "  projected outputs: about $((N * 55 / 1024)) GB from $N members, plus $((NB * 4)) GB of baseline snapshots"
echo "  scratch free: $(df -h "$SCR" | awk 'NR==2{print $4}')"

# ---------------------------------------------------------------- array size limit
MAXARR=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/{gsub(/ /,"",$2); print $2}')
MAXARR=${MAXARR:-1001}
# MaxArraySize caps the maximum task INDEX, not the task count: with MaxArraySize=1001 the
# largest legal index is 1000, so --array=1002-1506 is rejected. Chunks therefore always use
# low indices and set IDX_OFFSET to pick their block of run_dirs.txt.
MAXIDX=$((MAXARR - 1))
QOS=$(sacctmgr -n show assoc where user="$USER" format=QOS%40 2>/dev/null | head -1 | tr -d ' ' | cut -d, -f1)
MAXSUB=$(sacctmgr -n show qos "$QOS" format=MaxSubmitJobsPU 2>/dev/null | tr -d ' ')
[ -z "$MAXSUB" ] && MAXSUB=500
echo; echo "=== SLURM limits ==="
echo "  MaxArraySize $MAXARR, so the largest legal task index is $MAXIDX"
echo "  QOS $QOS MaxSubmitPU $MAXSUB, counting pending and running, array elements individually"
if [ "$N" -le "$MAXIDX" ] && [ "$N" -lt "$MAXSUB" ]; then
  ok "$N members fit in a single array within both limits"
  LAUNCH="sbatch --array=1-$N run_array.sbatch"
else
  log "$N members exceed a limit, so submit through the chunk driver, which waits for the"
  log "queue to drain between chunks rather than tripping MaxSubmitPU"
  LAUNCH="  cd $OUT
  nohup bash \$CORAL/workflows/slurm/submit_chunks.sh > submit_chunks.log 2>&1 &
  tail -f submit_chunks.log"
fi

echo
echo "============ PREP COMPLETE ============"
cat <<EOF

Launch the ensemble:

  cd $OUT
  sbatch --array=1-3%3 run_array.sbatch     # smoke test three members first

  # Confirm the three wrote .max, which also proves removing saveint did not suppress it:
  ls */results_matthew_sav/res_matthew_sav.max

  # then the rest:
$LAUNCH

Progress:
  squeue -u \$USER | wc -l
  ls $OUT/*/results_matthew_sav/res_matthew_sav.max 2>/dev/null | wc -l    # climbs to $N

Baseline members ($NB of them) write full 30 min snapshots for animations and flux fields.
To add snapshots for any other member later, copy the snapshot par over its symlink and
resubmit that one member; each is only about 10 minutes.
EOF
