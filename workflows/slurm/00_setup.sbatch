#!/bin/bash
# CORAL setup: directory layout, symlinks, environment, and every data fetch.
# Can resubmit after a failure.
#
#   sbatch workflows/slurm/00_setup.sbatch
#
# Assumes only that $BASE (the validated 30 m compound run) and the GeoClaw output are
# already on scratch. 
#SBATCH --job-name=coral-setup
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks-per-node=24 
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=coral-setup_%j.log

set -uo pipefail          # not -e: we handle failures per step so the log stays readable

export CORAL=/storage/home/hcoda1/6/smurugan9/data/dev/coral
export PROJ=/storage/home/hcoda1/6/smurugan9/data/projects/coral
export SCR=/storage/home/hcoda1/6/smurugan9/scratch/coral
export JOBS=/storage/home/hcoda1/6/smurugan9/data/jobs/coral
export LISF=/storage/home/hcoda1/6/smurugan9/data/software/LISFLOOD-FP-trunk/lisflood
export BASE=$SCR/runs/compound30m

STEP=0; FAILED=0
log()  { echo "[$(date +%H:%M:%S)] $*"; }
step() { STEP=$((STEP+1)); echo; echo "=== STEP $STEP: $* ==="; }
ok()   { echo "  [OK]   $*"; }
bad()  { echo "  [FAIL] $*"; FAILED=$((FAILED+1)); }
# assert a file exists and is non-empty; most fetches fail by writing nothing
have() { if [ -s "$1" ]; then ok "$(basename "$1") ($(du -h "$1" | cut -f1))"; else bad "missing or empty: $1"; fi; }
run()  { log "\$ $*"; if "$@"; then ok "$1 completed"; else bad "$* returned $?"; fi; }

echo "================ CORAL setup, job $SLURM_JOB_ID on $(hostname) ================"
log "CORAL=$CORAL"; log "PROJ=$PROJ"; log "SCR=$SCR"; log "BASE=$BASE"

# ---------------------------------------------------------------- 1. preconditions
step "preconditions"
[ -x "$LISF" ] && ok "LISFLOOD binary present" || bad "no binary at $LISF"
[ -d "$BASE" ] && ok "\$BASE exists" || bad "\$BASE missing: transfer compound30m first"
have "$BASE/SUB_DEM_SAV.asc"
have "$BASE/savannah_ascii.par"
if [ "$FAILED" -gt 0 ]; then echo; echo "ABORT: preconditions failed"; exit 1; fi

# ---------------------------------------------------------------- 2. directories
step "directory layout"
run mkdir -p "$PROJ/data" "$PROJ/data/slr" "$SCR/runs" "$SCR/interim" "$JOBS"
mkdir -p "$CORAL/data"
for d in "$PROJ/data" "$SCR/runs" "$SCR/interim" "$JOBS"; do
  [ -d "$d" ] && ok "$d" || bad "$d"
done

# ---------------------------------------------------------------- 3. symlinks
step "repo symlinks into PROJ and SCR"
cd "$CORAL" || exit 1
for pair in "data/raw:$PROJ/data" "data/interim:$SCR/interim" "runs:$SCR/runs"; do
  lnk=${pair%%:*}; tgt=${pair##*:}
  if [ -d "$lnk" ] && [ ! -L "$lnk" ]; then
    log "moving real directory $lnk aside to $lnk.bak"
    mv "$lnk" "$lnk.bak"
  fi
  ln -sfn "$tgt" "$lnk"
  [ "$(readlink "$lnk")" = "$tgt" ] && ok "$lnk -> $tgt" || bad "$lnk did not link to $tgt"
done

# ---------------------------------------------------------------- 4. environment
step "python environment"
module load anaconda3 gdal 2>/dev/null
source activate coral 2>/dev/null || conda activate coral 2>/dev/null
run python -c "import coral, numpy, scipy, rasterio, geopandas; print('imports ok')"
log "python: $(which python)"; log "gdal: $(gdalinfo --version 2>/dev/null)"

# ---------------------------------------------------------------- 5. base run hygiene
step "base run hygiene (netCDF pars and checkpoint)"
# Two pars enable netcdf_out, which segfaults on finalisation in this build.
rm -f "$BASE/test.par" "$BASE/savannah.par"
[ "$(ls "$BASE"/*.par | wc -l)" -eq 1 ] && ok "only savannah_ascii.par remains" \
  || bad "expected one .par, found $(ls "$BASE"/*.par | wc -l)"
# checkpoint makes a requeued ensemble member reload a stale restart file and stall.
sed -i -E '/^[[:space:]]*checkpoint\b/d' "$BASE/savannah_ascii.par"
[ "$(grep -c checkpoint "$BASE/savannah_ascii.par")" -eq 0 ] && ok "checkpoint removed" || bad "checkpoint still set"
# saveint writes 97 snapshots per member (38.8 MB each, 3.76 GB per member, roughly 5.7 TB over
# 1506 members). The emulator needs only .max, so snapshots and flux output are dropped for the
# bulk ensemble. What this gives up: the depth and water-surface time series, the Qx/Qy flux
# fields (hence velocity and depth-velocity hazard), and animation frames. What survives,
# because these are written at end of run regardless: .max (peak depth), .mxe (max water
# surface), .maxtm (time of peak), .inittm (arrival time), .totaltm (inundation duration),
# and .mass. Duration and arrival therefore do not need the snapshots.
# Keep full output for the small runs where evolution and hazard matter: the 4 m run, the
# drainage tier pair, and a handful of baseline members (see the note printed at the end).
sed -i -E '/^[[:space:]]*(saveint|qoutput)\b/d' "$BASE/savannah_ascii.par"
grep -qE '^[[:space:]]*saveint' "$BASE/savannah_ascii.par" && bad "saveint still set" || ok "saveint and qoutput removed"
log "NOTE: verify the smoke test still produces res_matthew_sav.max. If it does not, this"
log "      build needs saveint present; set 'saveint 172800' instead of removing it."
log "final par:"; sed 's/^/    /' "$BASE/savannah_ascii.par"

# ---------------------------------------------------------------- 6. data fetches
step "SAGIS layers"
DOM="-81.22 -80.82 31.82 32.08"
PP="-81.1103 -81.0727 31.9367 31.9690"     # corrected Pin Point clip
run python -m coral.preprocess.fetch_sagis --group physics    --bbox $DOM --out data/raw/sagis_savannah
run python -m coral.preprocess.fetch_sagis --group context    --bbox $DOM --out data/raw/sagis_savannah
run python -m coral.preprocess.fetch_sagis --group physics    --bbox $PP  --out data/raw/sagis_pinpoint
run python -m coral.preprocess.fetch_sagis --group stormwater --bbox $PP  --out data/raw/sagis_pinpoint
have data/raw/sagis_savannah/sagis_wetlands_nwi.geojson
have data/raw/sagis_savannah/sagis_soils_nrcs.geojson
have data/raw/sagis_pinpoint/sagis_soils_nrcs.geojson
have data/raw/sagis_pinpoint/sagis_inlets_sav.geojson

step "FEMA building footprints"
run python -m coral.preprocess.fetch_buildings --bbox $PP \
    --out data/raw/sagis_pinpoint/fema_structures_pinpoint.geojson
have data/raw/sagis_pinpoint/fema_structures_pinpoint.geojson

step "NLCD land cover"
run python -m coral.preprocess.download_nlcd --dem "$BASE/SUB_DEM_SAV.asc" \
    --out data/raw/nlcd_savannah.tif --year 2016
have data/raw/nlcd_savannah.tif

step "SSURGO Ksat and storage"
run python -m coral.preprocess.fetch_ssurgo \
    --soils-geojson data/raw/sagis_savannah/sagis_soils_nrcs.geojson \
    --out data/raw/ssurgo_savannah/ksat_awc_by_mukey.json
run python -m coral.preprocess.fetch_ssurgo \
    --soils-geojson data/raw/sagis_pinpoint/sagis_soils_nrcs.geojson \
    --out data/raw/ssurgo_pinpoint/ksat_awc_by_mukey.json
have data/raw/ssurgo_savannah/ksat_awc_by_mukey.json
have data/raw/ssurgo_pinpoint/ksat_awc_by_mukey.json
have data/raw/sagis_pinpoint/sagis_wetlands_nwi.geojson

step "CoNED lidar, 4 m clip floored at -1 m"
CONED=/vsicurl/https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/CoNED_GA_DEM_2022_9559/CoNED_GA_DEM_2022_m9559_EPSG-6346.vrt
run gdalwarp -q -overwrite -te -81.1103 31.9367 -81.0727 31.9690 -t_srs EPSG:4326 \
    -tr 0.000036 0.000036 -r bilinear "$CONED" "$PROJ/data/coned_pinpoint_4m.tif"
run gdal_translate -q -of AAIGrid "$PROJ/data/coned_pinpoint_4m.tif" "$PROJ/data/coned_pinpoint_4m.asc"
run python -m coral.preprocess.floor_dem --dem "$PROJ/data/coned_pinpoint_4m.asc" --floor -1 \
    --out data/raw/coned_pinpoint_4m_floored.asc
have data/raw/coned_pinpoint_4m_floored.asc
run python -m coral.preprocess.download_nlcd --dem data/raw/coned_pinpoint_4m_floored.asc \
    --out data/raw/nlcd_pinpoint.tif --year 2016
have data/raw/nlcd_pinpoint.tif

step "sea-level scenarios"
# Informational only. fetch_slr reads its values from the bundled FORT_PULASKI table, which
# was generated from this spreadsheet; no code reads the file at run time, so its absence is
# not a failure. It is kept as provenance for the numbers.
if [ -s "$PROJ/data/slr/sl_taskforce_scenarios_psmsl_id_395.xlsx" ]; then
  ok "task force spreadsheet present (provenance)"
else
  log "NOTE: spreadsheet not staged. Values come from the bundled table, so this is fine;"
  log "      scp it to $PROJ/data/slr/ if you want the source archived alongside the runs."
fi
run python -m coral.preprocess.fetch_slr --years 2040 2050 2070 2100

# ---------------------------------------------------------------- 7. summary
step "summary"
echo "  data/raw contents:"; ls -1 data/raw | sed 's/^/    /'
echo "  disk used in PROJ/data: $(du -sh "$PROJ/data" | cut -f1)"
echo "  scratch free: $(df -h "$SCR" | awk 'NR==2{print $4}')"
echo
if [ "$FAILED" -eq 0 ]; then
  echo "================ SETUP COMPLETE, $STEP steps, 0 failures ================"
  echo "Next: sbatch $CORAL/workflows/slurm/01_prep_ensemble.sbatch"
  exit 0
else
  echo "================ SETUP FINISHED WITH $FAILED FAILURE(S) ================"
  echo "Search this log for '[FAIL]'. The script is idempotent, so fix and resubmit."
  exit 1
fi
