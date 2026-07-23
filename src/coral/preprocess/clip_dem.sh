#!/usr/bin/env bash
# Step 1: clip the LISFLOOD domain DEM from the full Georgia CUDEM (30 m).
#
# Pin Point, Savannah is ~ (-81.13, 31.95). Keep the box TIGHT around your area
# of interest -- every extra cell costs LISFLOOD runtime (and the full CUDEM is
# 6766 x 8226 = 55 M cells). A ~0.3deg x 0.3deg box around Pin Point is plenty
# to start and runs fast.
#
# Edit the bounding box, then run. Needs GDAL (conda env 'carto' has gdal 3.6.4).
#
#   -projwin = ulx uly lrx lry  (upper-left lon/lat, lower-right lon/lat)

set -euo pipefail
GDAL="${GDAL:-$HOME/opt/anaconda3/envs/carto/bin}"
CUDEM="${1:?usage: clip_dem.sh <ga_cudem_30m.asc> <out.asc>}"
OUT="${2:?usage: clip_dem.sh <ga_cudem_30m.asc> <out.asc>}"

# --- EDIT THIS BOX (around Pin Point / Savannah) ---
ULX=-81.30   # west
ULY=32.10    # north
LRX=-80.95   # east  (toward the coast/ocean)
LRY=31.80    # south
# ----------------------------------------------------

"$GDAL/gdal_translate" -of AAIGrid \
    -projwin $ULX $ULY $LRX $LRY \
    -a_srs EPSG:4326 \
    "$CUDEM" "$OUT"

echo "wrote $OUT"
head -6 "$OUT"
echo "NOTE: the EAST edge should be open ocean so the surge boundary points land in water."
