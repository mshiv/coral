#!/usr/bin/env bash
# Deliverable 1: Cloud-Optimized GeoTIFF (COG) -- peak flood-depth snapshot.
# Converts a LISFLOOD ESRI-ASCII grid (.max = max water depth) to a COG.
#
# LISFLOOD `latlong` grids are WGS84 lon/lat -> EPSG:4326.
# LISFLOOD dry/NODATA is -9999 (and large negatives); we set -9999 as nodata.
#
# Needs GDAL >= 3.1 (has the native COG driver). On the HPC: `module load gdal`
# or use the gdal in your anaconda3 env (`conda install -c conda-forge gdal`).
#
# Usage: ./01_make_cog.sh  /path/to/res_rain_Sorted_1198.bdy.max  flood_peak_depth_cog.tif

set -euo pipefail
IN="${1:?usage: 01_make_cog.sh <input.max> <output_cog.tif>}"
OUT="${2:?usage: 01_make_cog.sh <input.max> <output_cog.tif>}"

# gdal reads ESRI ASCII via the AAIGrid driver regardless of the .max extension
# when we point it at the file directly.
gdal_translate \
    -of COG \
    -a_srs EPSG:4326 \
    -a_nodata -9999 \
    -co COMPRESS=DEFLATE \
    -co PREDICTOR=2 \
    -co OVERVIEWS=AUTO \
    -co BLOCKSIZE=512 \
    "$IN" "$OUT"

echo "wrote $OUT"
echo "validate with:  python -m rio_cogeo validate $OUT   (pip install rio-cogeo)"
gdalinfo "$OUT" | sed -n '1,25p'
