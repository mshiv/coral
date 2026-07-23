#!/usr/bin/env python3
"""
Land-masked peak-flood-depth COG. The raw .max includes permanently-wet ocean/
channels (tens of metres); this masks to LAND (DEM > sea level) and drops sub-
threshold depths, so the web layer shows inland inundation only.

Writes a Cloud-Optimized GeoTIFF (EPSG:4326, nodata -9999) via rasterio.

Deps: numpy, rasterio
Usage:
  python 01b_make_flood_cog.py --max ../lisflood/results_matthew_sav/res_matthew_sav.max \
      --dem ../inputs/SUB_DEM_SAV.asc --out flood_peak_depth_land_cog.tif [--sea-level 0.81]
"""
import argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin


def read_grid(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f).astype("float32")
    a = np.where((a == h.get("nodata_value", -9999)) | (a <= -9990), np.nan, a)
    return a, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", required=True, help="LISFLOOD .max grid")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", default="flood_peak_depth_land_cog.tif")
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--depth-floor", type=float, default=0.05)
    args = ap.parse_args()

    depth, h = read_grid(args.max)
    dem, _ = read_grid(args.dem)
    land = dem > args.sea_level
    d = np.where(land & (depth >= args.depth_floor), depth, np.nan).astype("float32")
    d = np.where(np.isnan(d), -9999.0, d).astype("float32")

    nx, ny, cs = int(h["ncols"]), int(h["nrows"]), h["cellsize"]
    top = h["yllcorner"] + ny * cs
    transform = from_origin(h["xllcorner"], top, cs, cs)

    profile = dict(driver="COG", dtype="float32", count=1, height=ny, width=nx,
                   crs="EPSG:4326", transform=transform, nodata=-9999.0,
                   compress="DEFLATE", blocksize=512)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(d, 1)

    wet = d[d > 0]
    print(f"wrote {args.out}")
    print(f"land flood: {wet.size:,} cells, max {wet.max():.2f} m, mean {wet.mean():.2f} m, "
          f"area ~{wet.size * cs*cs * (111320**2) * np.cos(np.radians(32)) / 1e6:.1f} km^2")


if __name__ == "__main__":
    main()
