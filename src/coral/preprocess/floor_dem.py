"""Floor an ASCII DEM at a minimum elevation and normalize the nodata sentinel.

Deep bathymetry (tidal channels to ~-10 m) collapses the LISFLOOD timestep
(dt = CFL*dx/sqrt(g*h)). Flooring the DEM at ~-1 m caps the water depth and keeps the run
tractable; the surge inundation is carried in through the nested boundary, so deep-channel
exchange is secondary. Also rewrites NODATA_value to -9999 (GDAL AAIGrid writes a huge float).

  python -m coral.preprocess.floor_dem --dem coned_pinpoint_4m.asc --floor -1 \
      --out data/raw/coned_pinpoint_4m_floored.asc
"""
from __future__ import annotations
import numpy as np


def floor_dem(dem_asc, out_asc, floor=-1.0, nodata=-9999.0):
    with open(dem_asc) as f:
        hdr = [f.readline() for _ in range(6)]
    a = np.loadtxt(dem_asc, skiprows=6)
    valid = a > -9990                         # existing data cells
    a = np.where(valid, np.maximum(a, floor), nodata)
    hdr = [h if not h.lower().startswith("nodata") else f"NODATA_value {nodata:.0f}\n" for h in hdr]
    with open(out_asc, "w") as f:
        f.writelines(hdr)
        np.savetxt(f, a, fmt="%.4f")
    print(f"floored at {floor} m, nodata -> {nodata:.0f}: {out_asc}  "
          f"(min {np.nanmin(np.where(valid, a, np.nan)):.2f}, max {a[valid].max():.2f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Floor an ASCII DEM and normalize nodata")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=-1.0)
    a = ap.parse_args()
    floor_dem(a.dem, a.out, floor=a.floor)
