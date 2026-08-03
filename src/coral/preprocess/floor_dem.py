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


def floor_dem(dem_asc, out_asc, floor=-1.0, nodata=-9999.0, offset=0.0):
    """Shift, then floor, then normalise nodata.

    `offset` exists because a source DEM can carry the wrong vertical datum. The CoNED Georgia
    2022 tile declares NAVD88 in its COMPOUNDCRS but holds ellipsoidal heights, so its values
    sit about 32.6 m low, which is the GEOID18 separation for the Georgia coast. Flooring that
    at -1 m clamps every cell to the floor and produces a DEM with no land in it, which passes
    every existence check downstream. Determine the offset by comparing against a DEM of known
    datum rather than assuming a geoid value.
    """
    with open(dem_asc) as f:
        hdr = [f.readline() for _ in range(6)]
    a = np.loadtxt(dem_asc, skiprows=6)
    # GDAL AAIGrid carries the source sentinel through, and CoNED uses 3.4e38, which is greater
    # than -9990 and so passed the old test as valid data.
    valid = (a > -9990) & (a < 1e30) & np.isfinite(a)
    a = np.where(valid, np.maximum(a + offset, floor), nodata)
    hdr = [h if not h.lower().startswith("nodata") else f"NODATA_value {nodata:.0f}\n" for h in hdr]
    with open(out_asc, "w") as f:
        f.writelines(hdr)
        np.savetxt(f, a, fmt="%.4f")
    v = a[valid]
    at_floor = float((v <= floor + 1e-6).mean())
    print(f"offset {offset:+.3f} m, floored at {floor} m, nodata -> {nodata:.0f}: {out_asc}")
    print(f"  min {v.min():.2f}, median {np.median(v):.2f}, max {v.max():.2f}, "
          f"{len(np.unique(v))} unique values")
    print(f"  fraction at the floor: {at_floor:.1%}")
    if at_floor > 0.5 or len(np.unique(v)) < 100:
        print("  WARNING: this DEM is mostly or entirely at the floor, so it carries no "
              "topography. Check the vertical datum of the source before using it.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Floor an ASCII DEM and normalize nodata")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=-1.0)
    ap.add_argument("--offset", type=float, default=0.0,
                    help="vertical shift applied BEFORE flooring, e.g. +32.588 for the CoNED "
                         "Georgia tile, which holds ellipsoidal heights despite declaring NAVD88")
    a = ap.parse_args()
    floor_dem(a.dem, a.out, floor=a.floor, offset=a.offset)
