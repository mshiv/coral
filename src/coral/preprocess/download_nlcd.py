#!/usr/bin/env python3
"""
Download an NLCD land-cover subset for the LISFLOOD domain from the MRLC WCS.

Reads the DEM bounds (EPSG:4326), pads them, reprojects to NLCD's native CRS
(EPSG:5070 Albers), and fetches a GeoTIFF subset via the MRLC GeoServer WCS.
No login required.

Deps: requests, pyproj, rasterio
Usage:
  python download_nlcd.py --dem inputs/SUB_DEM_SAV.asc --out inputs/nlcd_savannah.tif
  python download_nlcd.py --dem ... --year 2019      # 2016 (default), 2019, 2021
"""
import argparse
import sys
import requests
import rasterio
from pyproj import Transformer

WCS = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
COVERAGE = "mrlc_download__NLCD_{year}_Land_Cover_L48"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="scenario YAML (provides DEM path)")
    ap.add_argument("--dem", help="DEM whose bbox sets the download extent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=2016,
                    help="NLCD vintage (2016/2019/2021); 2016 matches Matthew")
    ap.add_argument("--margin", type=float, default=0.015,
                    help="degrees of padding around the DEM bbox")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    dem = args.dem
    if args.config and not dem:
        from coral import config
        dem = config.load(args.config).domain.dem
    if not dem:
        raise SystemExit("need --config or --dem")

    with rasterio.open(dem) as d:
        b = d.bounds
    W, E = b.left - args.margin, b.right + args.margin
    S, N = b.bottom - args.margin, b.top + args.margin

    t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = zip(*[t.transform(lo, la)
                   for lo, la in [(W, S), (W, N), (E, S), (E, N)]])
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    print(f"DEM bbox -> Albers X[{x1:.0f},{x2:.0f}] Y[{y1:.0f},{y2:.0f}]")

    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": COVERAGE.format(year=args.year),
        "format": "image/tiff",
        "subset": [f"X({x1},{x2})", f"Y({y1},{y2})"],
    }
    print(f"requesting NLCD {args.year} from MRLC WCS ...")
    r = requests.get(WCS, params=params, timeout=args.timeout)
    r.raise_for_status()
    if not r.content[:4] in (b"II*\x00", b"MM\x00*"):  # TIFF magic
        sys.exit(f"response is not a GeoTIFF (got {r.content[:80]!r})")
    with open(args.out, "wb") as f:
        f.write(r.content)

    with rasterio.open(args.out) as s:
        import numpy as np
        a = s.read(1)
        u, c = np.unique(a, return_counts=True)
        print(f"wrote {args.out}  {s.shape} {s.crs}")
        print("NLCD classes (code: cells):",
              {int(k): int(v) for k, v in zip(u, c)})


if __name__ == "__main__":
    main()
