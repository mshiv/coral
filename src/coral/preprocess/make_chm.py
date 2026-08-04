"""Build a canopy height model for the Pin Point clip from USGS 3DEP lidar.

Feeds `marsh_corrections dem --chm` (lidar canopy bias in the marsh DEM) and
`marsh_corrections manning --chm` (within-class modulation of n by vegetation height).

WHY: 3DEP GA_Statewide_2018_B18_DRRA / GA_Statewide_B5_2018 is the only candidate
that resolves sub-metre marsh vegetation at the site with verified full coverage of the clip
(21 tiles, ~400 MB). It carries a NAVD88 Geoid12B vertical datum, avoiding the
ellipsoidal-height ambiguity in the CoNED VRT used earlier. It is the same data
type Hladik and Alber (2012) corrected against for Georgia salt marsh, so their method transfers.
LANDFIRE EVH (30 m) and the global canopy products (Meta/WRI, ETH) are too coarse for a 4 m grid
and are trained on forest, not Spartina.

METHOD, per grid cell:
    ground  = minimum Z among ASPRS class 2 (ground) returns
    surface = maximum Z among all returns
    CHM     = surface - ground, clamped at zero
Cells with no ground return fall back to the run DEM. This is first-return-minus-ground, deemed 
sufficient for sub-metre marsh canopy. 

NOTE: Over dense Spartina, lidar often fails to reach true ground and class-2
returns sit within the canopy. The CHM then UNDERESTIMATES canopy height, and the classified
ground is itself biased high.

    python -m coral.preprocess.make_chm list  --bbox -81.1103 31.9367 -81.0727 31.9690
    python -m coral.preprocess.make_chm fetch --bbox ... --out-dir data/raw/lidar
    python -m coral.preprocess.make_chm build --laz-dir data/raw/lidar \\
        --dem runs/pinpoint_highres_4m/SUB_DEM_pinpoint_highres_4m.asc --out chm_4m.asc
"""
import argparse
import glob
import json
import os
import urllib.parse
import urllib.request

import numpy as np

TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
GROUND_CLASS = 2
# ASPRS noise classes. Birds, cloud and multipath returns sit tens to hundreds of metres above
# the surface, and because the surface accumulator is a per-cell MAXIMUM a single noise point
# sets the canopy height for that cell. Leaving them in gave a 590 m canopy over a salt marsh.
NOISE_CLASSES = (7, 18)
# Nothing at Pin Point is taller than this. Applied after the difference so that a noise point
# that survives classification cannot propagate into the DEM correction.
MAX_CANOPY_M = 45.0


def list_tiles(bbox, max_items=60):
    q = urllib.parse.urlencode({
        "bbox": ",".join(str(v) for v in bbox),
        "datasets": "Lidar Point Cloud (LPC)", "max": max_items})
    with urllib.request.urlopen(f"{TNM}?{q}", timeout=60) as r:
        d = json.load(r)
    tiles = [(i["title"], i["downloadURL"], i.get("sizeInBytes", 0)) for i in d.get("items", [])]
    print(f"{len(tiles)} tiles, {sum(t[2] for t in tiles)/1e6:.0f} MB total")
    return tiles


def fetch(bbox, out_dir, max_items=60):
    os.makedirs(out_dir, exist_ok=True)
    for title, url, size in list_tiles(bbox, max_items):
        dst = os.path.join(out_dir, os.path.basename(url))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  have {os.path.basename(dst)}"); continue
        print(f"  get  {os.path.basename(dst)} ({size/1e6:.0f} MB)")
        urllib.request.urlretrieve(url, dst)
    return out_dir


def _read_dem(dem_asc):
    h = {}
    with open(dem_asc) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    h["ncols"], h["nrows"] = int(h["ncols"]), int(h["nrows"])
    return np.loadtxt(dem_asc, skiprows=6), h


def build(laz_dir, dem_asc, out, nodata=-9999.0, dem_epsg=4326):
    try:
        import laspy
    except ImportError:
        raise SystemExit("needs laspy: pip install 'laspy[lazrs]' pyproj")
    from pyproj import Transformer

    dem, h = _read_dem(dem_asc)
    ny, nx, cs = h["nrows"], h["ncols"], h["cellsize"]
    x0, y1 = h["xllcorner"], h["yllcorner"] + ny * cs   # top-left; ASCII grids run top-down
    gmin = np.full((ny, nx), np.inf)
    # Max is tracked as a running minimum of the NEGATED height, so both accumulators can use
    # np.minimum.at (there is no unbuffered maximum.at that behaves identically across dtypes).
    # This must start at +inf, not -inf: -inf would win every comparison and nothing would
    # accumulate.
    snegmin = np.full((ny, nx), np.inf)

    files = sorted(glob.glob(os.path.join(laz_dir, "*.la[sz]")))
    if not files:
        raise SystemExit(f"no .las/.laz in {laz_dir}")
    for k, f in enumerate(files, 1):
        with laspy.open(f) as fh:
            tr = Transformer.from_crs(fh.header.parse_crs(), f"EPSG:{dem_epsg}", always_xy=True)
            for pts in fh.chunk_iterator(5_000_000):
                lon, lat = tr.transform(np.asarray(pts.x), np.asarray(pts.y))
                z = np.asarray(pts.z)
                col = ((lon - x0) / cs).astype(int)
                row = ((y1 - lat) / cs).astype(int)
                cl = np.asarray(pts.classification)
                keep = ~np.isin(cl, NOISE_CLASSES)
                # Withheld points are flagged as unusable by the vendor's own QC.
                wh = getattr(pts, "withheld", None)
                if wh is not None:
                    keep &= ~np.asarray(wh).astype(bool)
                ok = ((col >= 0) & (col < nx) & (row >= 0) & (row < ny)
                      & np.isfinite(z) & keep)
                if not ok.any():
                    continue
                r, c, zz = row[ok], col[ok], z[ok]
                np.minimum.at(snegmin, (r, c), -zz)
                g = cl[ok] == GROUND_CLASS
                if g.any():
                    np.minimum.at(gmin, (r[g], c[g]), zz[g])
        print(f"  [{k}/{len(files)}] {os.path.basename(f)}")

    surface = -snegmin
    ground = np.where(np.isfinite(gmin), gmin, np.where(dem == nodata, np.nan, dem))
    chm = np.where(np.isfinite(surface), surface - ground, 0.0)
    chm = np.clip(np.nan_to_num(chm, nan=0.0, posinf=0.0, neginf=0.0), 0.0, MAX_CANOPY_M)

    with open(out, "w") as f:
        f.write(f"ncols        {nx}\nnrows        {ny}\n"
                f"xllcorner    {h['xllcorner']:.12f}\nyllcorner    {h['yllcorner']:.12f}\n"
                f"cellsize     {cs:.12f}\nNODATA_value {nodata:.0f}\n")
        np.savetxt(f, chm, fmt="%.3f")
    cov = np.isfinite(gmin).mean()
    print(f"CHM -> {out}: {chm.min():.2f}-{chm.max():.2f} m, mean {chm.mean():.2f}, "
          f"p50 {np.percentile(chm,50):.2f}, p99 {np.percentile(chm,99):.2f}; "
          f"ground returns in {100*cov:.1f}% of cells")
    if cov < 0.5:
        print("  WARNING: sparse ground coverage; CHM falls back to the DEM in most cells.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "fetch"):
        p = s.add_parser(name)
        p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
        if name == "fetch":
            p.add_argument("--out-dir", required=True)
    b = s.add_parser("build")
    b.add_argument("--laz-dir", required=True); b.add_argument("--dem", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "list":
        list_tiles(a.bbox)
    elif a.cmd == "fetch":
        fetch(a.bbox, a.out_dir)
    else:
        build(a.laz_dir, a.dem, a.out)


if __name__ == "__main__":
    main()
