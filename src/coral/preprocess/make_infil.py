"""Build a distributed infiltration grid for LISFLOOD (the `infilfile`, Sarhadi
Eq A1 'I' term) from POLARIS soil data.

Sarhadi et al. (2024) use a topsoil (0-50 cm) soil map for infiltration. LISFLOOD's
distributed infiltration is a RATE per cell (the `infilfile` is read and divided by
3.6e6, i.e. it expects mm/hr -> m/s; input.cpp:904). The physically correct soil
property for an infiltration rate is saturated hydraulic conductivity (Ksat).

POLARIS (Chaney et al. 2019) is SSURGO remapped to 30 m COGs over CONUS:
  http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/ksat/mean/<depth>/lat{S}{N}_lon{W}{E}.tif
  ksat is stored as log10(cm/hr). We take the topsoil depth-weighted mean over
  0-5, 5-15, 15-30, 30-60 cm (weights 5,10,15,20 -> ~0-50 cm), convert to mm/hr,
  and resample onto the exact DEM grid (same origin/res LISFLOOD needs).

Output: ESRI-ASCII grid (mm/hr), same header as the DEM, nodata -9999.

Deps: rasterio, numpy (downloads tiles via curl -k; Duke server has a cert quirk).
"""
from __future__ import annotations
import os, subprocess, tempfile, math
from pathlib import Path
import numpy as np

POLARIS = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/ksat/mean"
LAYERS = [("0_5", 5), ("5_15", 10), ("15_30", 15), ("30_60", 20)]  # ~0-50 cm


def _tiles(bbox):
    W, E, S, N = bbox
    out = []
    for lat in range(math.floor(S), math.ceil(N)):
        for lon in range(math.floor(W), math.ceil(E)):
            out.append(f"lat{lat}{lat+1}_lon{lon}{lon+1}")
    return out


def _fetch(url, dest):
    subprocess.run(["curl", "-k", "-s", "--max-time", "120", "-o", dest, url], check=True)
    return os.path.getsize(dest) > 1000


def make_infil(bbox, dem_path, out_asc, *, max_rate_mm_hr=None, cap_to_dem=True):
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling
    tiles = _tiles(bbox)
    with rasterio.open(dem_path) as d:
        dem_prof = d.profile; dem_tr = d.transform
        dem_h, dem_w = d.height, d.width
        dem = d.read(1)
        nod = d.nodata if d.nodata is not None else -9999

    tmp = tempfile.mkdtemp()
    ksat_mm_hr = np.zeros((dem_h, dem_w), "float64")  # depth-weighted accumulator
    wsum = 0
    for depth, wt in LAYERS:
        srcs = []
        for t in tiles:
            f = os.path.join(tmp, f"ksat_{depth}_{t}.tif")
            try:
                if _fetch(f"{POLARIS}/{depth}/{t}.tif", f):
                    srcs.append(rasterio.open(f))
            except Exception as e:
                print(f"  warn: {t} {depth} fetch failed ({e})")
        if not srcs:
            raise SystemExit(f"no POLARIS tiles for depth {depth}; check bbox/network")
        mosaic, mtr = merge(srcs)                     # log10(cm/hr)
        layer = np.full((dem_h, dem_w), np.nan, "float32")
        reproject(mosaic[0], layer, src_transform=mtr, src_crs="EPSG:4326",
                  dst_transform=dem_tr, dst_crs="EPSG:4326",
                  resampling=Resampling.bilinear)
        for s in srcs:
            s.close()
        # log10(cm/hr) -> cm/hr -> mm/hr
        ksat_mm_hr += np.nan_to_num(10.0 ** layer * 10.0) * wt
        wsum += wt
        print(f"  depth {depth}: median {np.nanmedian(10.0**layer*10.0):.1f} mm/hr")
    ksat_mm_hr /= wsum

    if max_rate_mm_hr:
        ksat_mm_hr = np.minimum(ksat_mm_hr, max_rate_mm_hr)
    if cap_to_dem:
        ksat_mm_hr = np.where(dem == nod, -9999.0, ksat_mm_hr)  # nodata outside domain

    # write ESRI-ASCII with the DEM header (LISFLOOD requires matching grid)
    x0 = dem_tr.c; cs = dem_tr.a
    y0 = dem_tr.f + dem_tr.e * dem_h         # yllcorner (e is negative)
    Path(out_asc).parent.mkdir(parents=True, exist_ok=True)
    with open(out_asc, "w") as f:
        f.write(f"ncols        {dem_w}\nnrows        {dem_h}\n")
        f.write(f"xllcorner    {x0:.8f}\nyllcorner    {y0:.8f}\n")
        f.write(f"cellsize     {cs:.10f}\nNODATA_value -9999\n")
        np.savetxt(f, ksat_mm_hr, fmt="%.4f")
    valid = ksat_mm_hr[ksat_mm_hr > -9990]
    print(f"wrote {out_asc}: {dem_h}x{dem_w}, infil {np.nanmin(valid):.1f}-"
          f"{np.nanmax(valid):.1f} mm/hr (median {np.nanmedian(valid):.1f})")
    return out_asc


def _polaris_layer_on_dem(var, depth, tiles, tmp, dem_tr, dem_h, dem_w, log10=False):
    """Fetch+mosaic a POLARIS variable/depth, reproject onto the DEM grid."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling
    base = f"http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{var}/mean"
    srcs = []
    for t in tiles:
        f = os.path.join(tmp, f"{var}_{depth}_{t}.tif")
        try:
            if _fetch(f"{base}/{depth}/{t}.tif", f):
                srcs.append(rasterio.open(f))
        except Exception as e:
            print(f"  warn: {var} {t} {depth} ({e})")
    if not srcs:
        raise SystemExit(f"no POLARIS {var} tiles for depth {depth}")
    mosaic, mtr = merge(srcs)
    out = np.full((dem_h, dem_w), np.nan, "float32")
    reproject(mosaic[0], out, src_transform=mtr, src_crs="EPSG:4326",
              dst_transform=dem_tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
    for s in srcs:
        s.close()
    return (10.0 ** out) if log10 else out


def make_capacity(bbox, dem_path, out_asc):
    """AWC infiltration-capacity grid (mm) for the `infilcapfile`: depth-weighted
    (theta_s - theta_r) over 0-50 cm x layer thickness. POLARIS theta_s/theta_r are
    linear volumetric fractions (cm3/cm3)."""
    import rasterio, tempfile
    tiles = _tiles(bbox); tmp = tempfile.mkdtemp()
    with rasterio.open(dem_path) as d:
        dem_tr, dem_h, dem_w = d.transform, d.height, d.width
        dem = d.read(1); nod = d.nodata if d.nodata is not None else -9999
    cap_mm = np.zeros((dem_h, dem_w), "float64")
    for depth, thick_cm in LAYERS:                      # LAYERS thickness in cm
        ts = _polaris_layer_on_dem("theta_s", depth, tiles, tmp, dem_tr, dem_h, dem_w)
        tr = _polaris_layer_on_dem("theta_r", depth, tiles, tmp, dem_tr, dem_h, dem_w)
        avail = np.clip(np.nan_to_num(ts) - np.nan_to_num(tr), 0, None)  # fraction
        cap_mm += avail * (thick_cm * 10.0)             # cm->mm thickness
        print(f"  depth {depth}: median avail storage {np.nanmedian(avail)*thick_cm*10:.1f} mm")
    cap_mm = np.where(dem == nod, -9999.0, cap_mm)
    x0 = dem_tr.c; cs = dem_tr.a; y0 = dem_tr.f + dem_tr.e * dem_h
    Path(out_asc).parent.mkdir(parents=True, exist_ok=True)
    with open(out_asc, "w") as f:
        f.write(f"ncols        {dem_w}\nnrows        {dem_h}\n")
        f.write(f"xllcorner    {x0:.8f}\nyllcorner    {y0:.8f}\n")
        f.write(f"cellsize     {cs:.10f}\nNODATA_value -9999\n")
        np.savetxt(f, cap_mm, fmt="%.3f")
    v = cap_mm[cap_mm > -9990]
    print(f"wrote {out_asc}: capacity {np.nanmin(v):.1f}-{np.nanmax(v):.1f} mm "
          f"(median {np.nanmedian(v):.1f})")
    return out_asc


def make_infil_ssurgo(soils_geojson, ksat_awc_json, dem_path, out_asc,
                       *, field="ksat_r_mm_hr", max_rate_mm_hr=None, cap_to_dem=True):
    """Alternative Ksat (or AWC) grid built from SSURGO map-unit polygons + the
    per-MUKEY ksat_r/awc_r table from fetch_ssurgo.py, rasterized onto the DEM
    grid. Optional comparison/alternative to the POLARIS-based make_infil().
    `field` selects which per-mukey value to burn in: "ksat_r_mm_hr" (rate grid,
    infilfile) or "awc_r" (fraction; multiply by profile depth for a capacity
    grid comparable to make_capacity's AWC mm, done by the caller/figure script).
    """
    import json
    import rasterio
    from rasterio.features import rasterize
    with rasterio.open(dem_path) as d:
        dem_tr = d.transform
        dem_h, dem_w = d.height, d.width
        dem = d.read(1)
        nod = d.nodata if d.nodata is not None else -9999

    soils = json.load(open(soils_geojson))
    table = json.load(open(ksat_awc_json))
    shapes = []
    missing = 0
    for f in soils["features"]:
        mukey = f["properties"].get("MUKEY")
        rec = table.get(mukey)
        if rec is None or rec.get(field) is None:
            missing += 1
            continue
        shapes.append((f["geometry"], float(rec[field])))
    if not shapes:
        raise SystemExit(f"no SSURGO {field} values resolved from {ksat_awc_json}")
    grid = rasterize(shapes, out_shape=(dem_h, dem_w), transform=dem_tr,
                      fill=np.nan, dtype="float64")
    if max_rate_mm_hr and field == "ksat_r_mm_hr":
        grid = np.minimum(grid, max_rate_mm_hr)
    if cap_to_dem:
        grid = np.where(dem == nod, -9999.0, np.nan_to_num(grid, nan=-9999.0))
    else:
        grid = np.nan_to_num(grid, nan=-9999.0)

    x0 = dem_tr.c; cs = dem_tr.a; y0 = dem_tr.f + dem_tr.e * dem_h
    Path(out_asc).parent.mkdir(parents=True, exist_ok=True)
    with open(out_asc, "w") as fh:
        fh.write(f"ncols        {dem_w}\nnrows        {dem_h}\n")
        fh.write(f"xllcorner    {x0:.8f}\nyllcorner    {y0:.8f}\n")
        fh.write(f"cellsize     {cs:.10f}\nNODATA_value -9999\n")
        np.savetxt(fh, grid, fmt="%.4f")
    valid = grid[grid > -9990]
    if valid.size:
        print(f"wrote {out_asc} (SSURGO {field}): {dem_h}x{dem_w}, "
              f"{np.nanmin(valid):.3f}-{np.nanmax(valid):.3f} "
              f"(median {np.nanmedian(valid):.3f}); {missing} polygons w/o data")
    return out_asc


def from_config(cfg, dem=None):
    """Build the infiltration grid(s) for a scenario:
      - always the Ksat rate grid (infilfile)  -> infil_<name>.asc
      - if forcing.infil_capped, also the AWC capacity grid (infilcapfile)
        -> infilcap_<name>.asc  (requires the patched, storage-limited lisflood)
    Returns the path(s) written."""
    if cfg.forcing.infiltration is None:
        raise SystemExit(f"{cfg.name}: forcing.infiltration is null")
    dem = dem or cfg.domain.dem
    rate = make_infil(cfg.domain.bbox, dem, f"data/interim/infil_{cfg.name}.asc")
    if cfg.forcing.infil_capped:
        cap = make_capacity(cfg.domain.bbox, dem, f"data/interim/infilcap_{cfg.name}.asc")
        return rate, cap
    return rate


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="POLARIS Ksat -> LISFLOOD infilfile (mm/hr)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dem", default=None, help="override DEM path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-rate", type=float, default=None, help="cap mm/hr")
    ap.add_argument("--ssurgo-soils-geojson", default=None,
                    help="optional: use SSURGO instead of POLARIS (sagis_soils_nrcs.geojson)")
    ap.add_argument("--ssurgo-table", default=None,
                    help="per-MUKEY ksat_r/awc_r json from fetch_ssurgo.py (required with --ssurgo-soils-geojson)")
    a = ap.parse_args()
    from coral import config
    c = config.load(a.config)
    dem = a.dem or c.domain.dem
    if a.ssurgo_soils_geojson:
        if not a.ssurgo_table:
            raise SystemExit("--ssurgo-table is required with --ssurgo-soils-geojson")
        make_infil_ssurgo(a.ssurgo_soils_geojson, a.ssurgo_table, dem,
                           a.out or f"data/interim/infil_ssurgo_{c.name}.asc",
                           max_rate_mm_hr=a.max_rate)
    else:
        make_infil(c.domain.bbox, dem, a.out or f"data/interim/infil_{c.name}.asc",
                   max_rate_mm_hr=a.max_rate)
