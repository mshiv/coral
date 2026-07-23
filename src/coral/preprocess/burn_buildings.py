"""Burn building footprints into the (high-res) DEM for flow-around-buildings physics.

FEMA USA Structures / Microsoft US Building Footprints / OSM buildings -> rasterize onto
the DEM grid and edit terrain. Three standard methods (see docs/highres_pinpoint.md):
  - "block"      : raise cells under buildings to roof height (ground + add_height, or a
                   per-building height attribute) -> solid blocks water flows around.
                   Best physics; needs ~1-2 m so a building spans several cells.
  - "hole"       : raise very high (a wall) so footprint cells are effectively no-flow —
                   same diversion effect, avoids steep-gradient numerics.
  - "resistance" : leave the DEM; set very high Manning's n on footprints (works at
                   coarser resolution, but smooths out flow diversion).

Only meaningful on the 2 m (or finer) Pin Point clip. Deps: geopandas, rasterio, numpy.
  conda install -c conda-forge geopandas rasterio
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _read_asc(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.where(a <= -9990, np.nan, a)
    return a, h


def _write_asc(path, arr, header_src, nodata=-9999):
    with open(header_src) as f:
        hdr = [f.readline() for _ in range(6)]
    with open(path, "w") as f:
        f.writelines(hdr)
        np.savetxt(f, np.where(np.isnan(arr), nodata, arr), fmt="%.4f")


def burn_buildings(dem_asc, footprints, out_dem, *, method="block", add_height=4.0,
                   height_field=None, wall_height=50.0, manning_asc=None,
                   out_manning=None, building_n=2.0):
    """Edit the DEM (and optionally Manning) to represent buildings. `footprints` is any
    vector geopandas can read (GeoJSON/SHP/GPKG). Returns the count of footprint cells."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    dem, h = _read_asc(dem_asc)
    ny, nx = dem.shape; cs = h["cellsize"]
    transform = from_origin(h["xllcorner"], h["yllcorner"] + ny * cs, cs, cs)

    gdf = gpd.read_file(footprints).to_crs(4326)         # DEM grid is EPSG:4326
    if height_field and height_field in gdf.columns:     # rasterize per-building height
        shapes = [(geom, float(v) if v and v > 0 else add_height)
                  for geom, v in zip(gdf.geometry, gdf[height_field])]
        hgt = rasterize(shapes, out_shape=dem.shape, transform=transform, fill=0.0)
        mask = hgt > 0
    else:
        mask = rasterize([(g, 1) for g in gdf.geometry], out_shape=dem.shape,
                         transform=transform, fill=0).astype(bool)
        hgt = np.where(mask, add_height, 0.0)
    mask = mask & np.isfinite(dem)

    if method == "block":
        dem[mask] = dem[mask] + hgt[mask]                # raise to roof
    elif method == "hole":
        dem[mask] = dem[mask] + wall_height              # no-flow wall
    elif method == "resistance":
        if not (manning_asc and out_manning):
            raise SystemExit("resistance method needs --manning and --out-manning")
        man, _ = _read_asc(manning_asc)
        man[mask] = building_n
        _write_asc(out_manning, man, manning_asc)
    else:
        raise SystemExit(f"unknown method {method!r}")

    if method in ("block", "hole"):
        _write_asc(out_dem, dem, dem_asc)
    print(f"burned {int(mask.sum()):,} building cells ({method}) -> "
          f"{out_dem if method != 'resistance' else out_manning}")
    return int(mask.sum())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Burn building footprints into a DEM")
    ap.add_argument("--dem", required=True); ap.add_argument("--footprints", required=True)
    ap.add_argument("--out-dem", default="DEM_buildings.asc")
    ap.add_argument("--method", default="block", choices=["block", "hole", "resistance"])
    ap.add_argument("--add-height", type=float, default=4.0)
    ap.add_argument("--height-field", default=None, help="attribute with building height (m)")
    ap.add_argument("--manning", default=None); ap.add_argument("--out-manning", default=None)
    a = ap.parse_args()
    burn_buildings(a.dem, a.footprints, a.out_dem, method=a.method, add_height=a.add_height,
                   height_field=a.height_field, manning_asc=a.manning, out_manning=a.out_manning)
