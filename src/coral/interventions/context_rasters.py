"""Rasterize SAGIS context layers onto a DEM grid for SAGIS-conditioned intervention siting.

Turns the vector datasets into DEM-aligned arrays that `generate.apply_intervention` consumes:
  - wetlands_mask(nwi)      -> bool  : real marsh footprint (NWI) for marsh/mangrove siting
  - soil_ksat_grid(ssurgo)  -> float : per-cell SSURGO Ksat (mm/hr) to cap de-pave/permeable
  - buildings_mask(fema)    -> bool  : real building footprints for managed-retreat siting

Same ASC grid + rasterio pattern as preprocess/burn_buildings.py. Deps: geopandas, rasterio.
"""
from __future__ import annotations
import json
import numpy as np


def _grid(dem_asc):
    """Return (shape, rasterio transform) for an ESRI ASCII DEM (EPSG:4326)."""
    from rasterio.transform import from_origin
    h = {}
    with open(dem_asc) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    ny, nx, cs = int(h["nrows"]), int(h["ncols"]), h["cellsize"]
    return (ny, nx), from_origin(h["xllcorner"], h["yllcorner"] + ny * cs, cs, cs)


def wetlands_mask(nwi_geojson, dem_asc, *, cowardin_field="ATTRIBUTE",
                  cowardin_prefixes=None):
    """Boolean marsh/wetland footprint on the DEM grid from NWI polygons. By default
    every wetland polygon counts; pass e.g. cowardin_prefixes=("E2EM",) to restrict to
    estuarine emergent (salt marsh)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    shape, tr = _grid(dem_asc)
    gdf = gpd.read_file(nwi_geojson).to_crs(4326)
    if cowardin_prefixes and cowardin_field in gdf.columns:
        pref = tuple(cowardin_prefixes)
        gdf = gdf[gdf[cowardin_field].astype(str).str.startswith(pref)]
    if len(gdf) == 0:
        return np.zeros(shape, bool)
    return rasterize([(g, 1) for g in gdf.geometry if g is not None],
                     out_shape=shape, transform=tr, fill=0, all_touched=True).astype(bool)


def soil_ksat_grid(soils_geojson, ssurgo_table_json, dem_asc, *, mukey_field="MUKEY",
                   value_key="ksat_r_mm_hr"):
    """Per-cell SSURGO Ksat (mm/hr) on the DEM grid: join each soil polygon's MUKEY to
    the fetch_ssurgo table and rasterize `value_key`. Cells outside any polygon = NaN
    (treated as 'no cap' downstream)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    shape, tr = _grid(dem_asc)
    table = json.load(open(ssurgo_table_json))
    gdf = gpd.read_file(soils_geojson).to_crs(4326)
    shapes = []
    for g, mk in zip(gdf.geometry, gdf[mukey_field].astype(str)):
        rec = table.get(mk)
        if g is not None and rec and rec.get(value_key) is not None:
            shapes.append((g, float(rec[value_key])))
    if not shapes:
        return np.full(shape, np.nan, "float64")
    return rasterize(shapes, out_shape=shape, transform=tr, fill=np.nan,
                     all_touched=True, dtype="float64")


def buildings_mask(footprints_geojson, dem_asc):
    """Boolean building-footprint mask on the DEM grid (FEMA USA Structures / DPLAT / OSM)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    shape, tr = _grid(dem_asc)
    gdf = gpd.read_file(footprints_geojson).to_crs(4326)
    if len(gdf) == 0:
        return np.zeros(shape, bool)
    return rasterize([(g, 1) for g in gdf.geometry if g is not None],
                     out_shape=shape, transform=tr, fill=0, all_touched=True).astype(bool)
