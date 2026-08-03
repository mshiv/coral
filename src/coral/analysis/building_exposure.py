"""Peak flood depth sampled at building footprints: the consequence metric.

`exposure()` samples a LISFLOOD `.max` at each footprint and returns one row per building plus
a summary at several depth thresholds. The thresholds are:

  0.05 m  wet at all; below this is noise in a 30 m local-inertial model
  0.15 m  ankle deep; the usual "nuisance flooding" line, damage to floors and utilities
  0.30 m  the NFIP/HAZUS first-floor damage threshold for slab-on-grade construction
  1.00 m  life-safety relevant for adults on foot, and structural for single-storey homes

CAVEAT: LISFLOOD depth is ground-surface depth, not depth inside a building. 
Pin Point has many raised and pier-built homes, so ground depth overstates
interior flooding for those. `--first-floor-m` applies a uniform freeboard offset, which is a
stand-in for a proper first-floor-elevation survey; without one, report ground depth and
say so rather than implying interior damage.

Sampling is zonal maximum over every cell a footprint touches.
Deps: numpy, geopandas, rasterio.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# (label, metres) — see the module docstring for why these four
THRESHOLDS = (("wet_0.05m", 0.05), ("nuisance_0.15m", 0.15),
              ("damage_0.30m", 0.30), ("severe_1.0m", 1.00))


def _grid(dem_asc):
    """(shape, rasterio transform) for an ESRI ASCII grid in EPSG:4326."""
    from rasterio.transform import from_origin
    h = {}
    with open(dem_asc) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    ny, nx, cs = int(h["nrows"]), int(h["ncols"]), h["cellsize"]
    return (ny, nx), from_origin(h["xllcorner"], h["yllcorner"] + ny * cs, cs, cs)


_DEPTH_CACHE = {}
_DEPTH_CACHE_MAX = 4


def read_depth(max_asc, cache=True):
    """Read a `.max` as a depth array with nodata and negatives set to 0.

    Small cache because an ensemble comparison reads one baseline against every member at its
    sea level, so the same baseline grid is re-read a few hundred times in a row. Four entries
    is enough for that access pattern and costs about 40 MB. Callers must not mutate the
    returned array in place; every use here derives new arrays.
    """
    key = str(max_asc)
    if cache and key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    a = np.loadtxt(max_asc, skiprows=6)
    d = np.where((a <= -9990) | ~np.isfinite(a), 0.0, np.clip(a, 0, None))
    if cache:
        if len(_DEPTH_CACHE) >= _DEPTH_CACHE_MAX:
            _DEPTH_CACHE.pop(next(iter(_DEPTH_CACHE)))
        _DEPTH_CACHE[key] = d
    return d


_IDS_CACHE = {}


def building_ids(footprints_geojson, dem_asc):
    """Rasterized footprint index+1 on the DEM grid, memoized.

    This depends only on the footprints and the grid, so it is identical for every member of
    an ensemble. Recomputing it per member costs a few tenths of a second each, which over
    1506 members compared against their baselines is several thousand redundant
    rasterizations. Keyed on both paths plus the grid shape.
    """
    from rasterio.features import rasterize
    import geopandas as gpd
    shape, tr = _grid(dem_asc)
    key = (str(footprints_geojson), str(dem_asc), shape)
    hit = _IDS_CACHE.get(key)
    if hit is not None:
        return hit
    gdf = gpd.read_file(footprints_geojson).to_crs(4326).reset_index(drop=True)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        raise SystemExit(f"no usable geometry in {footprints_geojson}")
    ids = rasterize([(g, i + 1) for i, g in enumerate(gdf.geometry)],
                    out_shape=shape, transform=tr, fill=0, all_touched=True, dtype="int32")
    _IDS_CACHE[key] = (ids, gdf, shape, tr)
    return _IDS_CACHE[key]


def removed_buildings(footprints_geojson, dem_asc, base_dem, adapt_dem, drop_m=0.10):
    """Boolean per building: was it inside ground that the adaptation lowered?

    Managed retreat buys out and demolishes structures, then regrades the land to natural
    elevation. The footprint file is fixed across the ensemble, so without this those
    demolished houses keep being counted as flooded in the retreat scenarios, which conflates
    "water moved onto neighbours" with "we scored flooding at houses that no longer exist".

    A building counts as removed if any cell its footprint touches was lowered by more than
    `drop_m`. Lowering is the retreat signature specifically: a seawall raises the DEM and a
    marsh does not change it at all, so this does not fire for them.
    """
    ids, gdf, shape, _ = building_ids(footprints_geojson, dem_asc)
    lowered = np.nan_to_num(base_dem, nan=0.0) - np.nan_to_num(adapt_dem, nan=0.0) > drop_m
    out = np.zeros(len(gdf) + 1, bool)
    flat_ids = ids.ravel()
    hit = (flat_ids > 0) & lowered.ravel()
    out[flat_ids[hit]] = True
    return out[1:]


def exposure(max_asc, footprints_geojson, dem_asc, *, first_floor_m=0.0,
             thresholds=THRESHOLDS, id_field=None, depth=None, exclude=None):
    """Sample peak depth at every building footprint.

    Returns (rows, summary). `rows` is a list of dicts, one per building, with the zonal-max
    ground depth and the depth after the first-floor offset. `summary` counts buildings above
    each threshold, plus the mean and 90th-percentile depth over the flooded ones.
    """
    # `depth` lets a caller pass an already-read grid, so a pair comparison reads each .max
    # once rather than once for the domain metrics and again here.
    ids, gdf, shape, tr = building_ids(footprints_geojson, dem_asc)
    if depth is None:
        depth = read_depth(max_asc)
    if depth.shape != shape:
        raise SystemExit(f"{Path(max_asc).name} is {depth.shape} but the DEM grid is {shape}. "
                         "The .max must come from a run on this DEM.")

    # zonal max in one pass rather than per-polygon masking, which would be O(n_buildings) reads
    zmax = np.zeros(len(gdf) + 1, "float64")
    flat_ids, flat_dep = ids.ravel(), depth.ravel()
    hit = flat_ids > 0
    np.maximum.at(zmax, flat_ids[hit], flat_dep[hit])
    covered = np.zeros(len(gdf) + 1, bool)
    covered[flat_ids[hit]] = True

    # Footprints too small to touch any cell centre still get rasterized by all_touched, but a
    # polygon entirely outside the grid will not. Fall back to the centroid cell for those, and
    # count them, because silently dropping buildings would bias every count downward.
    outside = 0
    miss = np.where(~covered[1:])[0]
    if len(miss):
        from rasterio.transform import rowcol
        for i in miss:
            c = gdf.geometry.iloc[i].centroid
            try:
                r, cc = rowcol(tr, c.x, c.y)
            except Exception:
                outside += 1; continue
            if 0 <= r < shape[0] and 0 <= cc < shape[1]:
                zmax[i + 1] = depth[r, cc]
            else:
                outside += 1

    labels = list(gdf[id_field].astype(str)) if id_field and id_field in gdf.columns \
        else [str(i) for i in range(len(gdf))]
    ground = zmax[1:]
    keep = np.ones(len(gdf), bool) if exclude is None else ~np.asarray(exclude, bool)
    interior = np.clip(ground - first_floor_m, 0, None)
    rows = [{"building_id": labels[i], "depth_ground_m": float(ground[i]),
             "depth_interior_m": float(interior[i])} for i in range(len(gdf))]

    d = (interior if first_floor_m else ground)[keep]
    flooded = d[d > thresholds[0][1]]
    summary = {"n_buildings": int(keep.sum()),
               "n_excluded": int((~keep).sum()),
               "n_outside_grid": int(outside),
               "first_floor_m": float(first_floor_m),
               "mean_depth_flooded_m": float(flooded.mean()) if flooded.size else 0.0,
               "p90_depth_flooded_m": float(np.percentile(flooded, 90)) if flooded.size else 0.0,
               "total_depth_m": float(d.sum())}
    for label, t in thresholds:
        summary[f"n_{label}"] = int((d > t).sum())
    return rows, summary


def _main(argv=None):
    import argparse, csv, json
    ap = argparse.ArgumentParser(description="Peak flood depth at building footprints")
    ap.add_argument("--max", required=True, help="LISFLOOD .max peak-depth grid")
    ap.add_argument("--footprints", required=True, help="building footprint geojson")
    ap.add_argument("--dem", required=True, help="the DEM the run used, for the grid")
    ap.add_argument("--first-floor-m", type=float, default=0.0,
                    help="uniform freeboard offset; ground depth is reported without it")
    ap.add_argument("--id-field", default=None)
    ap.add_argument("--out-csv", default=None, help="per-building depths")
    ap.add_argument("--out-json", default=None, help="summary counts")
    a = ap.parse_args(argv)

    rows, summary = exposure(a.max, a.footprints, a.dem,
                             first_floor_m=a.first_floor_m, id_field=a.id_field)
    for k, v in summary.items():
        print(f"  {k:24s} {v}")
    if a.out_csv:
        Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"per-building depths -> {a.out_csv}")
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(a.out_json, "w"), indent=2)
        print(f"summary -> {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
