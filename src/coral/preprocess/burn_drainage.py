"""Burn the open drainage network (ditches, canals) into LISFLOOD-FP as sub-grid channels.

The SAGIS stormwater layers (fetch_sagis.py) split into two physics tracks:
  - Open channels: ditches, maintained ditches, canals. Surface conveyance that
    LISFLOOD-FP represents natively via sub-grid channels (SGC). Handled by this script.
  - Subsurface pipes: pipes/conduits/manholes/inlets/outfalls. A buried network that
    LISFLOOD-FP cannot represent alone; that is the separate SWMM 1D-2D coupling track.

SGC lets a channel narrower than a grid cell still carry flow: we rasterize the channel
centrelines to a per-cell width raster (metres). LISFLOOD reads `SGCwidth` and derives
a sub-grid bed below the cell's DEM bank so the creek/ditch conveys water the coarse grid
would otherwise miss. Optional `incise` mode instead lowers the DEM bed along the channel
(a crude non-SGC fallback for binaries built without SGC).

Only the open network is burned here. Widths come from a feature attribute (WIDTH), else a
per-type default. Lines are rasterized with all_touched so thin ditches stay connected.

Deps: geopandas, rasterio, numpy (same env as burn_buildings.py).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# fallback bankfull width (m) by network type when a feature has no usable WIDTH attribute
DEFAULT_WIDTH_M = {"canal": 8.0, "ditch": 3.0, "maintained_ditch": 3.0, "channel": 3.0}


def _read_asc(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    return np.where(a <= -9990, np.nan, a), h


def _write_asc(path, arr, header_src, nodata=-9999, fmt="%.4f"):
    with open(header_src) as f:
        hdr = [f.readline() for _ in range(6)]
    with open(path, "w") as f:
        f.writelines(hdr)
        np.savetxt(f, np.where(np.isnan(arr), nodata, arr), fmt=fmt)


def _sanitize_invert(v, sentinels=(999, 9999, -9999), lo=-50.0, hi=500.0):
    """Return a clean invert value (native units) or None. SAGIS uses 999 as 'missing'."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v in sentinels or v <= lo or v >= hi:
        return None
    return v


def burn_drainage(dem_asc, channels, out_width, *, kind="ditch", width_field=None,
                  width_scale=1.0, min_width=1.0, method="sgc", out_dem=None,
                  incise_depth=0.5, invert_fields=("INV_ELV_1", "INV_ELV_2"),
                  invert_scale=0.3048, max_drop=5.0):
    """Rasterize open channel centrelines onto the DEM grid.

    `channels` is any line vector (or comma-list) geopandas reads. Writes an SGC width
    raster (`out_width`, 0 = no channel). With method='incise'/'both', also lowers the DEM
    along the channel to give it a bed (needs `out_dem`).

    Bed geometry uses the SAGIS ditch invert fields (`invert_fields`) only as a relative
    drop, not an absolute elevation: those values are in feet with an unknown vertical
    datum (ELV_U_D='FT_DATUM_UNKNOWN') and use 999 as a missing sentinel, so their
    absolute value is untrustworthy. We take the difference between the two end inverts
    (times `invert_scale` -> metres, clamped to `max_drop`) to slope the incision along each
    segment (deeper toward the lower invert), on top of a base `incise_depth`. Where a
    clean invert pair is absent we fall back to a flat `incise_depth`. `width_scale`
    converts the WIDTH attribute to metres (0.3048 for feet). Returns channel-cell count.
    """
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    dem, h = _read_asc(dem_asc)
    ny, nx = dem.shape; cs = h["cellsize"]
    transform = from_origin(h["xllcorner"], h["yllcorner"] + ny * cs, cs, cs)

    paths = channels.split(",") if isinstance(channels, str) else list(channels)
    gdf = pd.concat([gpd.read_file(p).to_crs(4326) for p in paths], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)

    default_w = DEFAULT_WIDTH_M.get(kind, 3.0)
    def _w(row):
        if width_field and width_field in gdf.columns:
            v = row.get(width_field)
            if v is not None and np.isfinite(v) and v > 0:
                return max(float(v) * width_scale, min_width)
        return default_w

    from shapely.geometry import LineString

    def _endpoint_depths(row):
        """(d_first, d_last) incision depths (m) for a feature's vertex sequence, using the
        invert *drop* to slope the bed; flat incise_depth when a clean pair is absent."""
        i1 = _sanitize_invert(row.get(invert_fields[0])) if invert_fields[0] in gdf.columns else None
        i2 = _sanitize_invert(row.get(invert_fields[1])) if invert_fields[1] in gdf.columns else None
        if i1 is None or i2 is None:
            return incise_depth, incise_depth, False
        drop = min(abs(i1 - i2) * invert_scale, max_drop)          # metres, clamped
        deep, shallow = incise_depth + drop, incise_depth
        return (shallow, deep, True) if i1 >= i2 else (deep, shallow, True)

    def _parts(geom):
        return list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]

    # rasterize per-feature width (widest wins) + a sloped incision-depth raster
    width = np.zeros(dem.shape, dtype="float32")
    depth = np.zeros(dem.shape, dtype="float32")
    n_sloped = 0
    for geom, row in zip(gdf.geometry, (r for _, r in gdf.iterrows())):
        if geom is None or geom.is_empty:
            continue
        m = rasterize([(geom, 1)], out_shape=dem.shape, transform=transform,
                      fill=0, all_touched=True).astype(bool)
        width[m] = np.maximum(width[m], _w(row))
        if method == "sgc":
            continue
        d_first, d_last, sloped = _endpoint_depths(row)
        n_sloped += sloped
        coords = [c for part in _parts(geom) for c in part.coords]
        if len(coords) < 2:
            continue
        seglen = [LineString([coords[k], coords[k + 1]]).length for k in range(len(coords) - 1)]
        L = sum(seglen) or 1.0
        s = 0.0
        for k in range(len(coords) - 1):
            f0, f1 = s / L, (s + seglen[k]) / L
            d = d_first + (d_last - d_first) * (0.5 * (f0 + f1))    # depth at sub-seg midpoint
            sm = rasterize([(LineString([coords[k], coords[k + 1]]), 1)], out_shape=dem.shape,
                           transform=transform, fill=0, all_touched=True).astype(bool)
            depth[sm] = np.maximum(depth[sm], d)
            s += seglen[k]

    chan = (width > 0) & np.isfinite(dem)
    width[~chan] = 0.0
    _write_asc(out_width, width, dem_asc, nodata=0, fmt="%.3f")
    n = int(chan.sum())
    print(f"burned {n:,} open-channel cells ({kind}, {len(gdf)} features) -> {out_width}")

    if method in ("incise", "both"):
        if not out_dem:
            raise SystemExit("incise/both needs --out-dem")
        cut = chan & (depth > 0)
        dem[cut] = dem[cut] - depth[cut]
        _write_asc(out_dem, dem, dem_asc)
        print(f"  incised bed (base {incise_depth} m, invert-sloped on {n_sloped}/{len(gdf)} "
              f"features) -> {out_dem}")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Burn open drainage (ditches/canals) into LISFLOOD SGC")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--channels", required=True, help="line vector(s), comma-separated")
    ap.add_argument("--out-width", default="SGCwidth.asc")
    ap.add_argument("--kind", default="ditch", choices=list(DEFAULT_WIDTH_M))
    ap.add_argument("--width-field", default=None, help="attribute holding channel width")
    ap.add_argument("--width-scale", type=float, default=1.0, help="attribute->metres (0.3048 for ft)")
    ap.add_argument("--min-width", type=float, default=1.0)
    ap.add_argument("--method", default="sgc", choices=["sgc", "incise", "both"])
    ap.add_argument("--out-dem", default=None)
    ap.add_argument("--incise-depth", type=float, default=0.5)
    a = ap.parse_args()
    burn_drainage(a.dem, a.channels, a.out_width, kind=a.kind, width_field=a.width_field,
                  width_scale=a.width_scale, min_width=a.min_width, method=a.method,
                  out_dem=a.out_dem, incise_depth=a.incise_depth)
