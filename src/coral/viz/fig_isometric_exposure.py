"""Figure 5 — freeboard at house scale, drawn in axonometric view.

A plan-view depth map collapses the vertical axis. It gives depth above ground at a cell, but
the question that decides whether a house floods is the water surface elevation against that
house's finished floor. This view keeps the vertical axis, so the gap between the two is visible
as a gap.

Projection is axonometric (parallel), not perspective. Lengths along each axis stay to scale, so
the drawing can still be measured. Structures are extruded by the FEMA `HEIGHT` field where it
exists and by the median height where it does not, and those two cases are drawn differently, so
no assumed height is presented as a measurement.

First-floor elevation is not in any available dataset. It is assumed to sit a fixed height above
ground and that height is stated on the figure, because inventing an FFE per structure on a
figure about whether homes flood would be the worst possible thing to guess at.

    python -m coral.viz.fig_isometric_exposure --dem <dem.asc> --mxe <res.mxe> \\
        --buildings data/raw/sagis_pinpoint/fema_structures_pinpoint.geojson \\
        --center -81.0903 31.9522 --half-m 350 --out reports/figs/fig5_isometric.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc
from .pinpoint_style import PALETTE, extent_of, hillshade

ISO_X = np.cos(np.radians(30.0))
ISO_Y = np.sin(np.radians(30.0))


def project(x, y, z, *, zscale=1.0):
    """Axonometric projection. Parallel, so a metre is the same length anywhere on the page."""
    return (x - y) * ISO_X, (x + y) * ISO_Y + z * zscale


def _lonlat_to_m(lon, lat, lon0, lat0):
    return ((lon - lon0) * 111320.0 * np.cos(np.radians(lat0)),
            (lat - lat0) * 110540.0)


def _sample(grid, h, lon, lat):
    """Nearest-cell value of an .asc grid at lon/lat arrays."""
    col = ((lon - h["xllcorner"]) / h["cellsize"]).astype(int)
    row = (h["nrows"] - 1 - (lat - h["yllcorner"]) / h["cellsize"]).astype(int)
    ok = (col >= 0) & (col < h["ncols"]) & (row >= 0) & (row < h["nrows"])
    out = np.full(lon.shape, np.nan)
    out[ok] = grid[row[ok], col[ok]]
    return out


def build(dem_path, out, *, buildings, mxe=None, center=(-81.0903, 31.9522), half_m=500.0,
          ffe_m=0.45, zscale=2.5, sea_level=0.81, ground_step=2, sagis=None, roads=None,
          contours=(1.0, 2.0, 3.0, 4.0),
          marsh_classes=("E2EM", "E2SS", "E2FO", "E2US"),
          chm=None, nlcd=None, canopy_min_m=2.0, dpi=150, figsize=(13, 8.6)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    import geopandas as gpd

    dem, h = read_asc(dem_path)
    # Maximum DEPTH, not the .mxe surface. On dry cells .mxe holds the ground elevation rather
    # than nodata, so anything derived from it treats dry ground as water at ground level.
    dep = None
    if mxe and Path(mxe).exists():
        dep, _ = read_asc(mxe)
        dep = np.where(np.isfinite(dep) & (dep > -9990), dep, 0.0)
    wse = (dem + dep) if dep is not None else None

    # Marsh is drawn as its own class. Nearly all of it sits below the waterline, so without
    # this it would be painted as open water.
    marsh = None
    if sagis and Path(sagis, "sagis_wetlands_nwi.geojson").exists():
        from ..interventions.context_rasters import wetlands_mask
        # E2US (intertidal unconsolidated shore, i.e. mudflat) belongs here as well as the
        # vegetated classes. Around Pin Point the tidal flats are mapped E2US, so a filter of
        # vegetated classes alone leaves the whole intertidal zone drawn as open estuary.
        # E1UB stays out: that one really is subtidal water.
        marsh = wetlands_mask(Path(sagis) / "sagis_wetlands_nwi.geojson", dem_path,
                              cowardin_prefixes=tuple(marsh_classes))
        print(f"  NWI intertidal cells: {int(marsh.sum())} ({marsh.mean():.1%} of grid)")
    if nlcd and Path(nlcd).exists():
        # NLCD 90 woody wetlands, 95 emergent herbaceous wetlands. Wall-to-wall, so unlike the
        # NWI polygons it cannot leave the marsh unmapped over part of the domain.
        nl = read_asc(nlcd)[0]
        wet_nl = np.isin(np.round(nl), [90, 95])
        marsh = wet_nl if marsh is None else (marsh | wet_nl)
        print(f"  NLCD wetland cells: {int(wet_nl.sum())} ({wet_nl.mean():.1%} of grid)")

    if half_m <= 0:
        # Whole DEM. The window was only ever a crop, so the full clip works the same way;
        # it just costs more polygons, so raise --ground-step with it.
        e = extent_of(h)
        lon0, lat0 = (e[0] + e[1]) / 2, (e[2] + e[3]) / 2
        bbox = e
        half_m = max((e[1] - e[0]) * 111320.0 * np.cos(np.radians(lat0)),
                     (e[3] - e[2]) * 110540.0) / 2
    else:
        lon0, lat0 = center
        dlon = half_m / (111320.0 * np.cos(np.radians(lat0)))
        dlat = half_m / 110540.0
        bbox = (lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat)

    g = gpd.read_file(buildings).to_crs("EPSG:4326")
    g = g.cx[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    if g.empty:
        raise SystemExit("no structures in the window; move --center or raise --half-m")

    # Field names differ by inventory: FEMA USA Structures uses HEIGHT/POP_MEDIAN, Overture
    # uses height and carries no population. Detect rather than assume, so either can be drawn.
    hf = next((c for c in ("HEIGHT", "height") if c in g.columns), None)
    pf = next((c for c in ("POP_MEDIAN", "pop_median") if c in g.columns), None)
    med_h = float(np.nanmedian(g[hf])) if hf else 6.0
    if not np.isfinite(med_h):
        med_h = 6.0

    # --- ground mesh, coarse enough to draw as polygons ---
    cs = h["cellsize"]
    c0 = int((bbox[0] - h["xllcorner"]) / cs); c1 = int((bbox[1] - h["xllcorner"]) / cs)
    r1 = int(h["nrows"] - 1 - (bbox[2] - h["yllcorner"]) / cs)
    r0 = int(h["nrows"] - 1 - (bbox[3] - h["yllcorner"]) / cs)
    c0, c1 = max(c0, 0), min(c1, h["ncols"] - 1)
    r0, r1 = max(r0, 0), min(r1, h["nrows"] - 1)
    s = ground_step

    fig, ax = plt.subplots(figsize=figsize)

    # Block-mean the DEM once instead of slicing per cell. A flat colour per block reads as
    # pixels, so the ground is shaded: a terrain ramp for elevation, multiplied by a hillshade
    # of the blocked surface for relief.
    ny, nx = (r1 - r0) // s, (c1 - c0) // s
    sub = dem[r0:r0 + ny * s, c0:c0 + nx * s]
    zb = np.nanmean(sub.reshape(ny, s, nx, s), axis=(1, 3))
    db = np.zeros_like(zb)
    if dep is not None:
        d_ = dep[r0:r0 + ny * s, c0:c0 + nx * s]
        db = np.nanmean(d_.reshape(ny, s, nx, s), axis=(1, 3))
    marsh_b = None
    if marsh is not None:
        m_ = marsh[r0:r0 + ny * s, c0:c0 + nx * s].astype("float32")
        marsh_b = np.nanmean(m_.reshape(ny, s, nx, s), axis=(1, 3)) > 0.5

    chm_b = None
    if chm and Path(chm).exists():
        ch = read_asc(chm)[0]
        ch = np.where(np.isfinite(ch) & (ch > -9990), ch, 0.0)
        c_ = ch[r0:r0 + ny * s, c0:c0 + nx * s]
        chm_b = np.nanmean(c_.reshape(ny, s, nx, s), axis=(1, 3))

    filled = np.nan_to_num(zb, nan=float(np.nanmin(zb)))
    shade = hillshade(filled)
    shade = 0.55 + 0.45 * (shade - np.nanmin(shade)) / max(np.ptp(shade), 1e-9)
    lo, hi = np.nanpercentile(filled, [2, 98])
    # Muted greys for upland so the built environment is the light surface, and a green ramp
    # for marsh. The two must not share a ramp or the wetland and the neighbourhood merge.
    from matplotlib.colors import LinearSegmentedColormap
    # Upland is a warm neutral so the built-up ground reads as pale ground, and marsh is a
    # distinctly green ramp. Sharing a ramp is what made the wetland and the neighbourhood
    # look like the same surface.
    ramp = LinearSegmentedColormap.from_list("_up", ["#BFB6A4", "#F2EDE3"])
    marsh_ramp = LinearSegmentedColormap.from_list("_ma", ["#A8916A", "#E0D2B4"])
    canopy_ramp = LinearSegmentedColormap.from_list("_ca", ["#4A6B3A", "#87A868"])

    w = s * cs * 111320.0 * np.cos(np.radians(lat0))
    hh = s * cs * 110540.0
    quads, colors, depth = [], [], []
    cquads, ccolors, cdepth = [], [], []

    for i in range(ny):
        lat_a = h["yllcorner"] + (h["nrows"] - 1 - (r0 + i * s)) * cs
        for j in range(nx):
            zc = zb[i, j]
            if not np.isfinite(zc):
                continue
            lon_a = h["xllcorner"] + (c0 + j * s) * cs
            xa, ya = _lonlat_to_m(lon_a, lat_a, lon0, lat0)
            is_marsh = marsh_b is not None and marsh_b[i, j]
            # Marsh is tested FIRST. Almost all of it lies below the waterline, so any
            # ordering that checks open water first paints the whole marsh as estuary and
            # the wetland never appears at all.
            open_water = (zc <= sea_level) and not is_marsh
            flooded = (db[i, j] > 0.05) and not is_marsh
            # Water is drawn AT ITS OWN SURFACE, not on the bed. That is what turns the
            # shoreline into a visible edge instead of a change of colour.
            zdraw = max(sea_level, zc + db[i, j]) if (open_water or flooded) else zc
            if open_water:
                col = PALETTE["water"]
            elif flooded:
                t = min(db[i, j], 1.5) / 1.5
                col = plt.get_cmap("Blues")(0.40 + 0.50 * t)
            elif is_marsh:
                # Wetter marsh reads darker, so the creek network stays visible.
                t = float(np.clip((zc - (-1.2)) / 1.6, 0, 1))
                col = marsh_ramp(t)
            else:
                col = ramp(float(np.clip((zc - lo) / max(hi - lo, 1e-9), 0, 1)))
            if not open_water:
                k = shade[i, j]
                col = (col[0] * k, col[1] * k, col[2] * k, 1.0)
            corners = [(xa, ya), (xa + w, ya), (xa + w, ya - hh), (xa, ya - hh)]
            quads.append([project(cx, cy, zdraw, zscale=zscale) for cx, cy in corners])
            colors.append(col); depth.append(xa + ya + zdraw * 0.001)

            # Canopy as a second surface lifted to tree height. The CHM measures vegetation,
            # which is what it is good for -- it is useless for buildings, which it reads as
            # the trees overhanging them.
            if chm_b is not None and chm_b[i, j] > canopy_min_m and not open_water:
                ct = float(np.clip(chm_b[i, j] / 20.0, 0, 1))
                cc = canopy_ramp(0.25 + 0.6 * ct)
                k = shade[i, j]
                zt = zc + chm_b[i, j]
                cquads.append([project(cx, cy, zt, zscale=zscale) for cx, cy in corners])
                ccolors.append((cc[0] * k, cc[1] * k, cc[2] * k, 1.0))
                cdepth.append(xa + ya)

    # Canopy goes in its own collection above the ground. Sharing one collection loses it:
    # a tree is lifted about twenty cells' worth of screen distance, and every one of those
    # nearer ground cells sorts later under a single painter's ordering and repaints over it.
    if cquads:
        co = np.argsort(cdepth)
        ax.add_collection(PolyCollection([cquads[i] for i in co],
                                         facecolors=[ccolors[i] for i in co],
                                         edgecolors=[ccolors[i] for i in co],
                                         linewidths=0.35, zorder=2, rasterized=True))
        print(f"  canopy blocks: {len(cquads)} ({len(cquads) / max(len(quads), 1):.0%} "
              f"of ground)")

    order = np.argsort(depth)
    # Edge colour matches the face. With "none" the antialiased quad borders leave white
    # seams across the whole ground plane.
    ax.add_collection(PolyCollection([quads[i] for i in order],
                                     facecolors=[colors[i] for i in order],
                                     edgecolors=[colors[i] for i in order],
                                     linewidths=0.35, zorder=1, rasterized=True,
                                     antialiaseds=True))

    def ground_z(lon, lat):
        """Drawing height for a draped line: terrain, or the water surface where wet."""
        zg_ = _sample(dem, h, lon, lat)
        dg_ = _sample(dep, h, lon, lat) if dep is not None else np.zeros_like(zg_)
        dg_ = np.nan_to_num(dg_, nan=0.0)
        zg_ = np.nan_to_num(zg_, nan=sea_level)
        return np.maximum(zg_ + dg_, np.where(zg_ <= sea_level, sea_level, zg_))

    # --- coastline and contours ---------------------------------------------------------
    # An explicit shoreline is what makes the drawing read as a real place rather than a
    # raster. It is the sea-level contour of the same DEM the model ran on.
    _iso_lines(ax, _grid_contours(np.nan_to_num(zb, nan=99.0), sea_level, h, r0, c0, s),
               lambda lo_, la_: np.full(np.shape(lo_), sea_level), lon0, lat0, zscale,
               color="#2B5A72", lw=1.0, zorder=5, solid_capstyle="round")
    for lev in contours or ():
        _iso_lines(ax, _grid_contours(np.nan_to_num(zb, nan=-99.0), lev, h, r0, c0, s),
                   lambda lo_, la_, L=lev: np.full(np.shape(lo_), L), lon0, lat0, zscale,
                   color="#8A8378", lw=0.45, alpha=0.75, zorder=6)

    # --- roads ---------------------------------------------------------------------------
    if roads and Path(roads).exists():
        from shapely.geometry import box as _box
        rd = gpd.read_file(roads).to_crs("EPSG:4326")
        # Clip the GEOMETRY, not just select intersecting rows. Selecting alone keeps whole
        # segments that run far outside the window and wrecks the drawing extent.
        rd = gpd.clip(rd, _box(bbox[0], bbox[2], bbox[1], bbox[3]))
        rd = rd[~rd.geometry.is_empty & rd.geometry.notna()]
        widths = {"primary": 1.9, "secondary": 1.5, "tertiary": 1.2,
                  "residential": 0.9, "service": 0.5, "path": 0.35, "footway": 0.35}
        for _, rr in rd.iterrows():
            gm = rr.geometry
            gms = gm.geoms if gm.geom_type == "MultiLineString" else [gm]
            lw_ = widths.get(str(rr.get("class")), 0.5)
            for part in gms:
                lo_, la_ = np.array(part.coords.xy[0]), np.array(part.coords.xy[1])
                _iso_lines(ax, [(lo_, la_)], ground_z, lon0, lat0, zscale,
                           color="#6E675C", lw=lw_, zorder=7, solid_capstyle="round")
        print(f"  roads drawn: {len(rd)}")

    # --- drainage, draped on the surface ---------------------------------------------------
    if sagis:
        S = Path(sagis)
        from shapely.geometry import box as _box2
        clipbox = _box2(bbox[0], bbox[2], bbox[1], bbox[3])
        for fn, colr, lwr in (("sagis_pipes_chatham.geojson", "#5B6B7A", 0.5),
                              ("sagis_ditches_chatham.geojson", "#3F5668", 0.7),
                              ("sagis_canals_chatham.geojson", "#3F5668", 1.1)):
            fp = S / fn
            if not fp.exists():
                continue
            gl = gpd.read_file(fp).to_crs("EPSG:4326")
            gl = gpd.clip(gl, clipbox)
            gl = gl[~gl.geometry.is_empty & gl.geometry.notna()]
            for gm in gl.geometry:
                for part in (gm.geoms if gm.geom_type.startswith("Multi") else [gm]):
                    if part.geom_type != "LineString":
                        continue
                    lo_, la_ = np.array(part.coords.xy[0]), np.array(part.coords.xy[1])
                    _iso_lines(ax, [(lo_, la_)], ground_z, lon0, lat0, zscale,
                               color=colr, lw=lwr, zorder=7, alpha=0.9)
        fo = S / "sagis_outfalls_chatham.geojson"
        if fo.exists():
            go = gpd.read_file(fo).to_crs("EPSG:4326")
            go = go.cx[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            if len(go):
                lo_ = np.array([g.x for g in go.geometry])
                la_ = np.array([g.y for g in go.geometry])
                xx, yy = _lonlat_to_m(lo_, la_, lon0, lat0)
                zz = ground_z(lo_, la_)
                pts = [project(a, b, c, zscale=zscale) for a, b, c in zip(xx, yy, zz)]
                ax.scatter([q[0] for q in pts], [q[1] for q in pts], s=14,
                           c=PALETTE["outfall"], edgecolors="white", linewidths=0.4,
                           zorder=10)
                print(f"  outfalls drawn: {len(go)}")

    # --- structures ---
    faces, fcol, fdepth, edges = [], [], [], []
    n_flood = n_meas = 0
    pop_at_risk = 0.0
    depths, pops = [], []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)
        lon = np.array(geom.exterior.coords.xy[0])
        lat = np.array(geom.exterior.coords.xy[1])
        zg = np.nanmedian(_sample(dem, h, lon, lat))
        if not np.isfinite(zg):
            continue
        bh = row.get(hf, np.nan) if hf else np.nan
        measured = np.isfinite(bh) and bh > 0
        n_meas += int(measured)
        bh = float(bh) if measured else med_h

        d_here = np.nanmedian(_sample(dep, h, lon, lat)) if dep is not None else np.nan
        d_here = 0.0 if not np.isfinite(d_here) else float(d_here)
        depths.append(d_here)
        pv_ = row.get(pf) if pf else None
        pops.append(float(pv_) if pv_ is not None and np.isfinite(pv_) else 0.0)
        flooded = d_here > ffe_m
        n_flood += int(flooded)
        if flooded:
            pop_at_risk += pops[-1]

        x, y = _lonlat_to_m(lon, lat, lon0, lat0)
        key = float(np.mean(x) + np.mean(y))
        roof = PALETTE["intervention"] if flooded else "#E8E4DC"
        wall = "#8E3B2C" if flooded else "#BDB6AA"

        for k in range(len(x) - 1):
            quad = [project(x[k], y[k], zg, zscale=zscale),
                    project(x[k + 1], y[k + 1], zg, zscale=zscale),
                    project(x[k + 1], y[k + 1], zg + bh, zscale=zscale),
                    project(x[k], y[k], zg + bh, zscale=zscale)]
            faces.append(quad); fcol.append(wall); fdepth.append(key)
        top = [project(px, py, zg + bh, zscale=zscale) for px, py in zip(x, y)]
        faces.append(top); fcol.append(roof); fdepth.append(key + 0.01)
        edges.append((top, measured))

    order = np.argsort(fdepth)
    ax.add_collection(PolyCollection([faces[i] for i in order],
                                     facecolors=[fcol[i] for i in order],
                                     edgecolors="#6B655C", linewidths=0.3, zorder=8))
    # Roofs of structures with no measured height get a dashed outline, so an assumed
    # extrusion is never mistaken for a measured one.
    for top, measured in edges:
        if not measured:
            xs, ys = zip(*top)
            ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color="#4A453E", lw=0.6,
                    ls=(0, (2, 1.6)), zorder=9)

    # The first-floor height is assumed, so show the count across a range of assumptions
    # rather than asserting one number.
    dv = np.array(depths)
    pv = np.array(pops)
    ffes = np.array([0.15, 0.30, 0.45, 0.60, 0.90])
    counts = [(dv > f).sum() for f in ffes]
    axi = fig.add_axes([0.09, 0.10, 0.17, 0.16])
    axi.bar(range(len(ffes)), counts, color=PALETTE["intervention"], alpha=0.85, width=0.65)
    axi.axvline(float(np.argmin(np.abs(ffes - ffe_m))), color=PALETTE["text"], lw=0.8, ls=":")
    axi.set_xticks(range(len(ffes)))
    axi.set_xticklabels([f"{f:.2f}" for f in ffes], fontsize=6.5)
    axi.set_xlabel("assumed first floor above ground (m)", fontsize=6.8)
    axi.set_ylabel("structures", fontsize=6.8)
    axi.tick_params(labelsize=6.5)
    axi.set_title("sensitivity to the assumption", fontsize=7.4, color=PALETTE["text"])
    for sp in ("top", "right"):
        axi.spines[sp].set_visible(False)

    _add_scalebar(ax, half_m, zscale)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")

    total = len(g)
    ax.set_title("Freeboard at house scale — modelled peak water surface against assumed "
                 "first floor", fontsize=13.5, color=PALETTE["text"], pad=14)
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=9, color=PALETTE["intervention"],
               label="water above assumed first floor"),
        Line2D([], [], marker="s", ls="", ms=9, color="#B9B2A6",
               label="water below assumed first floor"),
        Line2D([], [], color="#4A453E", ls=(0, (2, 1.6)), lw=0.9,
               label="height assumed, not measured"),
        Line2D([], [], marker="s", ls="", ms=9, color=PALETTE["water"], label="open water"),
        Line2D([], [], marker="s", ls="", ms=9, color="#D3C3A2", label="tidal marsh and flat"),
        Line2D([], [], marker="s", ls="", ms=9, color="#5E7F49", label="tree canopy (lidar CHM)"),
        Line2D([], [], color="#3F5668", lw=1.2, label="ditches, canals and storm drains"),
        Line2D([], [], marker="o", ls="", ms=5, color=PALETTE["outfall"], label="outfalls"),
        Line2D([], [], marker="s", ls="", ms=9, color="#2C6FA6",
               label="modelled flood, deeper = darker"),
    ], loc="upper left", fontsize=8.4, frameon=False)

    caption = (f"{total} structures in view   ·   {n_meas} with measured height   ·   "
               f"{n_flood} above-floor")
    if pf:
        caption += f"   ·   about {pop_at_risk:.0f} residents"
    caption += (f"\nfirst floor assumed {ffe_m:.2f} m above ground; "
                f"vertical exaggeration {zscale:.0f}x   ·   footprints: "
                f"{Path(buildings).stem}")
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8.4,
             color=PALETTE["muted"])
    _profile_inset(fig, zb, db, sea_level, s * cs * 110540.0)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({total} structures, height field {hf}, {n_meas} measured, "
          f"{n_flood} above floor, pop {pop_at_risk:.0f})")


def _iso_lines(ax, segs, zf, lon0, lat0, zscale, **kw):
    """Draw lon/lat line segments in the axonometric view.

    `zf` returns the drawing height for a point, so a line can be draped on the terrain or
    held flat at a water level.
    """
    for lon, lat in segs:
        if len(lon) < 2:
            continue
        x, y = _lonlat_to_m(np.asarray(lon), np.asarray(lat), lon0, lat0)
        z = zf(np.asarray(lon), np.asarray(lat))
        pts = [project(xx, yy, zz, zscale=zscale) for xx, yy, zz in zip(x, y, z)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)


def _grid_contours(zb, level, h, r0, c0, s):
    """Contour of the blocked grid at `level`, returned as lon/lat segment pairs."""
    import matplotlib.pyplot as plt
    fig_ = plt.figure()
    try:
        cs_ = plt.contour(zb, levels=[level])
        segs = []
        for coll in cs_.allsegs[0]:
            j, i = coll[:, 0], coll[:, 1]
            lon = h["xllcorner"] + (c0 + j * s) * h["cellsize"]
            lat = h["yllcorner"] + (h["nrows"] - 1 - (r0 + i * s)) * h["cellsize"]
            segs.append((lon, lat))
        return segs
    finally:
        plt.close(fig_)


def _add_scalebar(ax, half_m, zscale, *, metres=200.0):
    """Horizontal scale bar on the ground plane. Parallel projection keeps it true."""
    gx, gy = -half_m * 1.15, -half_m * 0.15
    p1 = project(gx, gy, 0.0, zscale=zscale)
    p2 = project(gx + metres, gy, 0.0, zscale=zscale)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=PALETTE["text"], lw=2.4,
            solid_capstyle="butt", zorder=25)
    ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 - 18, f"{metres:.0f} m", ha="center",
            va="top", fontsize=7.5, color=PALETTE["text"], zorder=25)


def _profile_inset(fig, zb, db, sea_level, res_m, *, rect=(0.66, 0.07, 0.30, 0.16)):
    """Ground, still water and peak water along the row with the most flooding.

    An elevation ruler cannot work on this scene: the domain is about a kilometre wide and six
    metres tall, so the vertical axis is a fraction of a percent of the drawing. A profile has
    its own vertical scale and carries the same information legibly.
    """
    import numpy as np
    wet = db > 0.05
    if not wet.any():
        return
    r = int(np.argmax(wet.sum(axis=1)))
    g = zb[r]
    x = np.arange(g.size) * res_m
    ok = np.isfinite(g)
    if ok.sum() < 5:
        return
    ax = fig.add_axes(rect)
    ax.fill_between(x[ok], np.nanmin(g[ok]) - 1, g[ok], color="#D9CBA8", zorder=2)
    ax.plot(x[ok], g[ok], color=PALETTE["terrain"], lw=1.0, zorder=3, label="ground")
    ax.axhline(sea_level, color=PALETTE["water"], lw=1.4, zorder=4, label="still water")
    peak = np.where(db[r] > 0.05, g + db[r], np.nan)
    ax.plot(x[ok], peak[ok], color="#2C6FA6", lw=1.4, zorder=5, label="peak water")
    ax.set_xlabel("distance along profile (m)", fontsize=6.8)
    ax.set_ylabel("m NAVD88", fontsize=6.8)
    ax.tick_params(labelsize=6.2)
    ax.legend(fontsize=6.2, frameon=False, loc="upper right", ncol=3,
              handlelength=1.2, columnspacing=0.9)
    ax.set_title("profile across the flooded row", fontsize=7.4, color=PALETTE["text"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--mxe", default=None, help="maximum water surface elevation grid")
    ap.add_argument("--buildings", default="data/raw/sagis_pinpoint/"
                                           "fema_structures_pinpoint.geojson")
    ap.add_argument("--center", nargs=2, type=float, default=[-81.0903, 31.9522])
    ap.add_argument("--half-m", type=float, default=500.0)
    ap.add_argument("--ffe-m", type=float, default=0.45)
    ap.add_argument("--zscale", type=float, default=2.5)
    ap.add_argument("--ground-step", type=int, default=2,
                    help="DEM cells per drawn ground block; lower is finer")
    ap.add_argument("--sagis", default="data/raw/sagis_pinpoint")
    ap.add_argument("--roads", default="data/raw/overture_roads_pinpoint.geojson")
    ap.add_argument("--chm", default=None, help="canopy height grid; drawn as tree canopy")
    ap.add_argument("--nlcd", default=None, help="NLCD on the DEM grid; classes 90/95 = marsh")
    ap.add_argument("--canopy-min-m", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--figsize", nargs=2, type=float, default=[13.0, 8.6])
    ap.add_argument("--marsh-classes", nargs="+",
                    default=["E2EM", "E2SS", "E2FO", "E2US"],
                    help="NWI Cowardin prefixes drawn as intertidal")
    ap.add_argument("--contours", nargs="*", type=float,
                    default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--out", default="reports/figs/fig5_isometric.png")
    a = ap.parse_args()
    build(a.dem, a.out, buildings=a.buildings, mxe=a.mxe, center=tuple(a.center),
          half_m=a.half_m, ffe_m=a.ffe_m, zscale=a.zscale,
          ground_step=a.ground_step, sagis=a.sagis, roads=a.roads,
          contours=tuple(a.contours), marsh_classes=tuple(a.marsh_classes),
          chm=a.chm, nlcd=a.nlcd, canopy_min_m=a.canopy_min_m,
          dpi=a.dpi, figsize=tuple(a.figsize))


if __name__ == "__main__":
    main()
