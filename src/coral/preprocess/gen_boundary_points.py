#!/usr/bin/env python3
"""
Generate a seaward-front boundary for the GeoClaw -> LISFLOOD coupling.

On a barrier-island / marsh coast (like Savannah), a simple shoreline contour
is fragmented and winds through interior marsh channels, which is not useful for
surge forcing. Instead, for each latitude band we scan from the east (open ocean)
westward and place a gauge at the outermost ocean-to-land transition, nudged a few
cells seaward into well-resolved water. This yields a clean N-S seaward front:
surge is imposed along the outer coast and LISFLOOD propagates it inland.

Writes three identically-ordered outputs (so bcN matches gauge N after sorting):
  geoclaw_gauges.txt   - paste into setrun.py
  <name>.bci           - LISFLOOD point sources (P lon lat HVAR bcN)
  boundary_points.csv  - QC

Deps: numpy, scipy, rasterio, matplotlib
Usage:
  python gen_boundary_points.py --dem SUB_DEM_SAV.asc --level 0.81 \
      --spacing-m 400 --seaward-cells 3 --name savannah_coastline
"""
import argparse
import os
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="scenario YAML; fills level/spacing/name/ref point")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--level", type=float, default=None,
                    help="water level defining ocean (m); default from config sea_level")
    ap.add_argument("--spacing-m", type=float, default=None)
    ap.add_argument("--seaward-cells", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--lat", type=float, default=None, help="ref point lat (QC)")
    ap.add_argument("--lon", type=float, default=None, help="ref point lon (QC)")
    ap.add_argument("--qc", default="qc/boundary_points_QC.png")
    ap.add_argument("--color-center", type=float, default=0.0,
                    help="elevation (m) for the blue/brown split in the QC plot")
    # dense along-coastline observation gauges (Phase 5): march the shoreline at
    # ~obs_spacing_m arclength. Separate ID range, not wired into the .bci/.bdy coupling.
    ap.add_argument("--obs-spacing-m", type=float, default=None,
                    help="if set, also emit dense observation gauges every N m along the coast")
    ap.add_argument("--obs-seaward-cells", type=int, default=2,
                    help="snap each obs gauge to the nearest ocean cell within this radius")
    ap.add_argument("--trend-window-m", type=float, default=6000.0,
                    help="alongshore window for the shoreline trend used to reject inlet "
                         "excursions")
    ap.add_argument("--trend-tol-m", type=float, default=800.0,
                    help="how far west of the trend a row may sit before it is pulled back")
    ap.add_argument("--coast-min-len-m", type=float, default=500.0,
                    help="drop shoreline fragments shorter than this")
    args = ap.parse_args()

    # fill unset args from the scenario config (CLI args still override)
    cfg = None
    if args.config:
        from coral import config
        c = cfg = config.load(args.config)
        if args.level is None:         args.level = c.manning.sea_level
        if args.spacing_m is None:     args.spacing_m = c.coupling.gauge_spacing_m
        if args.seaward_cells is None: args.seaward_cells = c.coupling.seaward_cells
        if args.name is None:          args.name = f"{c.name}_coastline"
        if args.lon is None:           args.lon = c.domain.ref_point[0]
        if args.lat is None:           args.lat = c.domain.ref_point[1]
    # standalone fallbacks
    if args.level is None:         args.level = 0.0
    if args.spacing_m is None:     args.spacing_m = 400.0
    if args.seaward_cells is None: args.seaward_cells = 3
    if args.name is None:          args.name = "savannah_coastline"
    if args.lon is None:           args.lon = -81.0903
    if args.lat is None:           args.lat = 31.9522

    with rasterio.open(args.dem) as s:
        z = s.read(1).astype(float)
        tr = s.transform
        nod = s.nodata if s.nodata is not None else -9999
        b = s.bounds
    z = np.where((z == nod) | (z <= -9990), np.nan, z)
    ny, nx = z.shape
    midlat = (b.bottom + b.top) / 2
    cell_m = abs(tr.a) * 111000 * np.cos(np.radians(midlat))
    step = max(1, int(round(args.spacing_m / cell_m)))

    # largest connected ocean body = open ocean
    ocean = (z < args.level) & np.isfinite(z)
    lbl, n = ndimage.label(ocean)
    if n == 0:
        raise SystemExit("no ocean cells below --level; check the level/DEM")
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    openocean = (lbl == (1 + int(np.argmax(sizes))))

    # per latitude band: outermost ocean->land transition, nudged seaward (east)
    #
    # The walk west stops at the first non-ocean cell. At an inlet there is none: the open
    # ocean runs continuously through the mouth into the sound, so the walk carries on until
    # it hits land far inland. On this coast that happens at the Savannah River mouth, Wassaw
    # Sound and Ossabaw Sound, and the gauge lands kilometres inside the estuary.
    #
    # A seaward front crosses an inlet mouth rather than entering it, and the rows either side
    # already say where the coast is. So the shoreline column is found for every row first,
    # then rows that sit far west of their neighbours are pulled back to the local trend.
    shore_cols = np.full(ny, -1, dtype=int)
    for r in range(ny):
        if not openocean[r].any():
            continue
        c = int(np.where(openocean[r])[0].max())
        while c > 0 and openocean[r, c]:
            c -= 1
        shore_cols[r] = c + 1

    valid = shore_cols >= 0
    trend = np.array(shore_cols, dtype=float)
    trend[~valid] = np.nan
    if valid.any():
        win = max(3, int(round(args.trend_window_m / cell_m)) | 1)
        filled = np.where(valid, shore_cols, np.nan)
        idx = np.arange(ny)
        filled = np.interp(idx, idx[valid], np.asarray(shore_cols)[valid])
        # Maximum, not median. The front is the most seaward shoreline in the neighbourhood,
        # and every inlet excursion is westward, so a maximum ignores them however wide the
        # inlet is. A median still sits inside a sound that spans more than half the window,
        # which is what happens at the Savannah River mouth.
        trend = ndimage.maximum_filter(filled, size=win, mode="nearest")
    tol = max(1, int(round(args.trend_tol_m / cell_m)))

    pts, pulled = [], 0
    for r in range(0, ny, step):
        if shore_cols[r] < 0:
            continue
        shore_c = shore_cols[r]
        floor_c = int(round(trend[r])) - tol          # west of this is an inlet excursion
        if shore_c < floor_c:
            shore_c = floor_c
            pulled += 1
        g = min(nx - 1, shore_c + args.seaward_cells)
        if 0 <= g < nx and openocean[r, g]:
            lon, lat = xy(tr, r, g)
            pts.append((lon, lat))
    if pulled:
        print(f"{pulled} rows pulled back to the alongshore trend (inlet mouths)")
    if not pts:
        raise SystemExit("no seaward-front points found")
    print(f"{len(pts)} seaward-front points, "
          f"lat {min(p[1] for p in pts):.3f}-{max(p[1] for p in pts):.3f}")

    with open("geoclaw_gauges.txt", "w") as f:
        f.write(f"    # --- {args.name} seaward-front gauges (auto) ---\n")
        for i, (lon, lat) in enumerate(pts, 1):
            f.write(f"    rundata.gaugedata.gauges.append("
                    f"[{i}, {lon:.6f}, {lat:.6f}, "
                    f"rundata.clawdata.t0, rundata.clawdata.tfinal])\n")
    with open(f"{args.name}.bci", "w") as f:
        for i, (lon, lat) in enumerate(pts, 1):
            f.write(f"P\t{lon:.7f}\t{lat:.7f}\tHVAR\tbc{i}\n")
    with open("boundary_points.csv", "w") as f:
        f.write("gauge_id,bc_tag,lon,lat\n")
        for i, (lon, lat) in enumerate(pts, 1):
            f.write(f"{i},bc{i},{lon:.7f},{lat:.7f}\n")

    # dense along-coastline observation gauges (Phase 5), separate ID range, not coupled
    obs_pts = []
    if args.obs_spacing_m:
        cell_m_y = abs(tr.e) * 111000
        obs_pts = along_coast_gauges(z, openocean, args.level, tr, cell_m, cell_m_y,
                                     spacing_m=args.obs_spacing_m,
                                     seaward_cells=args.obs_seaward_cells,
                                     min_len_m=args.coast_min_len_m)
        base_id = len(pts)
        with open("geoclaw_gauges.txt", "a") as f:
            f.write(f"    # --- {args.name} dense coastline OBSERVATION gauges "
                    f"(auto, {args.obs_spacing_m:g} m; NOT coupled to LISFLOOD) ---\n")
            for j, (lon, lat) in enumerate(obs_pts, 1):
                f.write(f"    rundata.gaugedata.gauges.append("
                        f"[{base_id + j}, {lon:.6f}, {lat:.6f}, "
                        f"rundata.clawdata.t0, rundata.clawdata.tfinal])\n")
        with open("obs_gauges.csv", "w") as f:
            f.write("gauge_id,lon,lat\n")
            for j, (lon, lat) in enumerate(obs_pts, 1):
                f.write(f"{base_id + j},{lon:.7f},{lat:.7f}\n")
        print(f"{len(obs_pts)} dense coastline observation gauges "
              f"(ids {base_id + 1}-{base_id + len(obs_pts)}) @ {args.obs_spacing_m:g} m")
        if obs_pts:
            ol = [p[0] for p in obs_pts]; oa = [p[1] for p in obs_pts]
            print(">>> level-7 flagregion bbox (dense coastal gauges) for setrun.py:")
            print(f"    x1,x2 = {min(ol) - 0.005:.3f}, {max(ol) + 0.005:.3f}")
            print(f"    y1,y2 = {min(oa) - 0.005:.3f}, {max(oa) + 0.005:.3f}")

    # The box covers the gauges and nothing else. GeoClaw output is read at the coupling
    # gauges and nowhere else -- the 4 m nest reads the 30 m LISFLOOD result, not GeoClaw --
    # so refining inland spends cells resolving an estuary LISFLOOD already resolves. The
    # margin is what keeps the gauges off the edge of a refinement change.
    lons = [p[0] for p in pts]; lats = [p[1] for p in pts]
    fr = [min(lons) - 0.02, max(lons) + 0.02, min(lats) - 0.02, max(lats) + 0.02]
    lvl = cfg.geoclaw.amr_max if cfg is not None else 6
    print(f">>> level-{lvl} refine box (gauges + nested domain) for the scenario:")
    print(f"    refine_box: [{fr[0]:.3f}, {fr[1]:.3f}, {fr[2]:.3f}, {fr[3]:.3f}]")

    os.makedirs(os.path.dirname(args.qc) or ".", exist_ok=True)
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(9, 7))
    # continuous DEM: blue-green water -> brown land, diverging at sea level,
    # clipped to the coastal range so marsh/channels keep contrast
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    norm = TwoSlopeNorm(vmin=max(zmin, -6), vcenter=args.color_center,
                        vmax=min(zmax, 8))
    im = ax.imshow(z, extent=[b.left, b.right, b.bottom, b.top], origin="upper",
                   cmap="BrBG_r", norm=norm)
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label=f"elevation (m); center = {args.color_center:g} m")
    ax.plot(lons, lats, "-o", c="lime", ms=4, lw=.8, mec="k", mew=.2,
            label=f"{len(pts)} seaward-front gauges")
    if obs_pts:
        ax.scatter([p[0] for p in obs_pts], [p[1] for p in obs_pts], s=6, c="magenta",
                   ec="none", zorder=4,
                   label=f"{len(obs_pts)} obs gauges ({args.obs_spacing_m:g} m)")
    ax.scatter([args.lon], [args.lat], s=160, marker="*", c="yellow", ec="k",
               zorder=5, label="ref point")
    ax.add_patch(mpatches.Rectangle((fr[0], fr[2]), fr[1] - fr[0], fr[3] - fr[2],
                 fill=False, ec="black", lw=1.5, ls="--", label=f"L{lvl} refine box"))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("Seaward-front gauges over DEM (blue=water, brown=land)")
    fig.savefig(args.qc, dpi=130, bbox_inches="tight")
    print(f"wrote geoclaw_gauges.txt, {args.name}.bci, boundary_points.csv, {args.qc}")


def along_coast_gauges(z, openocean, level, tr, cell_m_x, cell_m_y, *,
                       spacing_m, seaward_cells=2, min_len_m=500.0):
    """Dense observation gauges marched along the shoreline at ~`spacing_m` arclength.

    Extracts the `level` iso-contour of the DEM (the shoreline, incl. winding marsh edges),
    keeps segments that border the open ocean, walks each by cumulative arclength dropping a
    point every `spacing_m`, and snaps each to the nearest open-ocean cell within
    `seaward_cells` so GeoClaw samples a wet cell. Returns a de-duplicated [(lon,lat), ...].
    """
    from rasterio.transform import xy
    try:                                   # subpixel contours; fall back to matplotlib
        from skimage.measure import find_contours
        contours = find_contours(np.where(np.isfinite(z), z, level - 1), level)
    except Exception:
        import matplotlib.pyplot as _plt
        cs = _plt.contour(np.where(np.isfinite(z), z, level - 1), levels=[level])
        contours = [seg[:, ::-1] for seg in cs.allsegs[0]]     # (x,y)->(row,col)
        _plt.close("all")

    ny, nx = z.shape
    def is_ocean(r, c):
        r, c = int(round(r)), int(round(c))
        return 0 <= r < ny and 0 <= c < nx and openocean[r, c]

    def snap(r, c):                        # nearest open-ocean cell within radius
        for rad in range(seaward_cells + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if is_ocean(r + dr, c + dc):
                        return int(round(r + dr)), int(round(c + dc))
        return None

    pts, seen = [], set()
    for ct in contours:
        # keep only shoreline that fronts open ocean (skip interior/inland contours)
        border = sum(is_ocean(r, c) for r, c in ct[:: max(1, len(ct) // 20)])
        if border == 0:
            continue
        seg_m = np.hypot(np.diff(ct[:, 0]) * cell_m_y, np.diff(ct[:, 1]) * cell_m_x)
        L = seg_m.sum()
        if L < min_len_m:
            continue
        cum = np.concatenate([[0], np.cumsum(seg_m)])
        d = 0.0
        while d <= L:
            k = int(np.searchsorted(cum, d) - 1); k = max(0, min(k, len(ct) - 2))
            f = (d - cum[k]) / (seg_m[k] + 1e-9)
            r = ct[k, 0] + f * (ct[k + 1, 0] - ct[k, 0])
            c = ct[k, 1] + f * (ct[k + 1, 1] - ct[k, 1])
            snapped = snap(r, c)
            if snapped and snapped not in seen:
                seen.add(snapped)
                lon, lat = xy(tr, snapped[0], snapped[1])
                pts.append((lon, lat))
            d += spacing_m
    return pts


if __name__ == "__main__":
    main()
