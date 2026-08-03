"""Nest a high-res LISFLOOD clip inside a coarse run: build the clip's boundary
forcing from the coarse run's time-varying output.

The building-scale (2 m) Pin Point clip does NOT get its own GeoClaw run. Instead it
is one-way forced by the coarse (30 m) compound run: we sample the coarse water-surface
elevation WSE(t) = DEM + depth along the clip's perimeter and write the clip's .bci
(HVAR boundary points) + .bdy (WSE time series). Standard practice for building-scale
LISFLOOD sub-domains (see docs/high-res notes).

fetch: read the coarse DEM + the ordered .wd depth snapshots (saveint apart, from t0);
sample along the clip bbox edges at `spacing_m`; keep points that are wet in the coarse
model (water actually enters there). Times are on the model clock, matching the .par.

Deps: numpy (+ the emulator read_asc helper).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..emulator.dataset import read_asc


def _coarse_wse_series(coarse_dir, root, dry_thresh=0.05):
    """Return (dem, x, y, times, wse_stack) for the coarse run. wse = dem+depth where
    wet, nan where dry; wse_stack is [T, ny, nx]."""
    d = Path(coarse_dir)
    dem, h = read_asc(next(d.glob("SUB_DEM*.asc")))
    ny, nx = dem.shape; cs = h["cellsize"]
    x = h["xllcorner"] + (np.arange(nx) + .5) * cs
    y = (h["yllcorner"] + ny * cs) - (np.arange(ny) + .5) * cs
    res = d / "results_matthew_sav" if (d / "results_matthew_sav").exists() else d
    wds = sorted(res.glob(f"{root}-*.wd"))
    stack = []
    for p in wds:
        depth, _ = read_asc(p)
        wse = np.where((depth > dry_thresh) & np.isfinite(dem), dem + depth, np.nan)
        stack.append(wse)
    return dem, x, y, np.array(stack)


def nest_bdy(coarse_dir, clip_bbox, out_bci, out_bdy, *, root="res_matthew_sav",
             saveint=1800.0, t0=86400.0, spacing_m=30.0, dry_thresh=0.05,
             wet_frac=0.25, inset_deg=1e-5):
    """Build the clip's .bci + .bdy from the coarse run. clip_bbox=[W,E,S,N].
    Boundary points are placed along the clip perimeter every ~spacing_m and kept only
    where the coarse cell is wet for >= wet_frac of the timesteps (water enters there).

    Points are inset from the exact bbox edge by `inset_deg` (about half a clip cell) so
    they land on the outermost valid clip cells. A point on the exact perimeter maps to a
    cell index of -1 or n in the high-res clip and makes LISFLOOD segfault."""
    dem, x, y, wse = _coarse_wse_series(coarse_dir, root, dry_thresh)
    T = wse.shape[0]
    W, E, S, N = clip_bbox
    Wi, Ei, Si, Ni = W + inset_deg, E - inset_deg, S + inset_deg, N - inset_deg
    step_deg = spacing_m / 111000.0
    # perimeter sample points (lon, lat), inset onto the outermost valid cells
    xs = np.arange(Wi, Ei, step_deg); ys = np.arange(Si, Ni, step_deg)
    pts = ([(xx, Si) for xx in xs] + [(xx, Ni) for xx in xs] +
           [(Wi, yy) for yy in ys] + [(Ei, yy) for yy in ys])

    kept, blocks = [], []
    for lon, lat in pts:
        i = int(np.argmin(np.abs(x - lon))); j = int(np.argmin(np.abs(y - lat)))
        series = wse[:, j, i]
        if np.isfinite(series).mean() < wet_frac:      # rarely wet -> not a water boundary
            continue
        # fill dry gaps with the coarse ground (DEM) so the series is complete
        filled = np.where(np.isfinite(series), series, dem[j, i])
        kept.append((lon, lat)); blocks.append(filled)

    if not kept:
        raise SystemExit("no wet clip-boundary points found in the coarse run")
    times = t0 + np.arange(T) * saveint
    with open(out_bci, "w") as f:
        for k, (lon, lat) in enumerate(kept, 1):
            f.write(f"P\t{lon:.7f}\t{lat:.7f}\tHVAR\tbc{k}\n")
    with open(out_bdy, "w") as f:
        f.write("comment\n")
        for k, ser in enumerate(blocks, 1):
            f.write(f"bc{k}\n{T}\t\tseconds\n")
            for v, tt in zip(ser, times):
                f.write(f"{v:.4f}\t{tt:.1f}\t\n")
    print(f"nested boundary: {len(kept)} wet points, {T} timesteps "
          f"({times[0]:.0f}-{times[-1]:.0f} s) -> {out_bci}, {out_bdy}")
    return {"points": len(kept), "timesteps": T, "bci": out_bci, "bdy": out_bdy}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Nest a high-res clip boundary from a coarse run")
    ap.add_argument("--coarse", required=True, help="coarse run dir (DEM + results)")
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "E", "S", "N"))
    ap.add_argument("--out-bci", required=True); ap.add_argument("--out-bdy", required=True)
    a = ap.parse_args()
    nest_bdy(a.coarse, a.bbox, a.out_bci, a.out_bdy)


def snap_bci_to_grid(bci_in, dem_asc, bci_out=None):
    """Move every boundary point onto the centre of a real perimeter cell.

    The points are generated from the requested clip bbox, but gdalwarp sets the DEM's actual
    corner to whatever the resampling produced, which differs by a fraction of a degree. At
    30 m a cell is 2.9e-4 degrees and that discrepancy is invisible. At 4 m a cell is 3.6e-5
    degrees, and points on the bottom edge then compute to row nrows, one past the last valid
    row. LISFLOOD receives a boundary point that maps to no cell and hangs searching for one:
    the run reports a healthy 10 s timestep and never completes a second one.

    Snapping preserves point count and order, so the paired .bdy stays valid. Points are moved
    to the nearest perimeter cell centre, which is at most half a cell, well inside the
    interpolation tolerance of the coarse field they were sampled from.
    """
    h = {}
    with open(dem_asc) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    nr, nc, cs = int(h["nrows"]), int(h["ncols"]), h["cellsize"]
    x0, y0 = h["xllcorner"], h["yllcorner"]
    ytop = y0 + nr * cs

    def centre(r, c):
        return x0 + (c + 0.5) * cs, ytop - (r + 0.5) * cs

    bci_out = bci_out or bci_in
    out, moved, kept = [], 0, 0
    for ln in open(bci_in):
        p = ln.split()
        if len(p) < 3 or p[0] != "P":
            out.append(ln.rstrip("\n")); continue
        x, y = float(p[1]), float(p[2])
        c = min(max(int((x - x0) / cs), 0), nc - 1)
        r = min(max(int((ytop - y) / cs), 0), nr - 1)
        # pull onto whichever perimeter it is nearest, so the point stays a boundary point
        d = {"top": r, "bot": nr - 1 - r, "left": c, "right": nc - 1 - c}
        side = min(d, key=d.get)
        r = 0 if side == "top" else nr - 1 if side == "bot" else r
        c = 0 if side == "left" else nc - 1 if side == "right" else c
        nx, ny = centre(r, c)
        if abs(nx - x) > 1e-12 or abs(ny - y) > 1e-12:
            moved += 1
        else:
            kept += 1
        out.append(f"P\t{nx:.7f}\t{ny:.7f}\t" + "\t".join(p[3:]))
    Path(bci_out).write_text("\n".join(out) + "\n")
    print(f"snapped {moved} points, {kept} already on centres -> {bci_out}")
    return bci_out


if __name__ == "__main__" and "--snap-bci" in __import__("sys").argv:
    import argparse, sys
    ap = argparse.ArgumentParser(description="Snap a .bci onto real perimeter cell centres")
    ap.add_argument("--snap-bci", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    snap_bci_to_grid(a.snap_bci, a.dem, a.out)
    sys.exit(0)
