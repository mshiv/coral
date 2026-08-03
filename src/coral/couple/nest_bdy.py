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
import pathlib
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


def _fill_holding(series):
    """Forward then backward fill a series, holding the nearest finite value.

    Returns None if the series is entirely dry, which the caller drops. Never invents a value
    from outside the observed stage range.
    """
    v = np.asarray(series, dtype="float64").copy()
    ok = np.isfinite(v)
    if not ok.any():
        return None
    idx = np.where(ok, np.arange(len(v)), -1)
    np.maximum.accumulate(idx, out=idx)                 # forward fill
    first = int(np.argmax(ok))
    idx[idx < 0] = first                                # backward fill the leading gap
    return v[idx]


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
        # Fill dry gaps by holding the nearest wet stage, never with the ground elevation.
        #
        # Filling with dem[j, i] writes a "water surface" at bed level, which in a tidal
        # channel is several metres below datum. LISFLOOD takes an HVAR value as a water
        # surface and forms depth = WSE - bed on the FINE grid, whose bed differs from the
        # coarse one, so the result is negative. sqrt(g*h) on a negative depth is NaN, the
        # timestep becomes NaN, and the run loops forever while reporting a healthy timestep
        # and consuming every core. That failure took thirteen diagnostic runs to isolate.
        #
        # Holding the nearest wet value keeps the series a physically meaningful stage
        # throughout. A point that is dry simply sits at the last stage it had, which is the
        # correct one-way nesting statement: no information flows from the fine grid back to
        # the coarse one, so a dry perimeter cell should impose no head gradient.
        filled = _fill_holding(series)
        if filled is None:
            continue
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


def extend_bdy_to_zero(bdy_in, bdy_out=None):
    """Extend every HVAR block back to t = 0 by holding its first value.

    LISFLOOD locates the pair of samples bracketing the current model time by scanning the
    series. A series that begins after t = 0 leaves nothing to find below its first entry, and
    the run hangs: it reports a healthy timestep, burns every core, and never completes a
    second step. The working 30 m boundary starts at 0.00000; this one started at 86400 s
    because nest_bdy wrote the coarse run's absolute clock.

    Holding the first value back to zero is safe here because the sampled series begins a day
    before landfall, when the coarse field is at its undisturbed tidal stage. It changes
    nothing inside the window that is actually simulated.
    """
    lines = pathlib.Path(bdy_in).read_text().splitlines()
    out = [lines[0]]                       # comment header
    i, nfix = 1, 0
    while i < len(lines):
        name = lines[i].strip()
        if not name:
            i += 1; continue
        parts = lines[i + 1].split()
        n, unit = int(parts[0]), (parts[1] if len(parts) > 1 else "seconds")
        rows = lines[i + 2:i + 2 + n]
        first = rows[0].split()
        t0 = float(first[1])
        if t0 > 0:
            rows = [f"{float(first[0]):.4f}\t0.0"] + rows
            n += 1; nfix += 1
        out += [name, f"{n}\t\t{unit}"] + rows
        i += 2 + int(parts[0])
    bdy_out = bdy_out or bdy_in
    pathlib.Path(bdy_out).write_text("\n".join(out) + "\n")
    print(f"extended {nfix} blocks back to t=0 -> {bdy_out}")
    return bdy_out


def clamp_bdy_to_bed(bci, bdy, dem_asc, margin=0.02, bdy_out=None):
    """Raise every boundary stage to at least the FINE grid's bed at that point.

    The series is sampled from the coarse run, whose channels reach about -10 m, while the
    fine DEM is floored at -1 m to keep the timestep tractable. A coarse cell with its bed at
    -3.5 m carrying 6 cm of water gives a legitimate water surface of -3.44 m, but on the fine
    grid that is 2.4 m below the bed. LISFLOOD forms depth = WSE - bed, gets a negative depth,
    and sqrt(g*h) returns NaN, which freezes the timestep and hangs the run.

    Flooring the DEM and nesting a boundary from an unfloored parent are individually
    reasonable and jointly inconsistent. This reconciles them at the boundary, which is where
    the two grids meet.

    Clamping only ever raises a stage, and only where the coarse channel is deeper than the
    fine floor, so it cannot suppress the surge.
    """
    h = {}
    with open(dem_asc) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    dem = np.loadtxt(dem_asc, skiprows=6)
    nr, nc, cs = int(h["nrows"]), int(h["ncols"]), h["cellsize"]
    ytop = h["yllcorner"] + nr * cs

    bed = {}
    for ln in open(bci):
        p = ln.split()
        if len(p) < 5 or p[0] != "P":
            continue
        x, y = float(p[1]), float(p[2])
        c = min(max(int((x - h["xllcorner"]) / cs), 0), nc - 1)
        r = min(max(int((ytop - y) / cs), 0), nr - 1)
        bed[p[4]] = float(dem[r, c])

    lines = pathlib.Path(bdy).read_text().splitlines()
    out, i, nraised, worst = [lines[0]], 1, 0, 0.0
    while i < len(lines):
        nm = lines[i].strip()
        if not nm:
            i += 1; continue
        parts = lines[i + 1].split()
        n, unit = int(parts[0]), (parts[1] if len(parts) > 1 else "seconds")
        floor = bed.get(nm, -1e9) + margin
        rows = []
        for r_ in lines[i + 2:i + 2 + n]:
            v, tt = r_.split()[:2]
            v = float(v)
            if v < floor:
                worst = max(worst, floor - v); v = floor; nraised += 1
            rows.append(f"{v:.4f}\t{float(tt):.1f}")
        out += [nm, f"{n}\t\t{unit}"] + rows
        i += 2 + n
    bdy_out = bdy_out or bdy
    pathlib.Path(bdy_out).write_text("\n".join(out) + "\n")
    print(f"clamped {nraised} samples to the fine bed (largest raise {worst:.2f} m) -> {bdy_out}")
    return bdy_out


if __name__ == "__main__" and "--clamp-bdy" in __import__("sys").argv:
    import argparse, sys
    ap = argparse.ArgumentParser(description="Clamp .bdy stages to the fine DEM bed")
    ap.add_argument("--clamp-bdy", required=True); ap.add_argument("--bci", required=True)
    ap.add_argument("--dem", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    clamp_bdy_to_bed(a.bci, a.clamp_bdy, a.dem, bdy_out=a.out)
    sys.exit(0)


if __name__ == "__main__" and "--extend-bdy" in __import__("sys").argv:
    import argparse, sys
    ap = argparse.ArgumentParser(description="Extend .bdy blocks back to t=0")
    ap.add_argument("--extend-bdy", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    extend_bdy_to_zero(a.extend_bdy, a.out)
    sys.exit(0)


if __name__ == "__main__" and "--snap-bci" in __import__("sys").argv:
    import argparse, sys
    ap = argparse.ArgumentParser(description="Snap a .bci onto real perimeter cell centres")
    ap.add_argument("--snap-bci", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    snap_bci_to_grid(a.snap_bci, a.dem, a.out)
    sys.exit(0)
