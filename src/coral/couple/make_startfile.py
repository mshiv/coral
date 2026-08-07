"""Build an initial water surface for a nested clip, sampled from its coarse parent.

Nested runs using EDGE boundary conditions under the subgrid model need one, and will not
work without it. In SGC_BCs (sgc.cpp:1354) an HFIX/HVAR edge computes flux only when the
edge cell is ALREADY wet:

    if (Arrptr->H[p0] > Solverptr->DepthThresh) { ...flux from head gradient... }
    else { *qptr = 0; *qoldptr = 0; *qSGoldptr = 0; }

On a dry start every edge cell has H = 0, the else branch fires, and no water can ever enter.
The boundary is bound correctly and passes nothing, forever. Point-source boundaries
(P ... HVAR, iterateq.cpp:566) have no such guard because they hard-set H, which is why the
30 m parent runs from dry and the 4 m clip could not.

Priming the clip with the parent's water surface at tstart breaks the deadlock: channels and
the wet parts of the perimeter start above the depth threshold, so the edge flux terms engage
from the first timestep.

Writes water surface ELEVATION, for use with `startelev`, which LISFLOOD converts to depth
against the DEM (input.cpp:1140). Cells the parent had dry are written at their own bed
elevation, giving zero depth rather than a spurious film.

    python -m coral.couple.make_startfile --coarse .../compound30m \\
        --fine-dem .../SUB_DEM_pinpoint_highres_4m.asc --tstart 64800 \\
        --out .../start_pinpoint_highres_4m.asc

Then in the .par:   startfile  start_pinpoint_highres_4m.asc
                    startelev
"""
import argparse
import numpy as np

from .nest_bdy import _coarse_wse_series


def _read_asc(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    h["ncols"], h["nrows"] = int(h["ncols"]), int(h["nrows"])
    return np.loadtxt(path, skiprows=6), h


def _write_asc(path, arr, h, nodata):
    hdr = (f"ncols        {h['ncols']}\nnrows        {h['nrows']}\n"
           f"xllcorner    {h['xllcorner']:.12f}\nyllcorner    {h['yllcorner']:.12f}\n"
           f"cellsize     {h['cellsize']:.12f}\nNODATA_value {nodata:.0f}\n")
    with open(path, "w") as f:
        f.write(hdr)
        np.savetxt(f, arr, fmt="%.4f")


def make_startfile(coarse_dir, fine_dem, out, tstart, root="res_matthew_sav", results=None,
                   saveint=1800.0, t0=86400.0, dry_thresh=0.05, nodata=-9999.0, level=None):
    """If `level` is given, prime to that still-water elevation instead of the parent's field.

    Sampling the parent's wet mask fails when the run starts before the parent's series: the
    earliest snapshot is pre-storm, the clip is dry in it, and the startfile comes back
    identical to the DEM (0 of 936468 cells primed). A still-water level is also the
    conventional way to initialise a coastal domain -- everything below the level is water,
    everything above is land -- and it does not depend on the parent having wetted the clip yet.
    """
    fd0, fh0 = _read_asc(fine_dem)
    if level is not None:
        bed0 = np.where(fd0 == nodata, np.nan, fd0)
        start0 = np.where(np.isfinite(bed0) & (bed0 < level), level, bed0)
        start0 = np.where(np.isfinite(start0), start0, nodata)
        _write_asc(out, start0, fh0, nodata)
        wet0 = np.isfinite(bed0) & (bed0 < level)
        print(f"startfile at still-water level {level:.2f} m -> {out}")
        print(f"  primed wet: {wet0.sum()} of {wet0.size} cells ({100*wet0.mean():.1f}%)")
        if wet0.mean() < 0.01:
            print("  WARNING: almost nothing is wet. Edge boundaries will not engage (sgc.cpp:1354).")
        return out

    dem, cx, cy, wse = _coarse_wse_series(coarse_dir, root, dry_thresh, results)
    times = t0 + np.arange(wse.shape[0]) * saveint
    # Nearest snapshot at or before tstart. Before the series begins, hold the first frame:
    # the series starts a day ahead of landfall at undisturbed tidal stage, so the earliest
    # frame is the right pre-storm condition rather than an extrapolation.
    k = int(np.clip(np.searchsorted(times, tstart, side="right") - 1, 0, len(times) - 1))
    field = wse[k]

    fd, fh = _read_asc(fine_dem)
    fx = fh["xllcorner"] + (np.arange(fh["ncols"]) + .5) * fh["cellsize"]
    fy = (fh["yllcorner"] + fh["nrows"] * fh["cellsize"]) - (np.arange(fh["nrows"]) + .5) * fh["cellsize"]
    i = np.clip(np.searchsorted(cx, fx) - 1, 0, len(cx) - 1)
    j = np.clip(np.searchsorted(-cy, -fy) - 1, 0, len(cy) - 1)
    prime = field[np.ix_(j, i)]

    bed = np.where(fd == nodata, np.nan, fd)
    # Never prime below the local bed: that is a negative depth, and LISFLOOD would clamp it
    # anyway. Dry parent cells fall back to the bed, i.e. zero depth.
    start = np.where(np.isfinite(prime) & (prime > bed), prime, bed)
    start = np.where(np.isfinite(start), start, nodata)

    _write_asc(out, start, fh, nodata)

    wet = np.isfinite(prime) & (prime > bed)
    print(f"startfile from snapshot {k} (t={times[k]:.0f} s, requested tstart={tstart:.0f}) -> {out}")
    print(f"  primed wet: {wet.sum()} of {wet.size} cells ({100*wet.mean():.1f}%), "
          f"WSE {np.nanmin(start[start>nodata]):.2f} to {np.nanmax(start):.2f} m")
    if wet.mean() < 0.01:
        print("  WARNING: almost nothing is wet. Edge boundaries will not engage (sgc.cpp:1354).")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse", required=True); ap.add_argument("--fine-dem", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tstart", type=float, required=True)
    ap.add_argument("--root", default="res_matthew_sav")
    ap.add_argument("--results", default=None,
                    help="results subdirectory of --coarse to prime from. Must match the one the "
                         "boundary was nested from.")
    ap.add_argument("--level", type=float, default=None,
                    help="still-water elevation (m) to prime to, instead of sampling the "
                         "parent. Use when the run starts before the parent's series, where "
                         "the earliest snapshot is pre-storm and the clip is dry in it.")
    a = ap.parse_args()
    make_startfile(a.coarse, a.fine_dem, a.out, a.tstart, root=a.root,
                   results=a.results, level=a.level)


if __name__ == "__main__":
    main()
