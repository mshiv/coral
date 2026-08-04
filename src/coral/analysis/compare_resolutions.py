"""Compare a nested fine run against its coarse parent on the overlap.

The 4 m Pin Point clip contains no USGS high-water marks -- all 23 Matthew marks that pass
the quality filter fall outside its 3.5 x 3.6 km extent -- so the fine run cannot be scored
against observations the way the 30 m parent is. The only available check is agreement with
the parent it was nested from, over the area they share.

That is a weaker claim than validation and should be reported as such: it shows the fine run
did not diverge from a parent that IS observationally validated, not that the fine run is
itself accurate. It does, however, catch a corrupted run -- a broken mass budget or an
unstable solve will not track the parent.

Water surface elevation is compared rather than depth. Depth is DEM-differenced, so two
resolutions disagree on it wherever the terrain differs, which is everywhere; elevation is
the physically comparable field. Depth is used only for flooded-area totals, each run
against its own DEM.

    python -m coral.analysis.compare_resolutions \\
        --coarse-mxe .../res_matthew_sav.mxe --coarse-dem .../SUB_DEM_SAV.asc \\
        --fine-mxe   .../res_pinpoint_highres_4m.mxe --fine-dem .../SUB_DEM_pinpoint_highres_4m.asc \\
        --out reports/resolution_4m_vs_30m.png
"""
import argparse
import numpy as np


def read_asc(path):
    """Return (array, header dict). Header keys are lowercased."""
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    a = np.loadtxt(path, skiprows=6)
    h["ncols"], h["nrows"] = int(h["ncols"]), int(h["nrows"])
    return a, h


def centres(h):
    """Cell-centre coordinate vectors (x ascending, y descending: ASCII grids are top-down)."""
    cs, nx, ny = h["cellsize"], h["ncols"], h["nrows"]
    x = h["xllcorner"] + (np.arange(nx) + 0.5) * cs
    y = (h["yllcorner"] + ny * cs) - (np.arange(ny) + 0.5) * cs
    return x, y


def sample_at(arr, h, xq, yq):
    """Nearest-neighbour sample of a grid at query coordinates. Out-of-extent -> nan.

    Nearest neighbour, not bilinear: the coarse field is what the fine run was nested from,
    and interpolating it would invent structure the fine run never saw at its boundary.
    """
    x, y = centres(h)
    out = np.full(np.shape(xq), np.nan)
    i = np.searchsorted(x, xq).clip(1, len(x) - 1)
    i = np.where(np.abs(xq - x[i - 1]) < np.abs(xq - x[i]), i - 1, i)
    yd = -y  # y descends, searchsorted needs ascending
    j = np.searchsorted(yd, -yq).clip(1, len(y) - 1)
    j = np.where(np.abs(-yq - yd[j - 1]) < np.abs(-yq - yd[j]), j - 1, j)
    inside = (xq >= x.min()) & (xq <= x.max()) & (yq >= y.min()) & (yq <= y.max())
    out[inside] = arr[j[inside], i[inside]]
    return out


def _corr(a, b):
    """Pearson r, or nan when either field is constant (np.corrcoef divides by zero there)."""
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(coarse_mxe, coarse_dem, fine_mxe, fine_dem, thresh=0.05, nodata=-9999.0, out=None):
    cm, ch = read_asc(coarse_mxe)
    cd, _ = read_asc(coarse_dem)
    fm, fh = read_asc(fine_mxe)
    fd, _ = read_asc(fine_dem)

    fx, fy = centres(fh)
    XX, YY = np.meshgrid(fx, fy)
    cm_on_fine = sample_at(np.where(cm == nodata, np.nan, cm), ch, XX, YY)
    cd_on_fine = sample_at(np.where(cd == nodata, np.nan, cd), ch, XX, YY)

    fine_wse = np.where(fm == nodata, np.nan, fm)
    fine_dep = fine_wse - np.where(fd == nodata, np.nan, fd)
    coarse_dep = cm_on_fine - cd_on_fine

    # Compare only where BOTH runs put water. A cell wet in one and dry in the other has no
    # meaningful elevation difference, and including it would mix a wet/dry disagreement into
    # a depth statistic. Wet/dry disagreement is reported separately as flooded area.
    # Cells the coarse grid can actually be sampled at. Nearest-neighbour sampling is clipped
    # to the coarse CELL CENTRES, so a half-coarse-cell border of the fine grid has no coarse
    # counterpart. Both flooded-area totals must be restricted to this domain or the fine run
    # is credited with area the coarse run was never given a chance to have.
    valid = np.isfinite(cm_on_fine) & np.isfinite(cd_on_fine)
    both = (fine_dep > thresh) & (coarse_dep > thresh) & np.isfinite(fine_wse) & valid
    d = (fine_wse - cm_on_fine)[both]

    res = {
        "n_compared": int(both.sum()),
        "bias_m": float(np.mean(d)) if d.size else float("nan"),
        "rmse_m": float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan"),
        "p95_abs_m": float(np.percentile(np.abs(d), 95)) if d.size else float("nan"),
        "corr": _corr(fine_wse[both], cm_on_fine[both]),
        "fine_wet_cells": int(np.nansum((fine_dep > thresh) & valid)),
        "coarse_wet_cells_on_fine": int(np.nansum((coarse_dep > thresh) & valid)),
    }
    cell_m2 = (fh["cellsize"] * 111000.0) ** 2  # degrees -> m, adequate for an area ratio
    res["fine_wet_km2"] = res["fine_wet_cells"] * cell_m2 / 1e6
    res["coarse_wet_km2"] = res["coarse_wet_cells_on_fine"] * cell_m2 / 1e6
    res["area_ratio"] = (res["fine_wet_km2"] / res["coarse_wet_km2"]
                         if res["coarse_wet_km2"] else float("nan"))

    if out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        diff = np.where(both, fine_wse - cm_on_fine, np.nan)
        lim = np.nanpercentile(np.abs(diff), 99) if d.size else 1.0
        for a, img, ttl, kw in [
            (ax[0], np.where(fine_dep > thresh, fine_dep, np.nan), "fine depth (m)", {}),
            (ax[1], np.where(coarse_dep > thresh, coarse_dep, np.nan), "coarse depth on fine grid (m)", {}),
            (ax[2], diff, "fine - coarse WSE (m)", dict(cmap="RdBu_r", vmin=-lim, vmax=lim)),
        ]:
            im = a.imshow(img, **kw)
            a.set_title(ttl, fontsize=10)
            a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle(f"n={res['n_compared']}  bias={res['bias_m']:+.3f} m  "
                     f"RMSE={res['rmse_m']:.3f} m  area ratio={res['area_ratio']:.2f}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse-mxe", required=True); ap.add_argument("--coarse-dem", required=True)
    ap.add_argument("--fine-mxe", required=True);   ap.add_argument("--fine-dem", required=True)
    ap.add_argument("--thresh", type=float, default=0.05, help="wet depth threshold (m)")
    ap.add_argument("--out", default=None, help="figure path")
    a = ap.parse_args()
    r = compare(a.coarse_mxe, a.coarse_dem, a.fine_mxe, a.fine_dem, a.thresh, out=a.out)
    print(f"\ncells compared (wet in both) : {r['n_compared']}")
    print(f"WSE bias (fine - coarse)     : {r['bias_m']:+.3f} m")
    print(f"WSE RMSE                     : {r['rmse_m']:.3f} m")
    print(f"WSE 95th pct |difference|    : {r['p95_abs_m']:.3f} m")
    print(f"correlation                  : {r['corr']:.4f}")
    print(f"flooded area, fine           : {r['fine_wet_km2']:.3f} km2")
    print(f"flooded area, coarse         : {r['coarse_wet_km2']:.3f} km2")
    print(f"area ratio (fine/coarse)     : {r['area_ratio']:.3f}")
    print("\nAgreement with the parent is a consistency check, not validation: the clip holds")
    print("no high-water marks. Report it as such.")


if __name__ == "__main__":
    main()
