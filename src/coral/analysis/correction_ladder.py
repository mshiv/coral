"""Attribution ladder: what each physics correction does to the modelled flood.

Runs share a grid and differ only in which corrections are applied, so differencing them
attributes the change to a single correction rather than to the model as a whole:

    base  -> uncorrected DEM, NLCD Manning, POLARIS infiltration
    corr  -> + marsh DEM canopy-bias correction, + NWI-masked infiltration
    full  -> + canopy-modulated marsh roughness

Reports wetted area, flood volume and the depth difference against the first run, and draws
the difference maps. Land-masked by default: the permanently-wet channels dominate any
domain-wide statistic and hide the inland change that matters.

    python -m coral.analysis.correction_ladder \\
        --runs runs/pinpoint_highres_4m/results_matthew_sav \\
               runs/pinpoint_highres_4m/results_corr \\
               runs/pinpoint_highres_4m/results_full \\
        --labels base corr full --dem runs/pinpoint_highres_4m/SUB_DEM_corr_...asc \\
        --out reports/correction_ladder.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc


def _find_max(run_dir):
    d = Path(run_dir)
    for pat in ("*.max", "*.mxe"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    raise SystemExit(f"no .max/.mxe in {run_dir}")


def ladder(runs, labels, dem, out=None, *, thresh=0.05, sea_level=0.81, res_m=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    dem0, h = read_asc(dem)
    res_m = res_m or h["cellsize"] * 111000.0
    cell_area = res_m ** 2
    land = np.isfinite(dem0) & (dem0 > sea_level)

    depths, names = [], []
    for r, lab in zip(runs, labels):
        p = _find_max(r)
        g, _ = read_asc(p)
        if g.shape != dem0.shape:
            raise SystemExit(f"{p} shape {g.shape} != DEM {dem0.shape}")
        depths.append(np.where(np.isfinite(g), g, 0.0))
        names.append(lab)
        print(f"  {lab:6s} <- {p.parent.name}/{p.name}")

    print(f"\n{'run':8s} {'wet cells':>10s} {'wet area km2':>13s} {'volume Mm3':>11s} "
          f"{'mean depth':>11s}   (land only, > {thresh} m)")
    stats = []
    for lab, d in zip(names, depths):
        w = land & (d > thresh)
        n = int(w.sum())
        vol = float(d[w].sum() * cell_area)
        stats.append(dict(label=lab, n=n, area=n * cell_area / 1e6,
                          vol=vol / 1e6, mean=float(d[w].mean()) if n else 0.0))
        s = stats[-1]
        print(f"{lab:8s} {s['n']:10d} {s['area']:13.2f} {s['vol']:11.2f} {s['mean']:11.3f}")

    print(f"\nchange against '{names[0]}' (land only):")
    for lab, s in zip(names[1:], stats[1:]):
        b = stats[0]
        print(f"  {lab:8s} area {100*(s['area']/b['area']-1):+6.1f}%   "
              f"volume {100*(s['vol']/b['vol']-1):+6.1f}%   "
              f"mean depth {s['mean']-b['mean']:+.3f} m")

    if out:
        k = len(depths) - 1
        fig, axes = plt.subplots(1, max(k, 1), figsize=(5.6 * max(k, 1), 5.4), squeeze=False)
        ny, nx, cs = h["nrows"], h["ncols"], h["cellsize"]
        ext = (h["xllcorner"], h["xllcorner"] + nx * cs,
               h["yllcorner"], h["yllcorner"] + ny * cs)
        diffs = [np.where(land, depths[i + 1] - depths[0], np.nan) for i in range(k)]
        finite = np.concatenate([d[np.isfinite(d) & (np.abs(d) > 1e-6)] for d in diffs]) \
            if k else np.array([0.0])
        vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 0.1
        vmax = max(vmax, 1e-3)
        for i in range(k):
            ax = axes[0][i]
            ax.imshow(np.where(land, dem0, np.nan), extent=ext, origin="upper",
                      cmap="Greys", alpha=0.35)
            im = ax.imshow(np.where(np.abs(diffs[i]) > 1e-6, diffs[i], np.nan), extent=ext,
                           origin="upper", cmap="RdBu_r",
                           norm=TwoSlopeNorm(0.0, -vmax, vmax))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{names[i+1]} − {names[0]}\n"
                         f"volume {100*(stats[i+1]['vol']/stats[0]['vol']-1):+.1f}%, "
                         f"mean depth {stats[i+1]['mean']-stats[0]['mean']:+.3f} m",
                         fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                         label="peak depth difference (m)")
        fig.suptitle("Physics-correction ladder, land cells only\n"
                     "red = the correction made it wetter, blue = drier", fontsize=11)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nwrote {out}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="result dirs, first is the reference")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--thresh", type=float, default=0.05)
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--out", default="reports/correction_ladder.png")
    a = ap.parse_args()
    if len(a.runs) != len(a.labels):
        ap.error("--runs and --labels must be the same length")
    ladder(a.runs, a.labels, a.dem, a.out, thresh=a.thresh, sea_level=a.sea_level)


if __name__ == "__main__":
    main()
