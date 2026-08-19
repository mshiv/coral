"""Compare any number of runs against the USGS high-water marks.

The single-run scripts answer "how far off is this run". This answers "which of these runs is
closer, and in what way", which is the question a calibration or a sensitivity test actually
asks. Runs are named on the command line, so the legend says what ran instead of whatever the
script was first written for.

Three numbers per run, because RMSE alone is misleading here. The marks span about 0.8 m of
water surface across the domain, and a model that returns a single flat level scores a
respectable RMSE while reproducing none of that variation. So this also reports:

  slope   least squares fit of modelled against observed. 1.0 means the model reproduces the
          spatial range; well under 1 means it is flattening it.
  NSE     Nash-Sutcliffe. Skill against predicting the observed mean everywhere. At or below 0
          the run carries no more information than a constant, whatever its RMSE.

Seed and debris lines record the maximum surface reached, including runup, so a depth-averaged
model is expected to sit below them. They are reported separately rather than pooled, following
Munoz et al. 2020.

    python -m coral.validate.fig_hwm_compare --dem <dem.asc> \\
        --runs baseline:<a>/res.mxe tide:<b>/res.mxe --out reports/figures/hwm_compare.png
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..analysis.physics_ab import _read_grid, _sample, _extent
from .hwm_per_mark import fetch_hwms_full, _summary
from ..viz.pinpoint_style import PALETTE

RUNUP_TYPES = ("Seed line", "Debris")


def score(mxe, dem_header_marks, *, max_quality=None):
    """[{mark fields, model, resid}] for one run, on a mark list already fetched."""
    grid, gh = _read_grid(mxe)
    rows = []
    for m in dem_header_marks:
        if max_quality is not None and (m["quality"] is None or m["quality"] > max_quality):
            continue
        mod = _sample(grid, gh, m["lon"], m["lat"])
        if not np.isfinite(mod):
            continue
        rows.append({**m, "model": float(mod), "resid": float(mod - m["obs"])})
    return rows


def skill(rows):
    """RMSE, bias, slope and NSE for one run."""
    o = np.array([r["obs"] for r in rows])
    m = np.array([r["model"] for r in rows])
    d = m - o
    sse = float(((o - m) ** 2).sum())
    sst = float(((o - o.mean()) ** 2).sum())
    slope = float(np.polyfit(o, m, 1)[0]) if o.size > 2 and o.std() > 0 else np.nan
    r = float(np.corrcoef(o, m)[0, 1]) if o.size > 2 and o.std() > 0 and m.std() > 0 else np.nan
    return dict(n=len(rows), bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                slope=slope, r=r, nse=(1.0 - sse / sst) if sst > 0 else np.nan)


def build(runs, dem, out, *, max_quality=None):
    """runs is [(label, mxe_path), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, dh = _read_grid(dem)
    marks = fetch_hwms_full(_extent(dh))
    print(f"{len(marks)} marks in the domain")

    scored = {}
    for label, path in runs:
        rows = score(path, marks, max_quality=max_quality)
        if not rows:
            print(f"  {label}: no marks scored, skipping")
            continue
        scored[label] = rows

    if not scored:
        raise SystemExit("no run scored any marks; check the .mxe paths and the DEM extent")

    hdr = f"{'run':22s} {'n':>3} {'bias':>7} {'RMSE':>6} {'slope':>6} {'r':>6} {'NSE':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    stats = {}
    for label, rows in scored.items():
        s = skill(rows)
        stats[label] = s
        print(f"{label:22s} {s['n']:3d} {s['bias']:+7.3f} {s['rmse']:6.3f} "
              f"{s['slope']:6.2f} {s['r']:6.2f} {s['nse']:+7.2f}")

    # A constant at the observed mean, as the thing every run has to beat.
    o = np.array([r["obs"] for r in next(iter(scored.values()))])
    print(f"{'(constant at obs mean)':22s} {len(o):3d} {0.0:+7.3f} {o.std():6.3f} "
          f"{0.0:6.2f} {0.0:6.2f} {0.0:+7.2f}")

    print("\nseed/debris lines against the rest (model should sit BELOW runup marks):")
    for label, rows in scored.items():
        a = [r for r in rows if r["type"] in RUNUP_TYPES]
        b = [r for r in rows if r["type"] not in RUNUP_TYPES]
        if a and b:
            sa, sb = _summary(a), _summary(b)
            print(f"  {label:20s} runup bias {sa['bias']:+.3f} (n={sa['n']}), "
                  f"other {sb['bias']:+.3f} (n={sb['n']}), "
                  f"difference {sa['bias'] - sb['bias']:+.3f} m")

    colours = [PALETTE["muted"], PALETTE["flood"], PALETTE["intervention"],
               PALETTE["terrain"], "#7FB2D3"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6),
                           gridspec_kw={"width_ratios": [1, 1.15, 1]})

    # --- A. the usual scatter, with the 1:1 and each run's fit ------------------------------
    lo = min(min(r["obs"] for rows in scored.values() for r in rows),
             min(r["model"] for rows in scored.values() for r in rows)) - 0.2
    hi = max(max(r["obs"] for rows in scored.values() for r in rows),
             max(r["model"] for rows in scored.values() for r in rows)) + 0.2
    ax[0].plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=PALETTE["text"], zorder=1)
    for k, (label, rows) in enumerate(scored.items()):
        c = colours[k % len(colours)]
        o = np.array([r["obs"] for r in rows]); m = np.array([r["model"] for r in rows])
        ax[0].scatter(o, m, s=34, color=c, alpha=0.8, zorder=3,
                      label=f"{label}  RMSE {stats[label]['rmse']:.3f}, "
                            f"slope {stats[label]['slope']:.2f}")
        if np.isfinite(stats[label]["slope"]):
            p = np.polyfit(o, m, 1)
            xs = np.array([o.min(), o.max()])
            ax[0].plot(xs, np.polyval(p, xs), lw=1.2, color=c, alpha=0.9, zorder=2)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("observed HWM water surface (m)", fontsize=9)
    ax[0].set_ylabel("modelled maximum water surface (m)", fontsize=9)
    ax[0].set_title("A  Against the marks", fontsize=11, color=PALETTE["text"])
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    ax[0].grid(alpha=0.25)

    # --- B. residual per mark, ordered by observed level -------------------------------------
    ax[1].axhline(0, color=PALETTE["text"], lw=1.0)
    for k, (label, rows) in enumerate(scored.items()):
        c = colours[k % len(colours)]
        rs = sorted(rows, key=lambda r: r["obs"])
        ax[1].plot(range(len(rs)), [r["resid"] for r in rs], "o-", ms=4.5, lw=1.0,
                   color=c, alpha=0.85, label=label)
    ax[1].set_xlabel("marks, ordered low to high observed level", fontsize=9)
    ax[1].set_ylabel("modelled minus observed (m)", fontsize=9)
    ax[1].set_title("B  Where each run goes wrong", fontsize=11, color=PALETTE["text"])
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.25)

    # --- C. bias by mark type -----------------------------------------------------------------
    types = sorted({r["type"] for rows in scored.values() for r in rows})
    w = 0.8 / max(len(scored), 1)
    for k, (label, rows) in enumerate(scored.items()):
        g = defaultdict(list)
        for r in rows:
            g[r["type"]].append(r)
        vals = [np.mean([x["resid"] for x in g[t]]) if t in g else np.nan for t in types]
        ax[2].bar(np.arange(len(types)) + k * w, vals, width=w,
                  color=colours[k % len(colours)], alpha=0.85, label=label)
    ax[2].axhline(0, color=PALETTE["text"], lw=1.0)
    ax[2].set_xticks(np.arange(len(types)) + 0.4 - w / 2)
    ax[2].set_xticklabels(types, fontsize=7.5, rotation=20, ha="right")
    ax[2].set_ylabel("mean residual (m)", fontsize=9)
    ax[2].set_title("C  Bias by mark type", fontsize=11, color=PALETTE["text"])
    ax[2].legend(fontsize=8, frameon=False)
    ax[2].grid(alpha=0.25, axis="y")

    for a in ax:
        a.tick_params(labelsize=7.5)
    fig.subplots_adjust(wspace=0.26, top=0.86)
    fig.suptitle("High-water-mark comparison", fontsize=14, y=0.965, color=PALETTE["text"])
    fig.text(0.5, -0.02,
             "Slope well under 1 means the run is flattening the spatial variation the marks "
             "record. NSE at or below 0 means it carries no more information than a constant "
             "at the observed mean, whatever its RMSE.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    metavar="LABEL:MXE", help="one or more label:path pairs")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--max-quality", type=int, default=None)
    ap.add_argument("--out", default="reports/figures/hwm_compare.png")
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        if ":" not in spec:
            raise SystemExit(f"--runs wants LABEL:PATH, got {spec!r}")
        label, path = spec.split(":", 1)
        runs.append((label, path))
    build(runs, a.dem, a.out, max_quality=a.max_quality)


if __name__ == "__main__":
    main()
