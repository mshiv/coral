"""Adaptation benefit against sea level, by total effect and by effect per unit built.

The two panels answer different planning questions and give different answers, which is the
point of drawing them together. Total benefit asks what to build given a free hand; benefit per
square metre asks what is worth building given a fixed one. A measure deployed at programme scale
can lead the first while sitting mid-table in the second.

Managed retreat is drawn but marked, because regrading a footprint toward surrounding grade
deepens water where a structure stood while removing the structure. Its depth-based benefit is
not comparable with the others and its result is an exposure change.

Reads the CSV that effect_metrics writes, so it needs no simulation.

Usage:
  python -m coral.analysis.benefit_curves --csv reports/adapt/effect_metrics_pp4_e01.csv \
      --out reports/figures/benefit_vs_slr.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# Sea-level offsets by scenario label. Sorting the labels as strings puts Low2050 after High2100,
# which silently reverses the x-axis.
OFFSET = {"slr0.0": 0.0, "slrLow2050": 0.219, "slrIntLow2050": 0.261, "slrInt2050": 0.301,
          "slrIntHigh2050": 0.361, "slrHigh2050": 0.420, "slrInt2100": 1.098,
          "slrHigh2100": 2.043}
COLOUR = {"floodwall": "#8c2d19", "road_raise": "#c26a3d", "living_shoreline": "#2c7fb8",
          "depave": "#4c9a6a", "marsh_restoration": "#7a9e3f", "marsh_migration": "#a9be8e",
          "retreat": "#8a8a8a"}
LABEL = {"floodwall": "floodwall", "road_raise": "raised road",
         "living_shoreline": "living shoreline", "depave": "de-paving",
         "marsh_restoration": "marsh restoration", "marsh_migration": "marsh migration",
         "retreat": "managed retreat"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="reports/figures/benefit_vs_slr.png")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if "+" not in r["kind"]]
    if not rows:
        raise SystemExit("no single-intervention rows in the CSV")
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["slr"])].append(r)
    levels = sorted({s for _, s in by}, key=lambda s: OFFSET.get(s, 0.0))
    kinds = sorted({k for k, _ in by})
    has_fp = any(r.get("benefit_per_m2") not in (None, "", "nan") for r in rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 2 if has_fp else 1
    fig, ax = plt.subplots(1, ncol, figsize=(7.2 * ncol, 5.0), squeeze=False)
    ax = ax[0]

    def series(kind, field):
        out = []
        for s in levels:
            v = by.get((kind, s))
            if not v:
                continue
            vals = [float(x[field]) for x in v
                    if x.get(field) not in (None, "", "nan")]
            if vals:
                out.append((OFFSET.get(s, 0.0), float(np.median(vals))))
        return out

    for k in kinds:
        pts = series(k, "benefit_m3")
        if len(pts) < 2:
            continue
        x, y = zip(*pts)
        dashed = k == "retreat"
        ax[0].plot(x, y, "o-" if not dashed else "o--", color=COLOUR.get(k, "0.4"),
                   lw=1.8, ms=4, label=LABEL.get(k, k) + (" (see note)" if dashed else ""))
    ax[0].set_yscale("log")
    ax[0].set_xlabel("sea-level offset (m)")
    ax[0].set_ylabel("median benefit within the target zone (m$^3$)")
    ax[0].set_title("total benefit", fontsize=11)
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    # The 2050 reporting horizon, shaded so the two 2100 offsets read as brackets rather than
    # as forecasts anyone plans against.
    ax[0].axvspan(0.219, 0.420, color="0.85", alpha=0.35, zorder=0)
    ax[0].annotate("2050 horizon", xy=(0.32, ax[0].get_ylim()[0]), xytext=(0.32, ax[0].get_ylim()[0]),
                   fontsize=8, color="0.45", ha="center", va="bottom")

    if has_fp:
        for k in kinds:
            pts = series(k, "benefit_per_m2")
            if len(pts) < 2:
                continue
            x, y = zip(*pts)
            ax[1].plot(x, y, "o-" if k != "retreat" else "o--",
                       color=COLOUR.get(k, "0.4"), lw=1.8, ms=4, label=LABEL.get(k, k))
        ax[1].set_yscale("log")
        ax[1].set_xlabel("sea-level offset (m)")
        ax[1].set_ylabel("median benefit per m$^2$ built (m$^3$ m$^{-2}$)")
        ax[1].set_title("benefit per unit built", fontsize=11)
        ax[1].grid(alpha=0.25, lw=0.5)
        ax[1].axvspan(0.219, 0.420, color="0.85", alpha=0.35, zorder=0)

    fig.suptitle("Adaptation benefit against sea level: what to build, and what is worth building",
                 fontsize=13)
    fig.text(0.5, 0.005,
             "Managed retreat is dashed: regrading a footprint deepens water where a structure "
             "stood while removing it, so its depth-based benefit is not comparable.",
             ha="center", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")

    print(f"\n{'kind':20s} " + " ".join(f"{OFFSET.get(s,0):>8.2f}" for s in levels))
    for k in kinds:
        d = dict(series(k, "benefit_m3"))
        print(f"{k:20s} " + " ".join(
            f"{d.get(OFFSET.get(s,0), float('nan')):8.0f}" for s in levels))


if __name__ == "__main__":
    main()
