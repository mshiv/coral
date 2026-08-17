"""Alongshore profile of the boundary forcing.

Peak water level and peak surge at each coupling gauge against latitude. This is what LISFLOOD
is forced with, so it is the first thing to look at after a GeoClaw run: a gauge that sits well
away from its neighbours is a bad coupling point, not a real alongshore gradient.

    python -m coral.viz.fig_boundary_profile --output <geoclaw>/_output --sea-level 0.0
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from .gauge_io import read_gauge_set
from .pinpoint_style import PALETTE


def build(output, out, *, n_gauges=63, sea_level=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = read_gauge_set(output, range(1, n_gauges + 1))
    if not g:
        raise SystemExit(f"no coupling gauges 1-{n_gauges} in {output}")
    lat = np.array([x[2] for x in g])
    peak = np.array([np.nanmax(x[4]) for x in g])
    o = np.argsort(lat)
    lat, peak = lat[o], peak[o]
    # sea_level is the run's still-water level. With the corrected setrun it is 0.0 and surge
    # equals eta; the old runs used 0.81 and subtracting the wrong value shifts every value.
    surge = peak - sea_level

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(lat, peak, "-o", color=PALETTE["flood"], lw=1.8, ms=4, label="peak water level")
    ax.plot(lat, surge, "-o", color=PALETTE["intervention"], lw=1.8, ms=4,
            label=f"peak surge (eta - {sea_level:.2f})")
    ax.fill_between(lat, surge, peak, color=PALETTE["flood"], alpha=0.08)
    if sea_level:
        ax.axhline(sea_level, color=PALETTE["muted"], ls=":", lw=1,
                   label=f"still water {sea_level:.2f} m")
    ax.set_xlabel("gauge latitude (deg N), south to north", fontweight="bold")
    ax.set_ylabel("peak water level (m)", fontweight="bold")
    ax.set_title(f"Boundary forcing, alongshore peak profile ({len(g)} coupling gauges)",
                 fontweight="bold", pad=10)
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    ax.annotate(f"mean {peak.mean():.2f} m\nrange {peak.min():.2f}-{peak.max():.2f} m",
                xy=(0.985, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=11, bbox=dict(facecolor="#F5F5F5", edgecolor="#AAAAAA",
                                       boxstyle="round"))
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({len(g)} gauges, peak {peak.min():.2f}-{peak.max():.2f} m, "
          f"mean {peak.mean():.2f} m)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="GeoClaw _output directory")
    ap.add_argument("--n-gauges", type=int, default=63, help="coupling gauge ids 1..N")
    ap.add_argument("--sea-level", type=float, default=0.0,
                    help="still-water level of the run, subtracted to give surge")
    ap.add_argument("--out", default="reports/figures/geoclaw/fig_boundary_profile.png")
    a = ap.parse_args()
    build(a.output, a.out, n_gauges=a.n_gauges, sea_level=a.sea_level)


if __name__ == "__main__":
    main()
