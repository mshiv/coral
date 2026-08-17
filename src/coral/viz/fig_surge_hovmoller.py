"""Alongshore surge evolution as a Hovmoller.

Time against gauge latitude, coloured by surge. One panel for every coupling gauge at once, so
the surge is visible propagating along the coast rather than as sixty overlaid lines. Useful for
spotting a gauge whose series breaks step with its neighbours.

    python -m coral.viz.fig_surge_hovmoller --output <geoclaw>/_output --sea-level 0.0
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from .gauge_io import read_gauge_set
from .pinpoint_style import PALETTE


def build(output, out, *, n_gauges=63, sea_level=0.0, shift_h=6.0,
          t_range=(-42.0, 30.0), dt_h=0.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = read_gauge_set(output, range(1, n_gauges + 1))
    if not g:
        raise SystemExit(f"no coupling gauges 1-{n_gauges} in {output}")

    # Every gauge on one time axis, so rows are comparable. shift_h moves gauge t=0 onto
    # closest approach; it is a property of the storm track, not of the gauges.
    tgrid = np.arange(t_range[0], t_range[1], dt_h)
    rows, lats = [], []
    for _, _, lat, t, eta in g:
        rows.append(np.interp(tgrid, t / 3600.0 + shift_h, eta - sea_level,
                              left=np.nan, right=np.nan))
        lats.append(lat)
    lats = np.array(lats); rows = np.array(rows)
    o = np.argsort(lats)
    lats, rows = lats[o], rows[o]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(tgrid, lats, rows, shading="auto", cmap="viridis",
                       vmin=np.nanmin(rows), vmax=np.nanmax(rows))
    ax.axvline(0, color="w", ls="--", lw=1.3, alpha=0.8)
    cb = fig.colorbar(im, ax=ax); cb.set_label("surge height (m)")
    ax.set_xlabel("hours from closest approach", fontweight="bold")
    ax.set_ylabel("gauge latitude (deg N)", fontweight="bold")
    ax.set_title(f"Alongshore surge evolution, {len(g)} coupling gauges",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({len(g)} gauges, surge {np.nanmin(rows):.2f} to "
          f"{np.nanmax(rows):.2f} m)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="GeoClaw _output directory")
    ap.add_argument("--n-gauges", type=int, default=63)
    ap.add_argument("--sea-level", type=float, default=0.0)
    ap.add_argument("--shift-h", type=float, default=6.0,
                    help="hours from gauge t=0 to closest approach")
    ap.add_argument("--out", default="reports/figures/geoclaw/fig_surge_hovmoller.png")
    a = ap.parse_args()
    build(a.output, a.out, n_gauges=a.n_gauges, sea_level=a.sea_level, shift_h=a.shift_h)


if __name__ == "__main__":
    main()
