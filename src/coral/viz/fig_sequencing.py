"""Figure 4 — compound flooding is a timing argument.

"Compound" means the drivers arrive at different times. A map of maximum depth hides that
completely, because it collapses the whole event onto one frame. This figure puts a row of
model snapshots above the forcing that produced them, so a reader can see water arrive from the
estuary on the surge peak and stand inland after the rain, hours apart.

The strip reads left to right in model time. The hydrograph below carries the same time axis,
with a marker under each snapshot, so every map can be located on the forcing.

    python -m coral.viz.fig_sequencing --results runs/pinpoint_highres_4m/results_corr \\
        --dem <dem.asc> --bdy runs/pinpoint_highres_4m/pinpoint_highres_4m.bdy \\
        --out reports/figs/fig4_sequencing.png
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc
from .pinpoint_style import PALETTE, extent_of, make_flood_cmap

MODEL_T0_UTC = "2016-10-06T13:00:00"


def snapshot_times(results, root=None):
    """Sorted (index, path, time_s) for the .wd snapshots. Time comes from the matching .mass
    line when available, otherwise from the snapshot interval in the run."""
    R = Path(results)
    wd = sorted(R.glob("*.wd"), key=lambda p: p.name)
    wd = [p for p in wd if re.search(r"-(\d{4})\.wd$", p.name)]
    idx = [int(re.search(r"-(\d{4})\.wd$", p.name).group(1)) for p in wd]
    return list(zip(idx, wd))


def read_bdy(path):
    """Water level series per block from a LISFLOOD .bdy file: {name: (t_s, z_m)}."""
    lines = [l.rstrip() for l in Path(path).read_text().splitlines()]
    out, i = {}, 1                                   # line 0 is a free-text comment
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        name = lines[i].split()[0]
        parts = lines[i + 1].split()
        n = int(parts[0])
        t, z = [], []
        for k in range(n):
            a, b = lines[i + 2 + k].split()[:2]
            z.append(float(a)); t.append(float(b))
        out[name] = (np.array(t), np.array(z))
        i += 2 + n
    return out


def build(results, dem_path, out, *, bdy=None, par=None, n_frames=6, sea_level=0.81,
          dry=0.05, zoom=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    dem, h = read_asc(dem_path)
    ext = extent_of(h)
    clip = zoom or ext
    snaps = snapshot_times(results)
    if not snaps:
        raise SystemExit(f"no .wd snapshots in {results}")

    # Even spread over the run, skipping the first frame which is usually still spinning up.
    pick = np.linspace(1, len(snaps) - 1, n_frames).round().astype(int)
    frames = [snaps[i] for i in pick]

    t0_s, saveint_s = par_timing(par) if par else (0.0, 3600.0)
    if not par:
        print("  warning: no --par, so the time axis is snapshot number, not model hours")
    series = read_bdy(bdy) if bdy else {}
    cmap = make_flood_cmap()

    fig = plt.figure(figsize=(3.0 * n_frames, 5.6))
    gs = GridSpec(2, n_frames, height_ratios=[3, 1.25], hspace=0.32, wspace=0.05, figure=fig)

    land = np.isfinite(dem) & (dem > sea_level)
    times_h, flooded = [], []

    for j, (idx, path) in enumerate(frames):
        wd, _ = read_asc(path)
        t_h = (t0_s + idx * saveint_s) / 3600.0
        times_h.append(t_h)
        wet = np.where(land & (wd > dry), wd, np.nan)
        flooded.append(float(np.nansum(land & (wd > dry)) * 16.0 / 1e4))   # hectares at 4 m

        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.where(np.isfinite(dem), 1.0, np.nan), extent=ext, origin="upper",
                  cmap=_solid(PALETTE["land"]), vmin=0, vmax=1)
        ax.imshow(np.where(np.isfinite(dem) & (dem <= sea_level), 1.0, np.nan), extent=ext,
                  origin="upper", cmap=_solid(PALETTE["water"]), vmin=0, vmax=1)
        im = ax.imshow(wet, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=2.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(clip[0], clip[1]); ax.set_ylim(clip[2], clip[3])
        for s in ax.spines.values():
            s.set_edgecolor(PALETTE["muted"]); s.set_linewidth(0.5)
        ax.set_title(f"t = {t_h:.0f} h", fontsize=10, color=PALETTE["text"], pad=6)
        ax.text(0.04, 0.04, f"{flooded[-1]:.0f} ha on land", transform=ax.transAxes,
                fontsize=7.4, color=PALETTE["text"],
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))

    # --- forcing, on the same time axis ---
    axh = fig.add_subplot(gs[1, :])
    if series:
        # The boundary has one block per point source. Drawing all of them is unreadable, so
        # this shows the spread as a band and the median as the line.
        # The file mixes water-level blocks with river-discharge blocks, which are in different
        # units. Plotting both on one axis is meaningless, so keep only blocks whose values sit
        # in a plausible water-level range.
        wl = {k: v for k, v in series.items()
              if -3.0 < np.nanmin(v[1]) and np.nanmax(v[1]) < 6.0}
        if not wl:
            wl = series
        grid = np.linspace(min(t.min() for t, _ in wl.values()),
                           max(t.max() for t, _ in wl.values()), 800)
        stack = np.vstack([np.interp(grid, t, z) for t, z in wl.values()])
        axh.fill_between(grid / 3600.0, np.percentile(stack, 10, axis=0),
                         np.percentile(stack, 90, axis=0),
                         color=PALETTE["flood"], alpha=0.25, lw=0,
                         label=f"10-90% across {len(wl)} water-level points")
        axh.plot(grid / 3600.0, np.median(stack, 0), lw=1.6, color=PALETTE["flood"],
                 label="median boundary water level")
        print(f"  boundary: {len(wl)} water-level blocks of {len(series)}; "
              f"peak median {np.median(stack, 0).max():.2f} m at "
              f"{grid[np.argmax(np.median(stack, 0))] / 3600.0:.1f} h")
    axh.axhline(sea_level, color=PALETTE["muted"], lw=0.7, ls=":")
    axh.text(0.004, sea_level, " mean sea level", transform=axh.get_yaxis_transform(),
             fontsize=6.5, color=PALETTE["muted"], va="bottom")
    for t_h in times_h:
        axh.axvline(t_h, color=PALETTE["intervention"], lw=0.8, alpha=0.6)
    axt = axh.twinx()
    axt.plot(times_h, flooded, "o-", color=PALETTE["intervention"], lw=1.4, ms=4,
             label="land flooded")
    axt.set_ylabel("land flooded (ha)", fontsize=8, color=PALETTE["intervention"])
    axt.tick_params(labelsize=7, colors=PALETTE["intervention"])
    axh.set_xlabel(f"hours from model start ({MODEL_T0_UTC} UTC)", fontsize=8)
    axh.set_ylabel("water level (m NAVD88)", fontsize=8)
    axh.tick_params(labelsize=7)
    axh.set_xlim(min(times_h) - 2, max(times_h) + 2)
    for s in (axh, axt):
        s.spines["top"].set_visible(False)
    axh.legend(fontsize=7, frameon=False, loc="upper left")

    cax = fig.add_axes([0.915, 0.42, 0.008, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("water depth on land (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("Compound flooding is a timing argument, not a total",
                 fontsize=14, y=0.99, color=PALETTE["text"])
    fig.text(0.5, -0.04,
             "Each map is one model snapshot; the red markers place it on the forcing below. "
             "A map of maximum depth would merge all of these into one frame and lose the "
             "order the drivers arrived in.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({n_frames} frames of {len(snaps)})")


def par_timing(par):
    """(tstart_s, saveint_s) from a LISFLOOD par file. Snapshot n is written at
    tstart + n*saveint, so these two numbers convert a snapshot index into model time."""
    t0 = si = None
    for line in Path(par).read_text().splitlines():
        f = line.split()
        if len(f) >= 2 and f[0] == "tstart":
            t0 = float(f[1])
        elif len(f) >= 2 and f[0] == "saveint":
            si = float(f[1])
    return (t0 or 0.0), (si or 3600.0)


def _solid(hexcolor):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("_s", [hexcolor, hexcolor])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--bdy", default=None)
    ap.add_argument("--par", default=None,
                    help="run par file; gives tstart and saveint for the time axis")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--dry", type=float, default=0.05)
    ap.add_argument("--out", default="reports/figs/fig4_sequencing.png")
    a = ap.parse_args()
    build(a.results, a.dem, a.out, bdy=a.bdy, par=a.par, n_frames=a.n_frames,
          sea_level=a.sea_level, dry=a.dry)


if __name__ == "__main__":
    main()
