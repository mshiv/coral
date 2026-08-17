"""Figure 3 — the scenario atlas: what each option buys, across sea levels.

A grid of small multiples. Rows are adaptation options, columns are sea-level rise levels, and
each cell shows the change in maximum flood depth against the no-action baseline at the same
sea level. Blue means the option removed water, red means it pushed water somewhere else.

The point of the layout is the comparison down a column and across a row at once: a reader can
see an option lose its benefit as sea level rises, and see which options trade protection in one
place for flooding in another.

Run against real ensemble output by giving --runs, a directory of member run dirs each holding a
`.max` grid and the knobs that produced it. With --synthetic the same layout is drawn from a
plausible field generated off the DEM, so the design can be reviewed before the ensemble lands.
No synthetic panel is ever labelled as a result.

    python -m coral.viz.fig_scenario_atlas --dem <dem.asc> --synthetic \\
        --out reports/figs/fig3_scenario_atlas.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc
from .pinpoint_style import PALETTE, extent_of

KINDS = ["Seawall", "Marsh restoration", "Living shoreline", "De-pave"]
SLR = [0.0, 0.3, 0.6, 1.0]

# How strongly each option reduces depth, and how much of its effect is displaced elsewhere.
# Synthetic only: these encode the expected qualitative behaviour, not a model result.
_BEHAVIOUR = {
    "Seawall":          dict(gain=0.55, displace=1.20, decay=0.75, local=True),
    "Marsh restoration": dict(gain=0.28, displace=0.15, decay=0.30, local=False),
    "Living shoreline": dict(gain=0.40, displace=0.40, decay=0.45, local=False),
    "De-pave":          dict(gain=0.16, displace=0.06, decay=0.10, local=False),
}


def _smooth(a, sigma):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, sigma)


def synthetic_baseline(dem, slr, *, sea_level=0.81, res_m=4.0):
    """Plausible max-depth field: water level plus a smooth landward decay. Not a model run."""
    from scipy import ndimage
    sea = np.isfinite(dem) & (dem <= sea_level)
    dist = ndimage.distance_transform_edt(~sea) * res_m          # metres, not cells
    head = sea_level + 1.9 + slr
    depth = np.clip(head - dem - 0.0004 * dist, 0, None)          # 0.4 m lost per km inland
    return np.where(np.isfinite(dem), depth, np.nan)


def synthetic_delta(dem, base, kind, slr, rng, *, sea_level=0.81, res_m=4.0):
    """Change in max depth for one option: a benefit field minus a displacement field.

    Restricted to land. Open water is where the option cannot help by construction, and
    including it swamped the colour scale.
    """
    from scipy import ndimage
    b = _BEHAVIOUR[kind]
    sea = np.isfinite(dem) & (dem <= sea_level)
    land = np.isfinite(dem) & (dem > sea_level)
    dist = ndimage.distance_transform_edt(~sea) * res_m        # metres from open water
    near = np.exp(-dist / (150.0 if b["local"] else 400.0))
    fade = np.exp(-b["decay"] * slr / 0.5)                     # benefit lost as SLR rises

    field = _smooth(rng.normal(size=dem.shape), 30.0)
    field = field / (np.abs(field).max() + 1e-9)
    wet = np.clip(base, 0, None)

    benefit = b["gain"] * fade * near * np.clip(field, 0, None) * wet
    # Displacement gathers where the benefit is weakest, which is what makes a wall move water
    # rather than remove it.
    displaced = b["displace"] * fade * np.clip(-field, 0, None) * (1.0 - near) * wet
    d = displaced - benefit
    return np.where(land & (wet > 0.0), d, np.nan)


def build(dem_path, out, *, synthetic=True, sea_level=0.81, seed=3, zoom=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not synthetic:
        raise SystemExit("real-run mode needs --runs; not implemented until the ensemble lands")

    dem, h = read_asc(dem_path)
    ext = extent_of(h)
    clip = zoom or ext
    rng = np.random.default_rng(seed)
    hill = np.where(np.isfinite(dem), 1.0, np.nan)

    nr, nc = len(KINDS), len(SLR)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc + 1.6, 2.9 * nr + 1.0))
    norm = TwoSlopeNorm(vmin=-0.30, vcenter=0.0, vmax=0.30)
    stats = np.zeros((nr, nc))

    for i, kind in enumerate(KINDS):
        for j, slr in enumerate(SLR):
            ax = axes[i, j]
            base = synthetic_baseline(dem, slr, sea_level=sea_level)
            d = synthetic_delta(dem, base, kind, slr, rng, sea_level=sea_level)
            ax.imshow(hill, extent=ext, origin="upper", cmap=_solid(PALETTE["land"]),
                      vmin=0, vmax=1)
            im = ax.imshow(d, extent=ext, origin="upper", cmap="RdBu_r", norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(clip[0], clip[1]); ax.set_ylim(clip[2], clip[3])
            for s in ax.spines.values():
                s.set_edgecolor(PALETTE["muted"]); s.set_linewidth(0.5)
            # Benefit as a single number: mean depth removed where the option helped.
            helped = np.nanmean(np.clip(-d, 0, None))
            worse = np.nanmean(np.clip(d, 0, None))
            stats[i, j] = helped
            ax.text(0.04, 0.05, f"−{helped:.03f} m  /  +{worse:.03f} m",
                    transform=ax.transAxes, fontsize=7.2,
                    color=PALETTE["text"],
                    bbox=dict(fc="white", ec="none", alpha=0.72, pad=2))
            if i == 0:
                ax.set_title(f"SLR +{slr:.1f} m", fontsize=10, color=PALETTE["text"], pad=8)
            if j == 0:
                ax.set_ylabel(kind, fontsize=10, color=PALETTE["text"], labelpad=10)

    fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.9, top=0.9)
    cax = fig.add_axes([0.915, 0.28, 0.012, 0.44])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("change in maximum depth (m)\nblue = water removed,  red = water displaced",
                 fontsize=8, labelpad=6)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("Scenario atlas — what each option buys, and how fast sea level takes it back",
                 fontsize=14, y=0.97, color=PALETTE["text"])
    fig.text(0.5, 0.055, "SYNTHETIC PLACEHOLDER — layout only, not model output",
             ha="center", fontsize=10, color=PALETTE["intervention"], fontweight="bold")
    fig.text(0.5, 0.02,
             "Read down a column to compare options at one sea level, and across a row to see an "
             "option lose its benefit as sea level rises. Red patches are the part of the "
             "argument a single headline number hides: water that moved rather than left. Each "
             "panel gives mean depth removed / mean depth added.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def _solid(hexcolor):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("_s", [hexcolor, hexcolor])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--synthetic", action="store_true",
                    help="draw the layout from a generated field; panels are marked as such")
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="reports/figs/fig3_scenario_atlas.png")
    a = ap.parse_args()
    build(a.dem, a.out, synthetic=a.synthetic, sea_level=a.sea_level, seed=a.seed)


if __name__ == "__main__":
    main()
