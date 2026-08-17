"""Figure 1 — Pin Point as a system, and the three ways water arrives.

A flood map shows where water ended up. This shows the place that produced it: the terrain, the
marsh, the houses, and the stormwater network, then the three drivers that act on them.

Panel A is the geography with no hazard on it at all, so the reader learns the place before any
result is shown. Panels B to D separate the drivers, because the compound argument only makes
sense once a reader has seen that they arrive by different routes: tide and surge across the
marsh edge from the estuary, rainfall onto the whole surface and into the pipes.

    python -m coral.viz.fig_water_system --dem runs/pinpoint_highres_4m/SUB_DEM_corr_*.asc \\
        --sagis data/raw/sagis_pinpoint --out reports/figs/fig1_water_system.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc
from .pinpoint_style import (PALETTE, base_map, add_vector, add_scalebar, add_north,
                             panel_title, extent_of)

PIN_POINT = (-81.0903, 31.9522)


def build(dem_path, out, *, sagis=None, drainage=None, buildings=None,
          sea_level=0.81, zoom=None, scalebar_km=1.0, surge_m=2.6,
          marsh_classes=("E2EM", "E2SS", "E2FO"), place="Pin Point, Georgia",
          ms_scale=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dem, h = read_asc(dem_path)
    ext = extent_of(h)
    marsh = None
    if sagis and Path(sagis, "sagis_wetlands_nwi.geojson").exists():
        from ..interventions.context_rasters import wetlands_mask
        # Restrict to VEGETATED estuarine wetland. Plain NWI also carries E1UB subtidal
        # bottom and marine deepwater, which at domain scale paints the open ocean as marsh.
        marsh = wetlands_mask(Path(sagis) / "sagis_wetlands_nwi.geojson", dem_path,
                              cowardin_prefixes=tuple(marsh_classes))
    clip = zoom or ext
    S = Path(sagis) if sagis else None
    D = Path(drainage) if drainage else S
    bld = buildings or (str(S / "fema_structures_pinpoint.geojson") if S else None)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))

    # --- A. the place, with no hazard on it ------------------------------------------------
    ax = axes[0]
    base_map(ax, dem, h, sea_level=sea_level, marsh=marsh)
    if bld:
        add_vector(ax, bld, color=PALETTE["building"], zorder=4, clip=clip)
    if clip[0] <= PIN_POINT[0] <= clip[1] and clip[2] <= PIN_POINT[1] <= clip[3]:
        ax.plot(*PIN_POINT, marker="*", ms=13, color=PALETTE["text"], zorder=22)
        ax.annotate("Pin Point", PIN_POINT, xytext=(6, 7), textcoords="offset points",
                    fontsize=8.5, color=PALETTE["text"], zorder=22)
    add_scalebar(ax, ext, km=scalebar_km); add_north(ax, ext)
    panel_title(ax, "A", "The place",
                "marsh, houses and high ground on a tidal peninsula")
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", color=PALETTE["water"], label="open water and channels"),
        Line2D([], [], marker="s", ls="", color=PALETTE["marsh"], label="tidal marsh (NWI)"),
        Line2D([], [], marker="s", ls="", color=PALETTE["building"], label="structures (FEMA)"),
    ], loc="lower right", fontsize=6.8, frameon=False)

    # --- B. tide ---------------------------------------------------------------------------
    ax = axes[1]
    base_map(ax, dem, h, sea_level=sea_level, marsh=marsh)
    # the intertidal band: ground the tide covers and uncovers twice a day
    _band(ax, dem, h, ext, sea_level, sea_level + 1.0, "#3E7FA6")
    panel_title(ax, "B", "Tide", "twice daily, through the channels; range about 2 m")

    # --- C. surge --------------------------------------------------------------------------
    ax = axes[2]
    base_map(ax, dem, h, sea_level=sea_level, marsh=marsh)
    _band(ax, dem, h, ext, sea_level, sea_level + surge_m, "#C0392B")
    panel_title(ax, "C", "Storm surge",
                "one event, from the open coast; Matthew reached 2.58 m")

    # --- D. rainfall and the pipes ---------------------------------------------------------
    # Only the Chatham County layers cover Pin Point. The City of Savannah stormwater layers
    # (inlets, manholes, conduits, tide gates) return nothing here, because Pin Point is
    # unincorporated county land outside the city inventory.
    ax = axes[3]
    base_map(ax, dem, h, sea_level=sea_level, marsh=marsh)
    if bld:
        add_vector(ax, bld, color=PALETTE["building"], alpha=0.35, zorder=4, clip=clip)
    if D:
        n_pipe = add_vector(ax, D / "sagis_pipes_chatham.geojson", color=PALETTE["pipe"],
                            lw=0.8 * ms_scale, zorder=5, clip=clip)
        n_dit = add_vector(ax, D / "sagis_ditches_chatham.geojson", color=PALETTE["inlet"],
                           lw=1.0 * ms_scale, zorder=6, clip=clip)
        n_can = add_vector(ax, D / "sagis_canals_chatham.geojson", color=PALETTE["inlet"],
                           lw=1.8 * ms_scale, zorder=7, clip=clip)
        n_out = add_vector(ax, D / "sagis_outfalls_chatham.geojson", color=PALETTE["outfall"],
                           ms=26 * ms_scale, zorder=9, clip=clip)
        print(f"  drainage drawn: {n_pipe} pipes, {n_dit} ditches, {n_can} canals, "
              f"{n_out} outfalls")
    panel_title(ax, "D", "Rainfall and drainage",
                "onto the whole surface, then out through ditches and outfalls")
    ax.legend(handles=[
        Line2D([], [], color=PALETTE["pipe"], lw=1.0, label="piped storm drains"),
        Line2D([], [], color=PALETTE["inlet"], lw=1.6, label="open ditches and canals"),
        Line2D([], [], marker="o", ls="", ms=5, color=PALETTE["outfall"], label="outfalls"),
    ], loc="lower right", fontsize=6.8, frameon=False)

    # geopandas sets a data aspect when it draws a layer, so panels that carry a vector layer
    # end up a different height from those that do not. Forcing the aspect makes all four the
    # same box.
    for ax in axes:
        ax.set_xlim(clip[0], clip[1]); ax.set_ylim(clip[2], clip[3])
        ax.set_aspect("auto")

    fig.subplots_adjust(wspace=0.06, top=0.86)
    fig.suptitle(f"{place} — the system, and the three ways water arrives",
                 fontsize=14, y=1.02, x=0.5, color=PALETTE["text"])
    fig.text(0.5, -0.02,
             "The drivers reach the same ground by different routes, so their effects are not "
             "additive: the tide sets the level the surge rides on, and the pipes that drain "
             "rainfall discharge into water the surge has already raised.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def _band(ax, dem, h, ext, lo, hi, color):
    """Ground between two water levels, drawn solid with an outline.

    A translucent wash over the base water is invisible, because the base water is already
    blue. A solid band with a contour edge separates the two.
    """
    from matplotlib.colors import LinearSegmentedColormap
    m = np.isfinite(dem) & (dem > lo) & (dem <= hi)
    ax.imshow(np.where(m, 1.0, np.nan), extent=ext, origin="upper",
              cmap=LinearSegmentedColormap.from_list("_b", [color, color]),
              vmin=0, vmax=1, alpha=0.75, zorder=3)
    ax.contour(np.where(np.isfinite(dem), dem, 1e6), levels=[hi], extent=ext,
               origin="upper", colors=[color], linewidths=0.8, zorder=4)
    return int(m.sum())


def _ramp(hexcolor):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("_r", ["#FFFFFF00", hexcolor])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--sagis", default="data/raw/sagis_pinpoint")
    ap.add_argument("--drainage", default="data/raw/sagis_pinpoint_v2",
                    help="dir holding the SAGIS stormwater layers for panel D")
    ap.add_argument("--buildings", default=None)
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--scalebar-km", type=float, default=1.0)
    ap.add_argument("--surge-m", type=float, default=2.6)
    ap.add_argument("--place", default="Pin Point, Georgia")
    ap.add_argument("--ms-scale", type=float, default=1.0,
                    help="shrink line and marker sizes for a wide domain")
    ap.add_argument("--marsh-classes", nargs="+",
                    default=["E2EM", "E2SS", "E2FO"],
                    help="NWI Cowardin prefixes counted as tidal marsh")
    ap.add_argument("--zoom", nargs=4, type=float, default=None, metavar=("W", "E", "S", "N"))
    ap.add_argument("--out", default="reports/figs/fig1_water_system.png")
    a = ap.parse_args()
    build(a.dem, a.out, sagis=a.sagis, drainage=a.drainage, buildings=a.buildings,
          sea_level=a.sea_level, zoom=a.zoom, scalebar_km=a.scalebar_km,
          surge_m=a.surge_m, place=a.place,
          ms_scale=a.ms_scale, marsh_classes=a.marsh_classes)


if __name__ == "__main__":
    main()
