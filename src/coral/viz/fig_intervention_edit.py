"""Figure 2 — an intervention as an edit to the landscape.

Adaptation options are usually drawn as icons on a map. Here each option is shown as what the
model actually does to the ground: which cells move, and by how much. The top row is the plan
view of the edited cells; the bottom row is a terrain profile across the same edit, so a reader
can see a seawall as a 2 m step and a marsh as a roughness change that leaves elevation alone.

Siting is not illustrative. Each panel calls the same siting code the ensemble uses, so the
placements are the ones the model would pick.

    python -m coral.viz.fig_intervention_edit --dem runs/pinpoint_highres_4m/SUB_DEM_corr_*.asc \\
        --sagis data/raw/sagis_pinpoint --out reports/figs/fig2_intervention_edit.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc
from ..interventions.generate import sample_intervention, apply_intervention
from .pinpoint_style import PALETTE, base_map, add_vector, panel_title, extent_of

KINDS = [("floodwall", "Floodwall", "a hard step along the shoreline"),
         ("marsh", "Marsh restoration", "roughness up, elevation unchanged"),
         ("living_shoreline", "Living shoreline", "a low sill with marsh behind it"),
         ("depave", "De-pave", "infiltration up, surface unchanged")]


def _context(sagis, dem_path):
    """Wetland and building masks on the DEM grid, or None when SAGIS is absent."""
    if not sagis:
        return None, None
    from ..interventions.context_rasters import wetlands_mask, buildings_mask
    S = Path(sagis)
    w = b = None
    try:
        w = wetlands_mask(S / "sagis_wetlands_nwi.geojson", dem_path)
    except Exception:
        pass
    try:
        b = buildings_mask(S / "fema_structures_pinpoint.geojson", dem_path)
    except Exception:
        pass
    return w, b


def _profile_row(mask, dem):
    """Grid row crossing the largest part of the edit. Returns None when nothing was edited."""
    counts = mask.sum(axis=1)
    return int(np.argmax(counts)) if counts.max() > 0 else None


def build(dem_path, out, *, sagis=None, nlcd=None, sea_level=0.81, res_m=4.0, seed=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dem, h = read_asc(dem_path)
    ext = extent_of(h)
    wet, bld = _context(sagis, dem_path)
    # NLCD matters more than it looks. Several kinds score only on developed classes, so
    # without it the score is near-constant over land, the quantile threshold hits a mass of
    # tied cells, and the placement blows past area_frac.
    cls = read_asc(nlcd)[0] if nlcd and Path(nlcd).exists() else None
    if nlcd and cls is None:
        print(f"  warning: NLCD {nlcd} not found; de-pave and permeable will over-place")
    rng = np.random.default_rng(seed)

    # Placeholder soil grids: this figure reads only the DEM edit and the edited-cell mask.
    man = np.full(dem.shape, 0.06, "float64")
    ksat = np.full(dem.shape, 10.0, "float64")
    awc = np.full(dem.shape, 0.15, "float64")

    n = len(KINDS)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.4),
                             gridspec_kw={"height_ratios": [3, 1.3]})

    for j, (kind, title, sub) in enumerate(KINDS):
        knobs = sample_intervention(kind, rng)
        try:
            d2, m2, k2, a2, inten = apply_intervention(
                knobs, dem, man, ksat, awc, sea_level=sea_level,
                wetlands=wet, buildings=bld, classes=cls,
                place="targeted", res_m=res_m)
        except Exception as e:
            d2, inten = dem, np.zeros(dem.shape, "float32")
            print(f"  {kind}: siting failed ({e})")

        edited = inten > 0
        dz = np.where(edited, d2 - dem, np.nan)

        ax = axes[0, j]
        base_map(ax, dem, h, sea_level=sea_level)
        if bld is not None:
            ax.imshow(np.where(bld, 1.0, np.nan), extent=ext, origin="upper",
                      cmap=_solid(PALETTE["building"]), vmin=0, vmax=1, alpha=0.5, zorder=3)
        ax.imshow(np.where(edited, 1.0, np.nan), extent=ext, origin="upper",
                  cmap=_solid(PALETTE["intervention"]), vmin=0, vmax=1, alpha=0.9, zorder=6)
        panel_title(ax, "ABCD"[j], title, sub)
        raised = float(np.nanmax(dz)) if np.isfinite(dz).any() else 0.0
        ax.text(0.03, 0.03, f"{int(edited.sum()):,} cells edited\nmax rise {raised:.2f} m",
                transform=ax.transAxes, fontsize=7.2, color=PALETTE["text"], va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=2.5))

        # --- profile across the edit ---
        axp = axes[1, j]
        r = _profile_row(edited, dem)
        if r is None:
            axp.text(0.5, 0.5, "no cells placed", ha="center", va="center",
                     transform=axp.transAxes, fontsize=8, color=PALETTE["muted"])
            axp.set_xticks([]); axp.set_yticks([])
            continue
        x = np.arange(dem.shape[1]) * res_m
        cols = np.where(edited[r])[0]
        lo, hi = cols.min(), cols.max()
        pad = max(60, int(0.35 * (hi - lo + 1)))
        sl = slice(max(0, lo - pad), min(dem.shape[1], hi + pad + 1))
        axp.axhspan(float(np.nanmin(dem[r, sl])) - 1, sea_level, color=PALETTE["water"],
                    alpha=0.55, lw=0)
        axp.fill_between(x[sl], np.nanmin(dem[r, sl]) - 1, dem[r, sl],
                         color=PALETTE["land"], zorder=2)
        # The before line is drawn thick and the after line thin on top, so a kind that does
        # not touch elevation still reads as two coincident lines rather than one.
        axp.plot(x[sl], dem[r, sl], color=PALETTE["terrain"], lw=2.4, alpha=0.8, zorder=3,
                 label="before")
        axp.plot(x[sl], d2[r, sl], color=PALETTE["intervention"], lw=1.0, zorder=4,
                 label="after")
        axp.set_xlabel("distance along profile (m)", fontsize=7)
        axp.tick_params(labelsize=6.5)
        if j == 0:
            axp.set_ylabel("elevation (m NAVD88)", fontsize=7)
            axp.legend(fontsize=6.5, frameon=False, loc="upper left")
        for s in axp.spines.values():
            s.set_edgecolor(PALETTE["muted"]); s.set_linewidth(0.6)
        axp.spines["top"].set_visible(False); axp.spines["right"].set_visible(False)

    fig.subplots_adjust(hspace=0.14, wspace=0.16, top=0.9)
    fig.suptitle("What each adaptation option does to the ground", fontsize=14, y=1.0,
                 color=PALETTE["text"])
    fig.text(0.5, -0.01,
             "Only the seawall and the living-shoreline sill change the terrain. Marsh and "
             "de-pave change roughness and infiltration, so the before and after profiles lie "
             "on top of each other — which is why these options cannot be compared by eye, and "
             "why the flood model has to be run.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def _solid(hexcolor):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("_s", [hexcolor, hexcolor])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--sagis", default="data/raw/sagis_pinpoint")
    ap.add_argument("--nlcd", default=None,
                    help="NLCD grid on the DEM; several kinds mis-place without it")
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--res-m", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports/figs/fig2_intervention_edit.png")
    a = ap.parse_args()
    build(a.dem, a.out, sagis=a.sagis, nlcd=a.nlcd, sea_level=a.sea_level, res_m=a.res_m, seed=a.seed)


if __name__ == "__main__":
    main()
