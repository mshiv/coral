"""Audit marsh-migration opportunity against the community focus radius.

No hydrodynamics are run.  The audit separates (1) growth of the eligible corridor from
(2) growth caused merely by applying an area fraction to a larger zone.  Alongside the usual
fractional selection it therefore holds the selected cell count fixed at the 3 km reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..interventions.context_rasters import buildings_mask, wetlands_mask
from ..interventions.generate import focus_region
from ..interventions.siting import suitability_score
from ..preprocess.make_manning import classes_on_dem

NLCD_DEVELOPED = (21, 22, 23, 24)


def header(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    return h


def extent(path):
    h = header(path)
    return [h["xllcorner"], h["xllcorner"] + h["ncols"] * h["cellsize"],
            h["yllcorner"], h["yllcorner"] + h["nrows"] * h["cellsize"]]


def ranked(score, n, seed):
    flat = score.ravel(); pos = np.flatnonzero(flat > 0)
    n = min(int(n), pos.size)
    out = np.zeros(flat.size, bool)
    if n:
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(pos.size), -flat[pos]))
        out[pos[order[:n]]] = True
    return out.reshape(score.shape)


def run(a):
    classes, dem = classes_on_dem(a.nlcd, a.dem)
    wet = wetlands_mask(a.wetlands, a.dem,
                        cowardin_prefixes=("E2EM", "E2SS", "E2FO", "E2US"))
    buildings = buildings_mask(a.buildings, a.dem) if a.buildings else np.zeros(dem.shape, bool)
    developed = np.isin(classes, NLCD_DEVELOPED)
    dist_wet = ndimage.distance_transform_edt(~wet) * a.cell_m
    ex = extent(a.dem)
    radii = [None if r <= 0 else float(r) for r in a.radii_km]
    focuses = {r: np.ones(dem.shape, bool) if r is None else
               focus_region(dem.shape, ex, a.ref_point, r) for r in radii}

    # Exact production suitability, changing only the focus mask.
    scores = {r: suitability_score(
        dem, "marsh_migration", sea_level=a.mhw + a.slr, wetlands=wet,
        buildings=buildings, classes=classes, focus=focuses[r], mhw=a.mhw,
        mlw=a.mlw, slr_buffer=a.slr, res_m=a.cell_m) for r in radii}
    ref = min(radii, key=lambda r: abs((r if r is not None else 1e9) - a.reference_km))
    nref = max(1, round(a.area_frac * np.count_nonzero(scores[ref] > 0)))
    area_cell_ha = a.cell_m ** 2 / 1e4
    rows, masks = [], {}
    for r in radii:
        focus = focuses[r]
        band = (np.isfinite(dem) & (dem >= a.mhw) & (dem <= a.mhw + a.slr) & focus)
        eligible = band & ~developed & ~buildings & ~wet
        connected = eligible & (dist_wet <= a.adjacency_m)
        fractional = ranked(scores[r], round(a.area_frac * np.count_nonzero(scores[r] > 0)), a.seed)
        fixed = ranked(scores[r], nref, a.seed)
        lab, ncomp = ndimage.label(connected, structure=np.ones((3, 3), int))
        sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([], int)
        rows.append({
            "radius_km": r, "label": "unrestricted" if r is None else f"{r:g} km",
            "focus_cells": int(focus.sum()), "band_ha": float(band.sum() * area_cell_ha),
            "eligible_new_land_ha": float(eligible.sum() * area_cell_ha),
            "connected_new_land_ha": float(connected.sum() * area_cell_ha),
            "connected_fraction": float(connected.sum() / max(eligible.sum(), 1)),
            "fractional_selected_ha": float(fractional.sum() * area_cell_ha),
            "fixed_selected_ha": float(fixed.sum() * area_cell_ha),
            "fixed_requested_ha": float(nref * area_cell_ha),
            "connected_components": int(ncomp),
            "largest_connected_patch_ha": float(sizes.max() * area_cell_ha) if sizes.size else 0.0,
        })
        masks[r] = (eligible, connected, fixed)
    return dem, ex, rows, masks, ref


def plot(dem, ex, rows, masks, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)
    ax = axes.ravel()
    x = np.arange(len(rows)); labels = [r["label"] for r in rows]
    ax[0].plot(x, [r["eligible_new_land_ha"] for r in rows], "o-", label="eligible new land")
    ax[0].plot(x, [r["connected_new_land_ha"] for r in rows], "o-", label="within adjacency")
    ax[0].plot(x, [r["fractional_selected_ha"] for r in rows], "o--", label="15% of each radius")
    ax[0].plot(x, [r["fixed_selected_ha"] for r in rows], "o--", label="fixed 3 km area")
    ax[0].set_xticks(x, labels); ax[0].set_ylabel("area (ha)")
    ax[0].set_title("(a) Opportunity and selection area", loc="left", fontweight="bold")
    ax[0].grid(alpha=.2); ax[0].legend(frameon=False, fontsize=7)
    shade = LightSource(315, 38).shade(np.nan_to_num(dem, nan=np.nanmedian(dem)),
                                      cmap=plt.cm.Greys, vert_exag=.3, blend_mode="soft")
    for i, row in enumerate(rows[:5], 1):
        r = row["radius_km"]; eligible, connected, fixed = masks[r]
        ax[i].imshow(shade, extent=ex, origin="upper")
        ax[i].imshow(np.where(eligible, 1, np.nan), extent=ex, origin="upper",
                     cmap="Oranges", alpha=.35, vmin=0, vmax=1)
        ax[i].imshow(np.where(connected, 1, np.nan), extent=ex, origin="upper",
                     cmap="Greens", alpha=.50, vmin=0, vmax=1)
        ax[i].contour(fixed.astype(float), levels=[.5], colors="#CC79A7", linewidths=.8,
                      extent=ex, origin="upper")
        ax[i].set_title(f"({chr(97+i)}) {row['label']}: {row['eligible_new_land_ha']:.1f} ha eligible",
                        loc="left", fontsize=9, fontweight="bold")
        ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.legend(handles=[Patch(fc="#E69F00", alpha=.35, label="eligible new land"),
                        Patch(fc="#009E73", alpha=.50, label="connected"),
                        Patch(fc="none", ec="#CC79A7", label="fixed-area targeted selection")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Marsh-migration opportunity versus community focus radius",
                 fontsize=14, fontweight="bold")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", required=True); ap.add_argument("--nlcd", required=True)
    ap.add_argument("--wetlands", required=True); ap.add_argument("--buildings")
    ap.add_argument("--mhw", type=float, required=True); ap.add_argument("--mlw", type=float, required=True)
    ap.add_argument("--slr", type=float, default=.3005); ap.add_argument("--cell-m", type=float, required=True)
    ap.add_argument("--ref-point", nargs=2, type=float, default=(-81.0903, 31.9522))
    ap.add_argument("--radii-km", nargs="+", type=float, default=(1, 3, 5, 10, 0),
                    help="zero means unrestricted")
    ap.add_argument("--reference-km", type=float, default=3.0)
    ap.add_argument("--area-frac", type=float, default=.15)
    ap.add_argument("--adjacency-m", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out-json", required=True); ap.add_argument("--out-fig", required=True)
    a = ap.parse_args()
    dem, ex, rows, masks, ref = run(a)
    report = {"slr_m": a.slr, "mhw_m": a.mhw, "reference_radius_km": ref,
              "area_frac": a.area_frac, "adjacency_m": a.adjacency_m, "rows": rows}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(report, indent=2) + "\n")
    plot(dem, ex, rows, masks, a.out_fig)
    print(f"{'radius':>12} {'eligible ha':>12} {'connected ha':>14} {'frac selected':>14} {'fixed selected':>14}")
    for r in rows:
        print(f"{r['label']:>12} {r['eligible_new_land_ha']:12.2f} {r['connected_new_land_ha']:14.2f} "
              f"{r['fractional_selected_ha']:14.2f} {r['fixed_selected_ha']:14.2f}")
    print(f"wrote {a.out_json}\nwrote {a.out_fig}")


if __name__ == "__main__":
    main()
