"""Audit whether the marsh-migration corridor represents connected new habitat.

This is a suitability/provenance diagnostic, not an intervention generator.  It compares the
current hydraulic corridor with nested, increasingly defensible masks: undeveloped/unbuilt,
outside existing mapped wetland, and within a stated distance of existing tidal wetland.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..interventions.context_rasters import buildings_mask, wetlands_mask
from ..preprocess.make_manning import classes_on_dem


NLCD_DEVELOPED = (21, 22, 23, 24)


def audit(*, dem_path, nlcd_path, wetlands_path, buildings_path, mhw, slrs,
          cell_m, adjacency_m):
    classes, dem = classes_on_dem(nlcd_path, dem_path)
    wet = wetlands_mask(wetlands_path, dem_path, cowardin_prefixes=("E2EM",))
    if not wet.any():  # some SAGIS exports omit ATTRIBUTE; retain an explicit fallback
        wet = wetlands_mask(wetlands_path, dem_path)
        wet_source = "all mapped NWI wetlands (E2EM filter returned empty)"
    else:
        wet_source = "NWI E2EM estuarine emergent wetland"
    buildings = buildings_mask(buildings_path, dem_path) if buildings_path else \
        np.zeros(dem.shape, bool)
    developed = np.isin(classes, NLCD_DEVELOPED)
    dist_m = ndimage.distance_transform_edt(~wet) * cell_m
    rows = []
    for slr in slrs:
        # A zero-rise member has no migration corridor by definition.
        band = np.isfinite(dem) & (dem >= mhw) & (dem <= mhw + slr) if slr > 0 \
            else np.zeros(dem.shape, bool)
        eligible = band & ~developed & ~buildings
        new = eligible & ~wet
        connected = new & (dist_m <= adjacency_m)
        labels, ncomp = ndimage.label(connected, structure=np.ones((3, 3), int))
        sizes = np.bincount(labels.ravel())[1:] if ncomp else np.array([], int)
        area = lambda m: float(m.sum() * cell_m * cell_m / 1e4)
        rows.append({
            "slr_m": float(slr),
            "band_ha": area(band),
            "undeveloped_unbuilt_ha": area(eligible),
            "new_land_ha": area(new),
            "connected_new_land_ha": area(connected),
            "connected_fraction_of_current": float(connected.sum() / max(eligible.sum(), 1)),
            "existing_wetland_fraction": float((eligible & wet).sum() / max(eligible.sum(), 1)),
            "developed_or_building_fraction_of_band":
                float((band & (developed | buildings)).sum() / max(band.sum(), 1)),
            "connected_components": int(ncomp),
            "largest_component_ha": float(sizes.max() * cell_m * cell_m / 1e4)
                if sizes.size else 0.0,
        })
    return rows, wet_source


def plot(rows, out):
    import matplotlib.pyplot as plt
    x = np.array([r["slr_m"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for key, label in (("band_ha", "datum corridor"),
                       ("undeveloped_unbuilt_ha", "current eligible"),
                       ("connected_new_land_ha", "connected new habitat")):
        ax[0].plot(x, [r[key] for r in rows], marker="o", label=label)
    ax[0].set(xlabel="Sea-level offset (m)", ylabel="Eligible area (ha)")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=.2)
    ax[1].plot(x, [100*r["connected_fraction_of_current"] for r in rows],
               marker="o", color="#D55E00")
    ax[1].set(xlabel="Sea-level offset (m)",
              ylabel="Connected new habitat / current eligible (%)", ylim=(0, 100))
    ax[1].grid(alpha=.2)
    fig.suptitle("Marsh-migration suitability audit")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--nlcd", required=True)
    ap.add_argument("--wetlands", required=True)
    ap.add_argument("--buildings")
    ap.add_argument("--mhw", type=float, required=True)
    ap.add_argument("--slr", nargs="+", type=float, required=True)
    ap.add_argument("--cell-m", type=float, required=True)
    ap.add_argument("--adjacency-m", type=float, default=300.0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    a = ap.parse_args(argv)
    rows, source = audit(dem_path=a.dem, nlcd_path=a.nlcd,
                         wetlands_path=a.wetlands, buildings_path=a.buildings,
                         mhw=a.mhw, slrs=a.slr, cell_m=a.cell_m,
                         adjacency_m=a.adjacency_m)
    report = {"wetland_source": source, "adjacency_m": a.adjacency_m, "rows": rows}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(report, indent=2) + "\n")
    plot(rows, a.out_fig)
    print(f"{'SLR m':>7} {'band ha':>10} {'eligible ha':>12} {'connected ha':>14} {'connected %':>12}")
    for r in rows:
        print(f"{r['slr_m']:7.3f} {r['band_ha']:10.1f} {r['undeveloped_unbuilt_ha']:12.1f} "
              f"{r['connected_new_land_ha']:14.1f} "
              f"{100*r['connected_fraction_of_current']:12.1f}")
    print(f"wrote {a.out_json}\nwrote {a.out_fig}")


if __name__ == "__main__":
    main()
