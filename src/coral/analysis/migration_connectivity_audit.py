"""Compare proximity, barrier-aware, and modelled marsh-migration connectivity.

This is a suitability audit, not a marsh process model.  A maximum-depth raster only
supports a *peak-envelope* connectivity test because its wet cells need not occur at the
same time.  Ordered ``.wd`` snapshots support the stronger simultaneous/recurrent test.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..emulator.dataset import read_asc
from ..interventions.context_rasters import buildings_mask, wetlands_mask
from ..interventions.generate import focus_region
from ..interventions.siting import suitability_score
from ..preprocess.make_manning import classes_on_dem

DEVELOPED = (21, 22, 23, 24)


def _extent(h):
    return [h["xllcorner"], h["xllcorner"] + h["ncols"] * h["cellsize"],
            h["yllcorner"], h["yllcorner"] + h["nrows"] * h["cellsize"]]


def _seeded_component(passable, seeds):
    """Passable cells in an 8-connected component containing at least one seed."""
    labels, _ = ndimage.label(passable, structure=np.ones((3, 3), int))
    seed_labels = np.unique(labels[seeds & (labels > 0)])
    return np.isin(labels, seed_labels) if seed_labels.size else np.zeros(passable.shape, bool)


def _hydraulic_mask(depth, wetland, dry):
    # Existing marsh is a seed corridor.  Other cells must be simultaneously wet.
    flooded = np.isfinite(depth) & (depth >= dry)
    return _seeded_component(flooded | wetland, wetland) & flooded


def _ranked(score, n, seed):
    flat = score.ravel(); pos = np.flatnonzero(flat > 0)
    out = np.zeros(flat.size, bool); n = min(int(n), pos.size)
    if n:
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(pos.size), -flat[pos]))
        out[pos[order[:n]]] = True
    return out.reshape(score.shape)


def audit(a):
    classes, dem = classes_on_dem(a.nlcd, a.dem)
    _, hdr = read_asc(a.dem)
    wet = wetlands_mask(a.wetlands, a.dem,
                        cowardin_prefixes=("E2EM", "E2SS", "E2FO", "E2US"))
    buildings = (buildings_mask(a.buildings, a.dem) if a.buildings
                 else np.zeros(dem.shape, bool))
    developed = np.isin(classes, DEVELOPED)
    ex = _extent(hdr)
    focus = (np.ones(dem.shape, bool) if a.radius_km <= 0 else
             focus_region(dem.shape, ex, a.ref_point, a.radius_km))
    reference_focus = focus_region(dem.shape, ex, a.ref_point, a.reference_radius_km)
    band = (np.isfinite(dem) & (dem >= a.mhw) & (dem <= a.mhw + a.slr) & focus)
    eligible = band & ~wet & ~developed & ~buildings

    def score(focus_mask, *, legacy=False):
        return suitability_score(
            dem, "marsh_migration", sea_level=a.mhw + a.slr, wetlands=wet,
            buildings=buildings, classes=classes, focus=focus_mask, mhw=a.mhw,
            mlw=a.mlw, slr_buffer=a.slr, res_m=a.cell_m,
            exclude_existing_wetland=not legacy)
    legacy_reference_score = score(reference_focus, legacy=True)
    legacy_n_fixed = max(1, round(a.area_frac * np.count_nonzero(legacy_reference_score > 0)))
    production_fixed_legacy = _ranked(score(focus, legacy=True), legacy_n_fixed, a.seed)
    corrected_reference_score = score(reference_focus)
    corrected_n_fixed = max(1, round(a.area_frac * np.count_nonzero(corrected_reference_score > 0)))
    production_fixed_corrected = _ranked(score(focus), corrected_n_fixed, a.seed)

    distance = ndimage.distance_transform_edt(~wet) * a.cell_m
    proximity = eligible & (distance <= a.proximity_m)

    # Conservative static pathway: terrain no higher than the future tidal plane, with
    # developed land and structures treated as barriers. Culverts are not inferred.
    passable = np.isfinite(dem) & (dem <= a.mhw + a.slr) & ~buildings
    if a.developed_barrier:
        passable &= ~developed
    passable |= wet
    static_network = _seeded_component(passable, wet)
    barrier_connected = eligible & static_network

    peak_connected = np.zeros(dem.shape, bool)
    if a.max_depth:
        depth, _ = read_asc(a.max_depth)
        if depth.shape != dem.shape:
            raise SystemExit("--max-depth grid does not match DEM")
        peak_connected = eligible & _hydraulic_mask(depth, wet, a.dry_threshold)

    paths = sorted(glob.glob(a.snapshots)) if a.snapshots else []
    recurrent_count = np.zeros(dem.shape, np.uint32)
    for path in paths:
        depth, _ = read_asc(path)
        if depth.shape != dem.shape:
            raise SystemExit(f"snapshot grid does not match DEM: {path}")
        recurrent_count += _hydraulic_mask(depth, wet, a.dry_threshold)
    required = max(1, int(np.ceil(a.min_snapshot_fraction * len(paths)))) if paths else 0
    recurrent = eligible & (recurrent_count >= required) if paths else np.zeros(dem.shape, bool)

    area = lambda m: float(m.sum() * a.cell_m ** 2 / 1e4)
    masks = {"eligible": eligible,
             "production_fixed_legacy": production_fixed_legacy,
             "production_fixed_corrected": production_fixed_corrected,
             "proximity": proximity,
             "barrier_connected": barrier_connected,
             "peak_envelope_connected": peak_connected,
             "recurrent_snapshot_connected": recurrent}
    report = {
        "slr_m": a.slr, "radius_km": a.radius_km,
        "reference_radius_km": a.reference_radius_km, "area_frac": a.area_frac,
        "proximity_m": a.proximity_m, "dry_threshold_m": a.dry_threshold,
        "developed_land_is_hard_barrier": a.developed_barrier,
        "snapshot_count": len(paths), "minimum_snapshot_fraction": a.min_snapshot_fraction,
        "minimum_snapshot_count": required,
        "areas_ha": {k: area(v) for k, v in masks.items()},
        "fractions_of_eligible": {k: float(v.sum() / max(eligible.sum(), 1))
                                  for k, v in masks.items() if k != "eligible"},
        "pairwise_jaccard": {},
        "interpretation": {
            "production_fixed_legacy": "reconstructs the ensemble siting rule before existing wetland was excluded",
            "production_fixed_corrected": "same rank rule restricted to new migration land",
            "proximity": "Euclidean screening buffer; not hydraulic connectivity",
            "barrier_connected": "static low-elevation path to mapped wetland; no dynamics or culverts",
            "peak_envelope_connected": "connected union of maximum wet cells; cells may peak at different times",
            "recurrent_snapshot_connected": "simultaneously connected in the stated fraction of saved frames",
        },
    }
    report["production_composition_ha"] = {
        "legacy_existing_wetland": area(production_fixed_legacy & wet),
        "legacy_eligible_new_land": area(production_fixed_legacy & eligible),
        "corrected_existing_wetland": area(production_fixed_corrected & wet),
        "corrected_eligible_new_land": area(production_fixed_corrected & eligible),
    }
    keys = list(masks)
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            union = np.count_nonzero(masks[x] | masks[y])
            report["pairwise_jaccard"][f"{x}__{y}"] = (
                float(np.count_nonzero(masks[x] & masks[y]) / union) if union else None)
    return dem, ex, masks, report


def plot(dem, ex, masks, report, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    names = [("proximity", "Euclidean proximity"),
             ("barrier_connected", "Barrier-aware terrain path"),
             ("peak_envelope_connected", "Peak-depth envelope"),
             ("recurrent_snapshot_connected", "Recurrent snapshot pathway")]
    shade = LightSource(315, 38).shade(np.nan_to_num(dem, nan=np.nanmedian(dem)),
                                      cmap=plt.cm.Greys, vert_exag=.3, blend_mode="soft")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, (key, title) in enumerate(names):
        ax = axes.ravel()[i]
        ax.imshow(shade, extent=ex, origin="upper")
        ax.imshow(np.where(masks["eligible"], 1, np.nan), extent=ex, origin="upper",
                  cmap="Oranges", alpha=.22, vmin=0, vmax=1)
        ax.imshow(np.where(masks[key], 1, np.nan), extent=ex, origin="upper",
                  cmap="Blues", alpha=.72, vmin=0, vmax=1)
        if masks["production_fixed_legacy"].any():
            ax.contour(masks["production_fixed_legacy"].astype(float), levels=[.5],
                       colors="#CC79A7", linewidths=1.0, extent=ex, origin="upper")
        if masks["production_fixed_corrected"].any():
            ax.contour(masks["production_fixed_corrected"].astype(float), levels=[.5],
                       colors="#009E73", linewidths=1.0, extent=ex, origin="upper")
        area = report["areas_ha"][key]
        frac = 100 * report["fractions_of_eligible"][key]
        ax.set_title(f"({chr(97+i)}) {title}: {area:.1f} ha ({frac:.1f}%)",
                     loc="left", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Marsh-migration connectivity definitions", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", required=True); ap.add_argument("--nlcd", required=True)
    ap.add_argument("--wetlands", required=True); ap.add_argument("--buildings")
    ap.add_argument("--mhw", type=float, required=True)
    ap.add_argument("--mlw", type=float, required=True)
    ap.add_argument("--slr", type=float, required=True)
    ap.add_argument("--cell-m", type=float, required=True)
    ap.add_argument("--ref-point", nargs=2, type=float, default=(-81.0903, 31.9522))
    ap.add_argument("--radius-km", type=float, default=3.0)
    ap.add_argument("--reference-radius-km", type=float, default=3.0)
    ap.add_argument("--area-frac", type=float, default=.15)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--proximity-m", type=float, default=300.0)
    ap.add_argument("--max-depth", help="LISFLOOD .max on the DEM grid")
    ap.add_argument("--snapshots", help="quoted glob for ordered LISFLOOD .wd grids")
    ap.add_argument("--dry-threshold", type=float, default=.05)
    ap.add_argument("--min-snapshot-fraction", type=float, default=.10)
    ap.add_argument("--developed-barrier", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-npz", help="optional masks for overlap tests and scenario staging")
    ap.add_argument("--out-json", required=True); ap.add_argument("--out-fig", required=True)
    a = ap.parse_args()
    dem, ex, masks, report = audit(a)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(report, indent=2) + "\n")
    if a.out_npz:
        Path(a.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out_npz, **{k: v.astype(np.uint8) for k, v in masks.items()})
    plot(dem, ex, masks, report, a.out_fig)
    print(f"{'definition':32s} {'area ha':>10s} {'eligible %':>12s}")
    for key, area in report["areas_ha"].items():
        frac = 1.0 if key == "eligible" else report["fractions_of_eligible"][key]
        print(f"{key:32s} {area:10.2f} {100*frac:12.1f}")
    print(f"wrote {a.out_json}\nwrote {a.out_fig}")
    if a.out_npz:
        print(f"wrote {a.out_npz}")


if __name__ == "__main__":
    main()
