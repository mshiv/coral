"""Peak-depth redistribution for a matched control/intervention pair.

Reports relief and adverse amplification separately, including their distance from the actual
edited footprint.  A near-zero domain mean is therefore not mistaken for "no effect".
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .physics_ab import _read_grid


def first(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one maximum raster for {pattern}; found {len(hits)}")
    return hits[0]


def read(path):
    a = np.asarray(_read_grid(path)[0], dtype=np.float32)
    return np.where(np.isfinite(a) & (a > -9000), a, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control-results", required=True)
    ap.add_argument("--intervention-results", required=True)
    ap.add_argument("--control-dem", required=True)
    ap.add_argument("--fields", nargs="+", required=True,
                    metavar="CONTROL_GRID:INTERVENTION_GRID")
    ap.add_argument("--waterline", type=float, required=True)
    ap.add_argument("--cell-m", type=float, required=True)
    ap.add_argument("--tol-m", type=float, default=0.01)
    ap.add_argument("--field-tol", type=float, default=1e-3,
                    help="minimum edited-grid change defining the intervention footprint; "
                         "must exceed ASCII rounding noise")
    ap.add_argument("--wet-m", type=float, default=0.05)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument('--publication', action='store_true')
    ap.add_argument('--error-limit', type=float, help='Explicit symmetric depth-change colour limit (m)')
    ap.add_argument('--export-npz', help='Export clean paired fields for a mechanism composite')
    a = ap.parse_args()

    c = np.nan_to_num(read(first(str(Path(a.control_results) / "*.max"))), nan=0.0)
    x = np.nan_to_num(read(first(str(Path(a.intervention_results) / "*.max"))), nan=0.0)
    dem = read(a.control_dem)
    if c.shape != x.shape or c.shape != dem.shape:
        raise SystemExit("control, intervention and DEM grids do not align")
    footprint = np.zeros(c.shape, bool)
    for spec in a.fields:
        if ":" not in spec:
            raise SystemExit(f"--fields requires CONTROL:INTERVENTION, got {spec}")
        p, q = spec.split(":", 1)
        g0, g1 = read(p), read(q)
        if g0.shape != c.shape or g1.shape != c.shape:
            raise SystemExit(f"field pair does not align: {spec}")
        footprint |= np.nan_to_num(np.abs(g1 - g0), nan=0.0) > a.field_tol
    if not footprint.any():
        raise SystemExit("no edited cells found in --fields")

    from scipy.ndimage import distance_transform_edt
    dist_km = distance_transform_edt(~footprint) * a.cell_m / 1000.0
    land = np.isfinite(dem) & (dem > a.waterline)
    wet = land & ((c > a.wet_m) | (x > a.wet_m))
    delta = np.where(wet, x - c, np.nan)
    better = wet & (delta < -a.tol_m)
    worse = wet & (delta > a.tol_m)
    area = a.cell_m ** 2

    def threshold_metrics(tol):
        good = wet & (delta < -tol)
        bad = wet & (delta > tol)
        return {"threshold_m": float(tol),
                "improved_cells": int(good.sum()), "worsened_cells": int(bad.sum()),
                "improved_fraction": float(good.sum() / max(wet.sum(), 1)),
                "worsened_fraction": float(bad.sum() / max(wet.sum(), 1)),
                "benefit_m3": float((-delta[good]).sum() * area),
                "adverse_m3": float(delta[bad].sum() * area)}

    sensitivities = [threshold_metrics(t) for t in (.005, .01, .02, .05, .10)]
    control_wet = land & (c > a.wet_m)
    control_depth = c[control_wet]
    baseline_stats = {"wet_threshold_m": float(a.wet_m),
                      "wet_cells": int(control_wet.sum())}
    if control_depth.size:
        baseline_stats.update({"median_m": float(np.median(control_depth)),
                               "p90_m": float(np.percentile(control_depth, 90)),
                               "p95_m": float(np.percentile(control_depth, 95)),
                               "max_m": float(control_depth.max())})
    bins = np.asarray([0, .1, .25, .5, 1, 2, 5, np.inf])
    radial = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = wet & (dist_km >= lo) & (dist_km < hi)
        radial.append({"lo_km": float(lo), "hi_km": None if np.isinf(hi) else float(hi),
                       "wet_cells": int(m.sum()),
                       "benefit_m3": float((-delta[m & better]).sum() * area),
                       "adverse_m3": float(delta[m & worse].sum() * area)})
    report = {"wet_cells": int(wet.sum()), "footprint_cells": int(footprint.sum()),
              "footprint_m2": float(footprint.sum() * area),
              "improved_cells": int(better.sum()), "worsened_cells": int(worse.sum()),
              "improved_fraction": float(better.sum() / max(wet.sum(), 1)),
              "worsened_fraction": float(worse.sum() / max(wet.sum(), 1)),
              "benefit_m3": float((-delta[better]).sum() * area),
              "adverse_m3": float(delta[worse].sum() * area),
              "depth_tolerance_m": float(a.tol_m),
              "field_tolerance": float(a.field_tol),
              "baseline_control_depth": baseline_stats,
              "threshold_sensitivity": sensitivities,
              "max_reduction_m": float(-np.nanmin(delta)),
              "max_increase_m": float(np.nanmax(delta)), "distance_bins": radial}
    report['provenance'] = dict(control_results=str(Path(a.control_results).resolve()),
        intervention_results=str(Path(a.intervention_results).resolve()),
        control_dem=str(Path(a.control_dem).resolve()), fields=a.fields,
        waterline_m=a.waterline,nominal_cell_m=a.cell_m,
        mask='baseline land above waterline, wet > wet_m in either maximum raster',
        interpretation='area integral of differences of cellwise maxima; not synchronous volume')
    outj = Path(a.out_json); outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    show = wet | footprint
    yy, xx = np.where(show)
    sl = np.s_[max(0, yy.min()-10):min(c.shape[0], yy.max()+11),
               max(0, xx.min()-10):min(c.shape[1], xx.max()+11)]
    vmax = np.nanpercentile(np.r_[c[wet], x[wet]], 99) if wet.any() else 1
    for aa, z, title in zip(ax[:2], (c, x), ("Control peak depth", "Intervention peak depth")):
        im = aa.imshow(np.where(land, z, np.nan)[sl], cmap="Blues", vmin=0, vmax=vmax,
                       interpolation='nearest')
        aa.set_title(title); aa.set_axis_off(); fig.colorbar(im, ax=aa, shrink=.75, label="m")
    material = wet & (np.abs(delta) > a.tol_m)
    v = np.nanpercentile(np.abs(delta[material]), 99) if material.any() else a.tol_m
    if a.error_limit is not None:
        if a.error_limit <= 0:
            raise SystemExit('--error-limit must be positive')
        v = a.error_limit
    display = np.where(material, delta, np.nan)
    if not a.publication:
        ax[2].imshow(np.where(footprint[sl], 1.0, np.nan), cmap="Greys",
                     vmin=0, vmax=1, alpha=.16)
    im = ax[2].imshow(display[sl], cmap="RdBu_r", vmin=-v, vmax=v, interpolation='nearest')
    ax[2].set_title(f"Material depth change (|Δh| ≥ {a.tol_m:g} m)\n"
                    "faint gray = edited input footprint"); ax[2].set_axis_off()
    fig.colorbar(im, ax=ax[2], shrink=.75, label="Intervention − control (m)", extend='both')
    labels = [f"{r['lo_km']:g}–{r['hi_km']:g}" if r['hi_km'] is not None
              else f">{r['lo_km']:g}" for r in radial]
    pos = np.arange(len(radial))
    ax[3].bar(pos, [r["benefit_m3"]/1e3 for r in radial], label="benefit", color="#2878b5")
    ax[3].bar(pos, [-r["adverse_m3"]/1e3 for r in radial], label="adverse", color="#c44e52")
    ax[3].axhline(0, color="black", lw=.7); ax[3].set_xticks(pos, labels, rotation=45, ha="right")
    ax[3].set_xlabel("distance from edited footprint (km)"); ax[3].set_ylabel("volume (10³ m³)")
    ax[3].set_title("Redistribution with distance"); ax[3].legend(frameon=False)
    fig.tight_layout()
    if a.publication:
        from coral.viz.publication_style import caption_first
        ax[3].set_ylabel('Peak-depth integral (10³ m³)')
        caption_first(fig, ax)
    outf = Path(a.out_fig); outf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outf, dpi=180, bbox_inches="tight")
    if a.publication:
        fig.savefig(outf.with_suffix('.pdf'), bbox_inches='tight')
    report['display'] = dict(error_limit_m=float(v), display_threshold_m=a.tol_m,
        clipped_fraction_of_material_cells=float(np.mean(np.abs(delta[material]) > v)) if material.any() else 0.,
        raster_interpolation='nearest', edit_overlay=not a.publication)
    outj.write_text(json.dumps(report, indent=2)+'\n')
    if a.export_npz:
        Path(a.export_npz).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.export_npz,control=c,intervention=x,delta=delta,
            land=land,wet=wet,footprint=footprint,metadata_json=np.asarray(json.dumps(report)))
    print(json.dumps(report, indent=2)); print(f"wrote {outj}\nwrote {outf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
