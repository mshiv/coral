"""Separate the compound effect from the nonlinear residual, and map driver dominance.

Two quantities, from the same four factorial runs, answering different questions:

    C = Y_full - max(Y_coastal, Y_inland)
    N = Y_full - Y_coastal - Y_inland + Y_baseline

C is how much the coupled run exceeds the best single-driver run. It is the quantity a
planner cares about, because running the drivers separately and taking a cellwise maximum
is what a separate-hazard-map workflow does. N is departure from linear superposition.

They are not interchangeable. A domain-mean N near zero is compatible with a large C, and
is also compatible with N being strongly positive in one landform and negative in another
and cancelling in the average. Reporting only the mean N understates the case for a coupled
model, which is why this module reports C, reports N stratified by landform, and never
reduces either to one number.

Usage:
  python -m coral.analysis.compound_attribution effect \
      --baseline <no-forcing .max> --coastal <surge+tide .max> \
      --inland <rain-only .max> --full <all-drivers .max> \
      --dem <DEM .asc> --waterline 1.114

  python -m coral.analysis.compound_attribution dominance \
      --coastal ... --inland ... --full ... --dem ... --waterline 1.114
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NODATA = -9999.0


def read_asc(path):
    """(array, header dict). Values at or below NODATA become NaN."""
    hdr, n = {}, 0
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and not parts[0][:1].isdigit() and parts[0][:1] != "-":
                hdr[parts[0].lower()] = float(parts[1])
                n += 1
            else:
                break
    a = np.loadtxt(path, skiprows=n)
    return np.where(a > NODATA + 1.0, a, np.nan), hdr


def _stack(paths):
    """Read several grids and check they share a shape. Depth grids, NaN treated as dry."""
    out, shape = [], None
    for k, p in paths.items():
        a, _ = read_asc(p)
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            raise SystemExit(f"{k}: shape {a.shape} does not match {shape}. "
                             "All four runs must be on the same grid.")
        # A dry cell has no depth, which is zero depth, not missing data. Leaving it NaN
        # would drop it from every comparison and quietly shrink the denominator -- the
        # same fault that made the factorial's two algebraic routes disagree.
        out.append(np.nan_to_num(a, nan=0.0))
    return dict(zip(paths, out)), shape


def land_mask(dem_path, waterline, shape):
    dem, _ = read_asc(dem_path)
    if dem.shape != shape:
        raise SystemExit(f"DEM shape {dem.shape} does not match the runs {shape}")
    return np.isfinite(dem) & (dem > waterline), dem


def effect(args):
    g, shape = _stack({"baseline": args.baseline, "coastal": args.coastal,
                       "inland": args.inland, "full": args.full})
    land, dem = land_mask(args.dem, args.waterline, shape)

    best_single = np.maximum(g["coastal"], g["inland"])
    C = g["full"] - best_single
    N = g["full"] - g["coastal"] - g["inland"] + g["baseline"]

    # One common footprint for every statistic: cells wet in ANY run, with dry counted as
    # zero depth. Using each run's own wet cells breaks the factorial algebra and makes a
    # rainier run look shallower, because it wets more land and dilutes the mean.
    wet_any = land & (np.maximum(g["full"], best_single) > args.threshold)
    n = int(wet_any.sum())
    if n == 0:
        raise SystemExit("no wet land cells; check --waterline and --threshold")

    def frac_over(a, t):
        return float((np.abs(a[wet_any]) >= t).sum()) / n

    rep = {
        "wet_land_cells": n,
        "cell_area_m2": args.cell_m ** 2,
        "wet_land_km2": n * args.cell_m ** 2 / 1e6,
        "compound_effect_C": {
            "mean_m": float(C[wet_any].mean()),
            "median_m": float(np.median(C[wet_any])),
            "p90_m": float(np.percentile(C[wet_any], 90)),
            "max_m": float(C[wet_any].max()),
            # The Gori-comparable statistic: how much floodplain a separate-driver maximum
            # underestimates by at least this much.
            "frac_underestimated_ge_0.1m": frac_over(np.maximum(C, 0), 0.1),
            "frac_underestimated_ge_0.2m": frac_over(np.maximum(C, 0), 0.2),
            "frac_underestimated_ge_0.5m": frac_over(np.maximum(C, 0), 0.5),
        },
        "nonlinear_residual_N": {
            "mean_m": float(N[wet_any].mean()),
            "median_m": float(np.median(N[wet_any])),
            "p10_m": float(np.percentile(N[wet_any], 10)),
            "p90_m": float(np.percentile(N[wet_any], 90)),
            "frac_positive": float((N[wet_any] > 0).sum()) / n,
            "frac_abs_ge_0.1m": frac_over(N, 0.1),
        },
    }

    # N stratified. A near-zero mean can hide opposite signs in channel and floodplain,
    # which is the reported behaviour elsewhere: surge occupies channel storage before
    # runoff arrives and pushes water onto adjacent lowland.
    bands = {
        "low (waterline to +0.5 m)": land & (dem <= args.waterline + 0.5),
        "mid (+0.5 to +1.5 m)": land & (dem > args.waterline + 0.5) & (dem <= args.waterline + 1.5),
        "upland (> +1.5 m)": land & (dem > args.waterline + 1.5),
    }
    rep["N_by_elevation_band"] = {}
    for k, m in bands.items():
        mm = m & wet_any
        if mm.sum() < 50:
            continue
        rep["N_by_elevation_band"][k] = {
            "cells": int(mm.sum()),
            "N_mean_m": float(N[mm].mean()),
            "C_mean_m": float(C[mm].mean()),
        }

    print(json.dumps(rep, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.out_json}")
    if args.out_fig:
        _plot_effect(C, N, wet_any, args)
    return rep


def _plot_effect(C, N, wet, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a, arr, ttl, cm in (
            (ax[0], np.where(wet, C, np.nan), "compound effect C = full - max(single)", "magma_r"),
            (ax[1], np.where(wet, N, np.nan), "nonlinear residual N", "RdBu_r")):
        v = np.nanpercentile(np.abs(arr), 99) or 1.0
        kw = {"vmin": -v, "vmax": v} if cm == "RdBu_r" else {"vmin": 0, "vmax": v}
        im = a.imshow(arr, cmap=cm, interpolation="none", **kw)
        a.set_title(ttl, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, label="m")

    ax[2].hist(C[wet], bins=80, alpha=0.75, label="C", color="#7a3b8f")
    ax[2].hist(N[wet], bins=80, alpha=0.6, label="N", color="#c26a3d")
    ax[2].axvline(0, color="0.3", lw=0.8)
    ax[2].set_xlabel("m"); ax[2].set_ylabel("wet land cells"); ax[2].legend()
    ax[2].set_title("C is not N", fontsize=11)
    fig.suptitle("Compound effect against nonlinear residual", fontsize=13)
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"wrote {args.out_fig}")


def dominance(args):
    """Per-cell driver attribution: which driver accounts for the depth here.

    Shares are taken from the single-driver runs rather than from the full run, because
    the full run cannot be decomposed cellwise without assuming additivity -- which is the
    thing under test. A cell is called transitional when neither driver holds a clear
    majority, and that zone is the interesting one.
    """
    g, shape = _stack({"coastal": args.coastal, "inland": args.inland, "full": args.full})
    land, dem = land_mask(args.dem, args.waterline, shape)

    c, i = g["coastal"], g["inland"]
    tot = c + i
    wet = land & (np.maximum(g["full"], np.maximum(c, i)) > args.threshold)
    share = np.where(tot > 1e-9, c / np.maximum(tot, 1e-9), np.nan)

    lab = np.full(shape, np.nan)
    lab[wet & (share >= args.split)] = 2.0                       # coastal-dominated
    lab[wet & (share <= 1 - args.split)] = 0.0                   # rain-dominated
    lab[wet & (share > 1 - args.split) & (share < args.split)] = 1.0   # transitional

    n = int(wet.sum())
    a_km2 = args.cell_m ** 2 / 1e6
    rep = {
        "wet_land_cells": n,
        "wet_land_km2": n * a_km2,
        "split_threshold": args.split,
        "rain_dominated_km2": float((lab == 0).sum()) * a_km2,
        "transitional_km2": float((lab == 1).sum()) * a_km2,
        "coastal_dominated_km2": float((lab == 2).sum()) * a_km2,
        "rain_dominated_frac": float((lab == 0).sum()) / max(n, 1),
        "transitional_frac": float((lab == 1).sum()) / max(n, 1),
        "coastal_dominated_frac": float((lab == 2).sum()) / max(n, 1),
    }
    print(json.dumps(rep, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.out_json}")

    if args.out_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        cols = ["#2c7fb8", "#d9c15b", "#a63f22"]        # rain, transitional, coastal
        fig, ax = plt.subplots(figsize=(8.5, 8))
        ax.imshow(lab, cmap=ListedColormap(cols), vmin=-0.5, vmax=2.5, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(handles=[
            Patch(color=cols[0], label=f"rain-dominated  {rep['rain_dominated_frac']*100:.0f}%"),
            Patch(color=cols[1], label=f"transitional  {rep['transitional_frac']*100:.0f}%"),
            Patch(color=cols[2], label=f"coastal-dominated  {rep['coastal_dominated_frac']*100:.0f}%")],
            loc="lower left", frameon=False)
        ax.set_title("Which driver accounts for the flood, cell by cell", fontsize=12)
        fig.tight_layout()
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_fig, dpi=150)
        print(f"wrote {args.out_fig}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_baseline):
        if need_baseline:
            p.add_argument("--baseline", required=True, help="no-driver (or tide-only) .max")
        p.add_argument("--coastal", required=True, help="surge+tide, no rain .max")
        p.add_argument("--inland", required=True, help="rain only .max")
        p.add_argument("--full", required=True, help="all drivers .max")
        p.add_argument("--dem", required=True)
        p.add_argument("--waterline", type=float, required=True,
                       help="land/sea split, datums.mhw (1.114 for the 2016 record)")
        p.add_argument("--threshold", type=float, default=0.10, help="wet depth, m")
        p.add_argument("--cell-m", type=float, default=30.0, help="grid resolution, m")
        p.add_argument("--out-json", default=None)
        p.add_argument("--out-fig", default=None)

    e = sub.add_parser("effect", help="C and N, with N stratified by elevation band")
    common(e, True)
    e.set_defaults(func=effect)

    d = sub.add_parser("dominance", help="per-cell driver attribution map")
    common(d, False)
    d.add_argument("--split", type=float, default=0.7,
                   help="share above which one driver is called dominant")
    d.set_defaults(func=dominance)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
