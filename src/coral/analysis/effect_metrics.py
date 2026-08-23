"""Intervention effect metrics: benefit, adverse redistribution, and rank stability.

Reporting an intervention by its mean depth change hides the thing a community would ask about
first. A wall that lowers water behind it and raises it on a neighbouring street has a mean near
zero, and so does a wall that does nothing. These are not the same outcome, and the metric has to
separate them.

Three things this reports that a mean does not:

  sign structure     what fraction of the wet footprint improved, was unchanged, or got worse.
                     A material-change tolerance is applied, because a millimetre is not an
                     effect and counting it as one makes every intervention look consequential.

  benefit vs cost    relief inside a target zone against amplification outside it, as volumes
                     rather than depths, plus their ratio. The ratio alone is misleading when the
                     denominator is small, so all three are reported together.

  rank stability     whether the ordering of interventions changes as sea level rises. A measure
                     that is best at today's level and third best at 2 m is a different planning
                     object from one that holds its place, and only a comparison across offsets
                     shows it.

Depth is the wrong metric for managed retreat, which lowers ground and therefore deepens water
where a structure used to be while removing the structure. Retreat is reported but flagged, and
its exposure change is the number that means something.

Usage:
  python -m coral.analysis.effect_metrics --ens <ensemble> --dem <DEM .asc> \
      --waterline 1.114 --cell-m 4 --focus-radius-km 2.0 --ref-point -81.0903 31.9522 \
      --out-csv reports/adapt/effect_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NODATA = -9999.0
THRESHOLDS = (0.05, 0.10, 0.30, 0.50, 1.00)
# Below this a difference is numerical or topographic noise, not an intervention effect.
TOL = 0.01


def rd(path):
    a = np.loadtxt(path, skiprows=6)
    return np.where(a > NODATA + 1.0, a, np.nan)


def header(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    return h


def find_max(run_dir):
    hits = sorted(glob.glob(str(Path(run_dir) / "results_*" / "*.max")))
    return hits[0] if hits else None


def kinds_of(e):
    return sorted({i["kind"] for i in (e.get("interventions") or [])})


def slr_of(name):
    return name.split("_")[0]


def target_zone(shape, hdr, ref_lon, ref_lat, radius_km):
    """Circle around the community. Benefit inside, adverse effect outside.

    Without a declared target the benefit/cost split is arbitrary, and 'the intervention helped'
    becomes a statement about wherever it happened to help.
    """
    ny, nx, cs = shape[0], shape[1], hdr["cellsize"]
    x = hdr["xllcorner"] + (np.arange(nx) + 0.5) * cs
    y = hdr["yllcorner"] + (ny - np.arange(ny) - 0.5) * cs
    X, Y = np.meshgrid(x, y)
    km_per_deg = 111.32
    d = np.hypot((X - ref_lon) * km_per_deg * np.cos(np.radians(ref_lat)),
                 (Y - ref_lat) * km_per_deg)
    return d <= radius_km


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--waterline", type=float, required=True)
    ap.add_argument("--cell-m", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=0.10, help="wet depth, m")
    ap.add_argument("--ref-point", type=float, nargs=2, default=None,
                    metavar=("LON", "LAT"))
    ap.add_argument("--focus-radius-km", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-csv", default="reports/adapt/effect_metrics.csv")
    a = ap.parse_args()

    manifest = json.load(open(Path(a.ens) / "manifest.json"))
    dem, hdr = rd(a.dem), header(a.dem)
    land = np.isfinite(dem) & (dem > a.waterline)
    cell = a.cell_m ** 2
    T = (target_zone(dem.shape, hdr, a.ref_point[0], a.ref_point[1], a.focus_radius_km)
         if a.ref_point else land)
    print(f"land {int(land.sum()):,} cells; target zone {int((T & land).sum()):,} cells")

    base = {}
    for e in manifest:
        if not kinds_of(e):
            f = find_max(e["run_dir"])
            if f:
                base[slr_of(e["name"])] = np.nan_to_num(rd(f), nan=0.0)
    print(f"baselines available at {len(base)} sea level(s): {sorted(base)}")
    if not base:
        raise SystemExit("no finished baselines; effects are undefined without them")

    import time
    rows, n, t0 = [], 0, time.time()
    todo = sum(1 for e in manifest if kinds_of(e))
    for e in manifest:
        ks = kinds_of(e)
        if not ks:
            continue
        lvl = slr_of(e["name"])
        if lvl not in base:
            continue
        f = find_max(e["run_dir"])
        if f is None:
            continue
        m = np.nan_to_num(rd(f), nan=0.0)
        b = base[lvl]
        d = np.where(land, m - b, np.nan)
        wet = land & ((m > a.threshold) | (b > a.threshold))
        if not wet.any():
            continue

        better = wet & (d < -TOL)
        worse = wet & (d > TOL)
        same = wet & (np.abs(d) <= TOL)
        # Volumes, not depths: a decimetre over a hectare and a metre over a square metre are
        # different outcomes and a depth statistic cannot tell them apart.
        B = float(np.clip(-d[T & wet], 0, None).sum()) * cell
        E = float(np.clip(d[~T & wet], 0, None).sum()) * cell
        dv = d[wet]
        r = {"name": e["name"], "kind": "+".join(ks), "slr": lvl,
             "slr_m": float((e.get("forcing") or {}).get("slr_m", 0.0)),
             "siting": e.get("siting"),
             "wet_cells": int(wet.sum()),
             "mean_m": float(np.nanmean(dv)),
             "p01_m": float(np.nanpercentile(dv, 1)),
             "p50_m": float(np.nanpercentile(dv, 50)),
             "p99_m": float(np.nanpercentile(dv, 99)),
             "min_m": float(np.nanmin(dv)), "max_m": float(np.nanmax(dv)),
             "frac_improved": float(better.sum()) / int(wet.sum()),
             "frac_unchanged": float(same.sum()) / int(wet.sum()),
             "frac_worsened": float(worse.sum()) / int(wet.sum()),
             "benefit_m3": B, "adverse_m3": E,
             "spillover_ratio": E / (B + 1e-9),
             "signed_volume_m3": float(np.nansum(dv)) * cell,
             "abs_volume_m3": float(np.nansum(np.abs(dv))) * cell}
        for t in THRESHOLDS:
            r[f"d_area_ge_{t:g}m_km2"] = (float((land & (m > t)).sum())
                                          - float((land & (b > t)).sum())) * cell / 1e6
        rows.append(r)
        n += 1
        # Progress, because this reads two ASCII grids per member across nearly two thousand
        # members and a silent multi-hour job is indistinguishable from a stuck one.
        if n % 50 == 0 or n == todo:
            el = time.time() - t0
            print(f"  {n}/{todo} members  {el/60:.1f} min elapsed  "
                  f"~{el/n*(todo-n)/60:.0f} min remaining", flush=True)
        if a.limit and n >= a.limit:
            break

    if not rows:
        raise SystemExit("no member/baseline pairs with results")
    out = Path(a.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} members)")

    # ---- per-kind summary, and whether the ordering survives sea-level rise
    by = defaultdict(list)
    # Sea levels ordered by their offset, not by their label. Sorting the labels as strings put
    # slrLow2050 (0.219 m) last and slrHigh2100 (2.043 m) third, so the rank-stability comparison
    # ran between the two smallest offsets and reported a reversal that was not the one asked for.
    lvl_m = {r["slr"]: r["slr_m"] for r in rows}
    for r in rows:
        by[(r["kind"], r["slr"])].append(r)
    print(f"\n{'kind':22s} {'slr':14s} {'n':>4} {'median benefit m3':>18} "
          f"{'median adverse m3':>18} {'%worse':>7}")
    for (k, s), v in sorted(by.items()):
        if "+" in k:
            continue
        print(f"{k:22s} {s:14s} {len(v):4d} "
              f"{np.median([x['benefit_m3'] for x in v]):18.0f} "
              f"{np.median([x['adverse_m3'] for x in v]):18.0f} "
              f"{100*np.median([x['frac_worsened'] for x in v]):7.2f}")

    print("\nintervention ranking by median benefit, per sea level "
          "(a reversal means the ordering is not stable):")
    levels = sorted({s for _, s in by}, key=lambda s: lvl_m.get(s, 0.0))
    order = {}
    for s in levels:
        ks = [(k, np.median([x["benefit_m3"] for x in by[(k, s)]]))
              for (k, ss) in by if ss == s and "+" not in k]
        order[s] = [k for k, _ in sorted(ks, key=lambda x: -x[1])]
        print(f"  {s:14s} ({lvl_m.get(s, 0):.3f} m)  {' > '.join(order[s])}")
    if len(levels) > 1:
        first, last = order[levels[0]], order[levels[-1]]
        moved = [k for k in first if k in last and first.index(k) != last.index(k)]
        print(f"\n{len(moved)} kind(s) change rank between {levels[0]} "
              f"({lvl_m.get(levels[0],0):.3f} m) and {levels[-1]} ({lvl_m.get(levels[-1],0):.3f} m)"
              f"{': ' + ', '.join(moved) if moved else ''}")
        print("\nbenefit against sea level, median m3 per kind:")
        kinds = sorted({k for k, _ in by if "+" not in k})
        print(f"  {'kind':20s} " + " ".join(f"{lvl_m.get(s,0):>9.2f}" for s in levels))
        for k in kinds:
            vals = [np.median([x["benefit_m3"] for x in by[(k, s)]]) if (k, s) in by else float("nan")
                    for s in levels]
            print(f"  {k:20s} " + " ".join(f"{v:9.0f}" for v in vals))

    print("\nNOTE: managed retreat lowers ground toward the surrounding grade, so it deepens water "
          "\nwhere a structure stood while removing the structure. Its depth-based benefit is not "
          "\ncomparable to the others; report exposure removed instead.")


if __name__ == "__main__":
    main()
