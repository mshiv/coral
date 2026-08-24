"""Emulator error against the size of the thing it is predicting.

A held-out RMSE of about a centimetre is excellent against a metre of water and poor against a
three-millimetre signal, and the intervention effects in this ensemble span both. Marsh
restoration moves peak depth by roughly 0.003 m; a floodwall moves it by 1.7 m locally. Reporting
one error figure across all of them says nothing about whether the network has learned the
intervention or has learned to reproduce the baseline.

So this reports three things per member and per kind:

  RMSE            what the training report already gives
  NRMSE           RMSE divided by the spread of the target, so splits validating on shallow and
                  deep water can be compared. Splits differ in depth distribution -- land depth
                  p90 is 0.93 m at the zero offset and 3.21 m at 2.043 m -- so part of any
                  difference between splits is the target, not the model.
  skill           RMSE against a benchmark that ignores the intervention and returns the
                  no-intervention baseline at the same sea level. This is the same device the
                  chapter uses for high-water marks, where a constant at the observed mean scores
                  0.177 m against the model's 0.236 m and shows what the marks can actually
                  establish.

The benchmark needs no inference: predicting the baseline means the error IS the intervention
effect, which is already available from the physics. A skill score at or below zero means the
emulator would have done as well by ignoring the intervention, which for a scenario-comparison
tool is the failure that matters most.

Usage:
  python -m coral.analysis.emulator_skill --report <emulator .report.json> \
      --ens <ensemble dir> --dem <DEM .asc> --waterline 1.114
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NODATA = -9999.0


def rd(path):
    a = np.loadtxt(path, skiprows=6)
    return np.where(a > NODATA + 1.0, a, np.nan)


def find_max(run_dir):
    hits = sorted(glob.glob(str(Path(run_dir) / "results_*" / "*.max")))
    return hits[0] if hits else None


def kinds_of(e):
    return sorted({i["kind"] for i in (e.get("interventions") or [])})


def slr_of(name):
    return name.split("_")[0]


def per_member(report):
    for r in report.get("runs") or []:
        for k in ("holdout_metrics", "val_members", "held_out"):
            if isinstance(r.get(k), list) and r[k]:
                return r[k]
    raise SystemExit("no per-member block in the report")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--ens", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--waterline", type=float, required=True)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--out-csv", default=None)
    a = ap.parse_args()

    rep = json.load(open(a.report))
    held = {r["name"]: r for r in per_member(rep)}
    manifest = {e["name"]: e for e in json.load(open(Path(a.ens) / "manifest.json"))}
    dem = rd(a.dem)
    land = np.isfinite(dem) & (dem > a.waterline)

    base = {}
    for e in manifest.values():
        if not kinds_of(e):
            f = find_max(e["run_dir"])
            if f:
                base[slr_of(e["name"])] = np.nan_to_num(rd(f), nan=0.0)

    rows = []
    for name, rec in held.items():
        e = manifest.get(name)
        lvl = slr_of(name)
        if e is None or lvl not in base:
            continue
        f = find_max(e["run_dir"])
        if f is None:
            continue
        truth = np.nan_to_num(rd(f), nan=0.0)
        b = base[lvl]
        wet = land & ((truth > a.threshold) | (b > a.threshold))
        if wet.sum() < 100:
            continue
        y = truth[wet]
        # The benchmark: return the baseline and ignore the intervention. Its error is the
        # intervention effect itself.
        bench = float(np.sqrt(np.mean((b[wet] - y) ** 2)))
        rmse = float(rec.get("rmse_m", np.nan))
        spread = float(y.max() - y.min()) or np.nan
        rows.append({"name": name, "kind": "+".join(kinds_of(e)) or "base", "slr": lvl,
                     "rmse_m": rmse, "baseline_rmse_m": bench,
                     "nrmse_range": rmse / spread if spread else np.nan,
                     "nrmse_sd": rmse / (float(y.std()) or np.nan),
                     "skill": 1.0 - (rmse / bench) if bench > 0 else np.nan,
                     "target_p90_m": float(np.percentile(y, 90))})

    if not rows:
        raise SystemExit("no scored members")
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)

    print(f"{len(rows)} held-out members scored\n")
    print(f"{'kind':20s} {'n':>4} {'RMSE m':>9} {'baseline m':>11} {'skill':>8} "
          f"{'NRMSE/sd':>9} {'target p90':>11}")
    for k in sorted(by):
        v = by[k]
        md = lambda f: float(np.nanmedian([x[f] for x in v]))
        flag = "   <-- baseline is better" if md("skill") <= 0 else ""
        print(f"{k:20s} {len(v):4d} {md('rmse_m'):9.4f} {md('baseline_rmse_m'):11.4f} "
              f"{md('skill'):8.3f} {md('nrmse_sd'):9.4f} {md('target_p90_m'):11.3f}{flag}")

    allsk = np.nanmedian([r["skill"] for r in rows])
    print(f"\nmedian skill across all held-out members: {allsk:.3f}")
    print("Skill is 1 minus the ratio of emulator error to the error of a predictor that returns")
    print("the no-intervention baseline. Zero means the emulator does no better than ignoring the")
    print("intervention; one means it reproduces the parent exactly.")

    if a.out_csv:
        import csv
        p = Path(a.out_csv); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
