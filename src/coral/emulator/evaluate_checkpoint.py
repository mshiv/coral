"""Evaluate a trained emulator checkpoint against a completed physics ensemble.

This command does not fit or update the model. It applies the normalization stored in the
checkpoint, computes per-member RMSE and CSI, and writes a report that can be passed to
``coral.analysis.emulator_vs_physics``.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--selection-report", default=None,
                    help="evaluate only member names listed in an existing split report")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-csv", default=None,
                    help="optional flat per-member metrics table")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--cell-m", type=float, default=None,
                    help="grid spacing in metres; enables absolute area and volume metrics")
    a = ap.parse_args()

    from .dataset import FloodDataset, build_manifest, missing_runs, set_grid_cache
    from .inference import load_model
    from .train import evaluate

    planned = json.loads(Path(a.manifest).read_text())
    if not isinstance(planned, list):
        raise ValueError(f"{a.manifest} must contain a JSON list")
    if a.selection_report:
        prior = json.loads(Path(a.selection_report).read_text())
        blocks = []
        if isinstance(prior, dict):
            blocks.extend(prior.get(k) for k in ("holdout_metrics", "val_members", "held_out")
                          if isinstance(prior.get(k), list))
            for run in prior.get("runs", []):
                if isinstance(run, dict):
                    blocks.extend(run.get(k) for k in
                                  ("holdout_metrics", "val_members", "held_out")
                                  if isinstance(run.get(k), list))
        names = {str(r["name"]) for block in blocks for r in block
                 if isinstance(r, dict) and r.get("name")}
        if not names:
            raise ValueError(f"{a.selection_report} contains no per-member holdout names")
        planned = [r for r in planned if str(r.get("name")) in names]
        missing_names = names - {str(r.get("name")) for r in planned}
        if missing_names:
            raise ValueError(f"{len(missing_names)} selected names are absent from the manifest; "
                             f"first: {sorted(missing_names)[:5]}")
        print(f"selected {len(planned)} members from {a.selection_report}")
    samples = build_manifest(planned, skip_missing=True)
    missing = missing_runs(planned)
    if not samples:
        raise RuntimeError("no completed members with valid .max files")

    set_grid_cache(a.cache)
    model, stats, device = load_model(a.ckpt)
    dataset = FloodDataset(samples, stats=stats, lazy=True)
    rows = evaluate(model, dataset, device, thresh=a.threshold, workers=a.workers,
                    cell_m=a.cell_m)
    rmse = np.asarray([r["rmse_m"] for r in rows], dtype=float)
    csi = np.asarray([r["csi"] for r in rows], dtype=float)
    def mean(key):
        return float(np.nanmean([r[key] for r in rows]))
    summary = {
        "n_planned": len(planned),
        "n_evaluated": len(rows),
        "n_missing": len(missing),
        "mean_rmse_m": float(np.nanmean(rmse)),
        "median_rmse_m": float(np.nanmedian(rmse)),
        "p90_rmse_m": float(np.nanpercentile(rmse, 90)),
        "max_rmse_m": float(np.nanmax(rmse)),
        "mean_csi": float(np.nanmean(csi)),
        "mean_mae_m": mean("mae_m"),
        "mean_bias_m": mean("bias_m"),
        "mean_nrmse_sd": mean("nrmse_sd"),
        "mean_nrmse_p95_p05": mean("nrmse_p95_p05"),
        "mean_probability_of_detection": mean("probability_of_detection"),
        "mean_false_alarm_ratio": mean("false_alarm_ratio"),
        "mean_flooded_area_error_pct": mean("flooded_area_error_pct"),
        "mean_flood_volume_error_pct": mean("flood_volume_error_pct"),
    }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(a.ckpt).resolve()),
        "manifest": str(Path(a.manifest).resolve()),
        "selection_report": str(Path(a.selection_report).resolve()) if a.selection_report else None,
        "split": "external completed ensemble; no fitting",
        "threshold_m": a.threshold,
        "cell_m": a.cell_m,
        "summary": summary,
        "holdout_metrics": rows,
        "missing": [{"name": n, "reason": why} for n, why in missing],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    if a.out_csv:
        csv_out = Path(a.out_csv)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"per-member metrics -> {csv_out}")
    print(json.dumps(summary, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
