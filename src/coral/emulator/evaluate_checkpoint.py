"""Evaluate a trained emulator checkpoint against a completed physics ensemble.

This command does not fit or update the model. It applies the normalization stored in the
checkpoint, computes per-member RMSE and CSI, and writes a report that can be passed to
``coral.analysis.emulator_vs_physics``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.10)
    a = ap.parse_args()

    from .dataset import FloodDataset, build_manifest, missing_runs, set_grid_cache
    from .inference import load_model
    from .train import evaluate

    planned = json.loads(Path(a.manifest).read_text())
    if not isinstance(planned, list):
        raise ValueError(f"{a.manifest} must contain a JSON list")
    samples = build_manifest(planned, skip_missing=True)
    missing = missing_runs(planned)
    if not samples:
        raise RuntimeError("no completed members with valid .max files")

    set_grid_cache(a.cache)
    model, stats, device = load_model(a.ckpt)
    dataset = FloodDataset(samples, stats=stats, lazy=True)
    rows = evaluate(model, dataset, device, thresh=a.threshold, workers=a.workers)
    rmse = np.asarray([r["rmse_m"] for r in rows], dtype=float)
    csi = np.asarray([r["csi"] for r in rows], dtype=float)
    summary = {
        "n_planned": len(planned),
        "n_evaluated": len(rows),
        "n_missing": len(missing),
        "mean_rmse_m": float(np.nanmean(rmse)),
        "median_rmse_m": float(np.nanmedian(rmse)),
        "p90_rmse_m": float(np.nanpercentile(rmse, 90)),
        "max_rmse_m": float(np.nanmax(rmse)),
        "mean_csi": float(np.nanmean(csi)),
    }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(a.ckpt).resolve()),
        "manifest": str(Path(a.manifest).resolve()),
        "split": "external completed ensemble; no fitting",
        "threshold_m": a.threshold,
        "summary": summary,
        "holdout_metrics": rows,
        "missing": [{"name": n, "reason": why} for n, why in missing],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
