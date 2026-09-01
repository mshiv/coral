#!/usr/bin/env python
"""Controlled U-Net objective ablation: absolute depth versus absolute + paired delta."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coral.emulator import train as train_mod
from coral.emulator.dataset import FloodDataset, set_grid_cache
from coral.emulator.paired_loss import PairedFloodDataset, difference_metrics, train_paired
from train_ensemble import coverage, kind_of, load_runs, split


def means(rows):
    keys = sorted({key for row in rows for key in row if key not in {"name", "forcing_key"}})
    result = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        result["mean_" + key] = float(np.nanmean(values)) if np.isfinite(values).any() else None
    return result


def by_kind(rows):
    groups = {}
    for row in rows:
        groups.setdefault(kind_of(row["name"]), []).append(row)
    return {kind: means(members) for kind, members in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--objective", choices=["absolute", "paired"], required=True)
    parser.add_argument("--holdout-per-kind", type=float, default=0.2)
    parser.add_argument("--delta-lambda", type=float, default=1.0)
    parser.add_argument("--delta-train-threshold", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cell-m", type=float, default=4.0)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    set_grid_cache(args.cache_dir)
    runs = load_runs([args.ensemble])
    samples = coverage(runs, "A/B pool")
    train_samples, val_samples, split_text = split(
        samples, holdout_per_kind=args.holdout_per_kind, seed=args.seed)
    baselines = [s for s in samples if kind_of(s.name) == "base"]
    print(split_text)
    print(f"objective={args.objective}; train={len(train_samples)}; val={len(val_samples)}; baselines={len(baselines)}")

    train_ds = FloodDataset(train_samples, lazy=True, seed=args.seed)
    val_ds = FloodDataset(val_samples, stats=train_ds.stats, lazy=True)
    paired_train = PairedFloodDataset(train_samples, stats=train_ds.stats, lazy=True,
                                      seed=args.seed, baseline_pool=baselines)
    paired_val = PairedFloodDataset(val_samples, stats=train_ds.stats, lazy=True,
                                    seed=args.seed, baseline_pool=baselines)
    history = []
    if args.objective == "absolute":
        model = train_mod.train(train_ds, val=val_ds, epochs=args.epochs,
                                patience=args.patience, workers=args.workers,
                                ckpt=args.ckpt, history=history, seed=args.seed)
    else:
        model = train_paired(paired_train, val=paired_val,
                             delta_lambda=args.delta_lambda,
                             delta_train_threshold=args.delta_train_threshold,
                             epochs=args.epochs, patience=args.patience,
                             workers=args.workers, ckpt=args.ckpt,
                             history=history, seed=args.seed)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    absolute_rows = train_mod.evaluate(model, val_ds, device, workers=args.workers,
                                       cell_m=args.cell_m)
    delta_rows = difference_metrics(model, paired_val, device, workers=args.workers,
                                    cell_m=args.cell_m)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": "Adding a paired response loss reduces held-out delta-h error without degrading absolute flood-field skill.",
        "objective": args.objective,
        "controlled_variables": ["ensemble", "split", "seed", "architecture", "optimizer", "epochs", "early-stopping metric"],
        "primary_metric": "mean_delta_mae_response_0p05_m",
        "guardrails": ["absolute RMSE", "CSI", "flooded-area error", "flood-volume error"],
        "ensemble": args.ensemble,
        "split": split_text,
        "seed": args.seed,
        "n_train": len(train_samples),
        "n_validation": len(val_samples),
        "delta_lambda": args.delta_lambda if args.objective == "paired" else 0.0,
        "delta_train_threshold_m": args.delta_train_threshold,
        "checkpoint": args.ckpt,
        "history": history,
        "absolute_summary": means(absolute_rows),
        "delta_summary": means(delta_rows),
        "delta_by_kind": by_kind(delta_rows),
        "holdout_absolute_metrics": absolute_rows,
        "holdout_delta_metrics": delta_rows,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"absolute": report["absolute_summary"],
                      "delta": report["delta_summary"]}, indent=2))
    print("report ->", args.report)


if __name__ == "__main__":
    main()
