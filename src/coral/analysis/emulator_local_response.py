"""Evaluate emulator accuracy near interventions and for paired intervention responses.

This is an evaluation-only command: it never updates model weights.  It preserves the
member names in an existing training report, predicts those members and their corresponding
same-sea-level baselines, and writes one row per member and distance buffer.

The two central diagnostics are:

* local field error: emulator minus LISFLOOD-FP within a buffer around the cells edited by
  the intervention; and
* paired response error: (predicted intervention - predicted baseline) minus
  (parent intervention - parent baseline) on the same fixed cells.

Run on the HPC because the full ensemble input grids and LISFLOOD-FP ``.max`` rasters are
required.  The small chapter figure exports containing only best/median/worst members are
not sufficient for a population summary.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


STATIC_CHANNELS = 4  # DEM, Manning n, infiltration conductivity, infiltration storage


def _member_blocks(report):
    """Return the first explicit final-holdout block in a training/evaluation report."""
    for key in ("holdout_metrics", "held_out", "val_members"):
        block = report.get(key)
        if isinstance(block, list) and block:
            return block
    for run in report.get("runs", []):
        if not isinstance(run, dict):
            continue
        for key in ("holdout_metrics", "held_out", "val_members"):
            block = run.get(key)
            if isinstance(block, list) and block:
                return block
    raise ValueError("selection report contains no per-member holdout block")


def _level_key(entry):
    """Match intervention and baseline members using the manifest naming contract."""
    return str(entry["name"]).split("_", 1)[0]


def _kinds(entry):
    values = sorted({str(x.get("kind")) for x in (entry.get("interventions") or [])
                     if x.get("kind")})
    return "+".join(values) if values else "baseline"


def _rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def _safe_percentile(values, q):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, q)) if values.size else float("nan")


def _error_stats(error):
    error = np.asarray(error, dtype=float)
    if not error.size:
        return {"rmse_m": float("nan"), "mae_m": float("nan"),
                "bias_m": float("nan"), "p95_abs_m": float("nan"),
                "p99_abs_m": float("nan"), "max_abs_m": float("nan")}
    ae = np.abs(error)
    return {"rmse_m": _rmse(error), "mae_m": float(ae.mean()),
            "bias_m": float(error.mean()), "p95_abs_m": _safe_percentile(ae, 95),
            "p99_abs_m": _safe_percentile(ae, 99), "max_abs_m": float(ae.max())}


def _csi(pred, truth, mask, threshold):
    pf = mask & (pred > threshold)
    tf = mask & (truth > threshold)
    den = int((pf | tf).sum())
    return float((pf & tf).sum()) / den if den else float("nan")


def intervention_footprint(member_x, baseline_x, atol=1e-6):
    """Union of cells changed in any static emulator input field."""
    changed = np.abs(member_x[:STATIC_CHANNELS] - baseline_x[:STATIC_CHANNELS]) > atol
    return np.any(changed, axis=0)


def local_masks(footprint, land, truth, baseline_truth, *, radius_m, cell_m,
                wet_threshold, response_threshold):
    """Fixed, truth-defined masks; no predicted value influences membership."""
    if not np.any(footprint):
        shape = footprint.shape
        empty = np.zeros(shape, dtype=bool)
        return empty, empty, empty
    distance_m = distance_transform_edt(~footprint, sampling=float(cell_m))
    neighbourhood = land & (distance_m <= float(radius_m))
    union_wet = neighbourhood & ((truth > wet_threshold) |
                                 (baseline_truth > wet_threshold))
    true_delta = truth - baseline_truth
    response_active = neighbourhood & (np.abs(true_delta) >= response_threshold)
    return neighbourhood, union_wet, response_active


def _distribution(rows, key):
    a = np.asarray([r[key] for r in rows], dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0, "mean": None, "median": None, "p90": None, "maximum": None}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "maximum": float(a.max())}


def _summaries(rows):
    metrics = ("local_rmse_m", "local_mae_m", "local_p99_abs_m", "local_max_abs_m",
               "local_csi",
               "response_rmse_m", "active_response_rmse_m", "active_response_skill",
               "active_response_sign_accuracy")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["radius_m"], row["kind"])].append(row)
        grouped[(row["radius_m"], "ALL")].append(row)
    out = []
    for (radius, kind), members in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out.append({"radius_m": radius, "kind": kind, "members": len(members),
                    "metrics": {key: _distribution(members, key) for key in metrics}})
    return out


def _json_safe(value):
    """Convert non-finite diagnostics to JSON null rather than emitting invalid NaN tokens."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--selection-report", required=True,
                    help="training report whose original holdout member names are preserved")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=0,
                    help="reserved for interface compatibility; inference is member-wise")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cell-m", type=float, default=4.0)
    ap.add_argument("--radii-m", type=float, nargs="+", default=(50.0, 100.0, 250.0))
    ap.add_argument("--wet-threshold", type=float, default=0.10)
    ap.add_argument("--response-threshold", type=float, default=0.01)
    ap.add_argument("--analysis-elevation-cutoff", type=float, default=0.81,
                    help="NAVD88 cutoff used by the trained emulator mask")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test only: evaluate the first N selected members")
    args = ap.parse_args()

    import torch
    from coral.emulator.dataset import build_manifest, sample_to_arrays, set_grid_cache
    from coral.emulator.inference import load_model

    manifest_path = Path(args.manifest)
    report_path = Path(args.selection_report)
    entries = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    selected_names = [str(x["name"]) for x in _member_blocks(report) if x.get("name")]
    if args.limit:
        selected_names = selected_names[:args.limit]
    selected_set = set(selected_names)
    entry_by_name = {str(x["name"]): x for x in entries}
    absent = selected_set - set(entry_by_name)
    if absent:
        raise ValueError(f"{len(absent)} selected members absent from manifest; "
                         f"first: {sorted(absent)[:5]}")

    selected_entries = [entry_by_name[name] for name in selected_names]
    baseline_entries = [x for x in entries if not (x.get("interventions") or [])]
    baseline_by_level = {_level_key(x): x for x in baseline_entries}
    missing_levels = sorted({_level_key(x) for x in selected_entries} - set(baseline_by_level))
    if missing_levels:
        raise ValueError(f"no no-intervention baseline for levels: {missing_levels}")

    # Build only the requested members and their baselines, while retaining manifest order in
    # the final rows through selected_names.
    selected_samples = {x.name: x for x in build_manifest(selected_entries)}
    needed_baselines = [baseline_by_level[k] for k in sorted({_level_key(x)
                                                               for x in selected_entries})]
    baseline_samples = {_level_key(entry_by_name[x.name]): x
                        for x in build_manifest(needed_baselines)}
    set_grid_cache(args.cache)
    model, stats, loaded_device = load_model(args.ckpt, device=args.device)
    device = loaded_device
    mean = np.asarray(stats["mean"], dtype="float32")[:, None, None]
    std = np.asarray(stats["std"], dtype="float32")[:, None, None]

    def predict(sample):
        x, y, original_land = sample_to_arrays(sample)
        xn = (x - mean) / std
        with torch.no_grad():
            p = model(torch.from_numpy(xn)[None].to(device))[0, 0].detach().cpu().numpy()
        # Match the target contract: each field is zero outside its own analysis-land mask.
        # A later union mask then compares a DEM-edited member and baseline on fixed cells
        # without treating an unconstrained raw network value as a valid baseline prediction.
        p = np.where(original_land, p, 0.0)
        return x, y.astype(float), original_land, p.astype(float)

    baseline_cache = {}
    rows = []
    for index, name in enumerate(selected_names, start=1):
        entry = entry_by_name[name]
        level = _level_key(entry)
        if level not in baseline_cache:
            baseline_cache[level] = predict(baseline_samples[level])
        bx, by, _, bp = baseline_cache[level]
        mx, my, _, mp = predict(selected_samples[name])

        # Use a fixed union land mask so a DEM edit cannot decide which field is scored.
        land = ((mx[0] > args.analysis_elevation_cutoff) |
                (bx[0] > args.analysis_elevation_cutoff))
        my = np.where(land, my, 0.0)
        by_fixed = np.where(land, by, 0.0)
        mp = np.where(land, mp, 0.0)
        bp_fixed = np.where(land, bp, 0.0)
        footprint = intervention_footprint(mx, bx)
        field_error = mp - my
        true_delta = my - by_fixed
        predicted_delta = mp - bp_fixed
        response_error = predicted_delta - true_delta

        for radius in args.radii_m:
            neighbourhood, union_wet, active = local_masks(
                footprint, land, my, by_fixed, radius_m=radius, cell_m=args.cell_m,
                wet_threshold=args.wet_threshold,
                response_threshold=args.response_threshold)
            fs = _error_stats(field_error[union_wet])
            rs = _error_stats(response_error[union_wet])
            ars = _error_stats(response_error[active])
            signal_rms = _rmse(true_delta[active])
            active_skill = (1.0 - ars["rmse_m"] / signal_rms
                            if np.isfinite(signal_rms) and signal_rms > 0 else float("nan"))
            sign_accuracy = (float(np.mean(np.sign(predicted_delta[active]) ==
                                           np.sign(true_delta[active])))
                             if active.any() else float("nan"))
            row = {
                "name": name, "kind": _kinds(entry), "level": level,
                "slr_m": float((entry.get("forcing") or {}).get("slr_m", 0.0)),
                "siting": entry.get("siting"), "radius_m": float(radius),
                "footprint_cells": int(footprint.sum()),
                "footprint_m2": float(footprint.sum()) * args.cell_m ** 2,
                "neighbourhood_cells": int(neighbourhood.sum()),
                "local_union_wet_cells": int(union_wet.sum()),
                "active_response_cells": int(active.sum()),
                "local_rmse_m": fs["rmse_m"], "local_mae_m": fs["mae_m"],
                "local_bias_m": fs["bias_m"], "local_p95_abs_m": fs["p95_abs_m"],
                "local_p99_abs_m": fs["p99_abs_m"],
                "local_max_abs_m": fs["max_abs_m"],
                "local_csi": _csi(mp, my, neighbourhood, args.wet_threshold),
                "response_rmse_m": rs["rmse_m"], "response_mae_m": rs["mae_m"],
                "response_bias_m": rs["bias_m"],
                "active_response_rmse_m": ars["rmse_m"],
                "active_response_mae_m": ars["mae_m"],
                "active_response_bias_m": ars["bias_m"],
                "active_response_signal_rms_m": signal_rms,
                "active_response_skill": active_skill,
                "active_response_sign_accuracy": sign_accuracy,
            }
            rows.append(row)
        if index % 25 == 0 or index == len(selected_names):
            print(f"evaluated {index}/{len(selected_names)} members", flush=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "manifest": str(manifest_path.resolve()),
        "selection_report": str(report_path.resolve()),
        "evaluation_only": True,
        "n_selected_members": len(selected_names),
        "radii_m": [float(x) for x in args.radii_m],
        "definitions": {
            "analysis_land": ("union of member and same-sea-level baseline cells above "
                              f"{args.analysis_elevation_cutoff:g} m NAVD88"),
            "intervention_footprint": ("union of changed DEM, Manning n, infiltration "
                                       "conductivity, and infiltration-storage cells"),
            "local_field_mask": ("within the footprint buffer and wet above "
                                 f"{args.wet_threshold:g} m in either parent field"),
            "active_response_mask": ("within the footprint buffer with absolute parent "
                                     f"intervention response at least {args.response_threshold:g} m"),
            "response_error": ("(emulator intervention - emulator baseline) - "
                               "(parent intervention - parent baseline)"),
            "response_skill": ("1 - active response RMSE / RMS parent response; values <= 0 "
                               "mean ignoring the intervention is at least as accurate"),
        },
        "summaries": _summaries(rows),
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    print(f"CSV -> {out_csv}")
    print(f"JSON -> {out_json}")


if __name__ == "__main__":
    main()
