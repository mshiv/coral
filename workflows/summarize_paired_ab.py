#!/usr/bin/env python
"""Summarize the controlled absolute-depth versus paired-difference U-Net ablation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PRIMARY = "mean_delta_mae_response_0p05_m"
ABS_RMSE = "mean_rmse_m"
CSI = "mean_csi"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rmse-degradation-pct", type=float, default=5.0)
    parser.add_argument("--max-csi-drop", type=float, default=0.005)
    args = parser.parse_args()
    control = json.loads(Path(args.control).read_text())
    paired = json.loads(Path(args.paired).read_text())
    fixed = ["ensemble", "split", "seed", "n_train", "n_validation"]
    mismatch = {key: (control.get(key), paired.get(key)) for key in fixed
                if control.get(key) != paired.get(key)}
    if mismatch:
        raise ValueError(f"A/B reports do not describe the same experiment: {mismatch}")

    c_delta = control["delta_summary"][PRIMARY]
    p_delta = paired["delta_summary"][PRIMARY]
    c_rmse = control["absolute_summary"][ABS_RMSE]
    p_rmse = paired["absolute_summary"][ABS_RMSE]
    c_csi = control["absolute_summary"][CSI]
    p_csi = paired["absolute_summary"][CSI]
    delta_improvement = 100.0 * (c_delta - p_delta) / c_delta
    rmse_change = 100.0 * (p_rmse - c_rmse) / c_rmse
    csi_change = p_csi - c_csi
    passes_primary = p_delta < c_delta
    passes_guardrails = (rmse_change <= args.max_rmse_degradation_pct and
                         csi_change >= -args.max_csi_drop)

    kinds = sorted(set(control.get("delta_by_kind", {})) | set(paired.get("delta_by_kind", {})))
    by_kind = []
    for kind in kinds:
        cv = control.get("delta_by_kind", {}).get(kind, {}).get(PRIMARY)
        pv = paired.get("delta_by_kind", {}).get(kind, {}).get(PRIMARY)
        by_kind.append({"kind": kind, "control_delta_mae_0p05_m": cv,
                        "paired_delta_mae_0p05_m": pv,
                        "improvement_pct": (100.0 * (cv - pv) / cv
                                            if cv not in (None, 0) and pv is not None else None)})
    result = {
        "decision_rule": {
            "primary": "paired mean delta-h MAE on parent-response cells >=0.05 m is lower",
            "guardrails": [f"absolute RMSE degradation <= {args.max_rmse_degradation_pct}%",
                           f"CSI drop <= {args.max_csi_drop}"],
        },
        "primary": {"control_m": c_delta, "paired_m": p_delta,
                    "improvement_pct": delta_improvement, "passes": passes_primary},
        "guardrails": {
            "absolute_rmse_control_m": c_rmse, "absolute_rmse_paired_m": p_rmse,
            "absolute_rmse_change_pct": rmse_change,
            "csi_control": c_csi, "csi_paired": p_csi, "csi_change": csi_change,
            "passes": passes_guardrails,
        },
        "verdict": ("paired objective improves intervention response without degrading the flood field"
                    if passes_primary and passes_guardrails else
                    "paired objective does not satisfy the predeclared primary and guardrail criteria"),
        "by_kind": by_kind,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_kind[0]) if by_kind else ["kind"])
        writer.writeheader(); writer.writerows(by_kind)
    print(json.dumps(result, indent=2))
    print("summary ->", out)


if __name__ == "__main__":
    main()
