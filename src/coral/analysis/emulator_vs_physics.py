"""Emulator prediction against the physics it approximates, on held-out members.

A held-out RMSE is a summary and a reader cannot see a field in it. This draws the comparison
directly: the parent model's peak depth, the emulator's prediction, and their difference, for the
members that matter most — the best case, the median, and the worst, chosen from the report the
training run already wrote.

The worst case is included deliberately. An emulator figure that shows only agreement invites the
question of what disagreement looks like, and in this ensemble the answer is specific: the worst
members are floodwalls, where a near-discontinuity in elevation produces a step in depth that a
convolutional decoder reconstructing from downsampled features tends to smooth.

Usage:
  python -m coral.analysis.emulator_vs_physics --ckpt <emulator .pt> \
      --report <emulator .report.json> --ens <ensemble dir> --dem <DEM .asc> \
      --waterline 1.114 --out reports/figures/emulator_vs_physics.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

NODATA = -9999.0


def rd(path):
    a = np.loadtxt(path, skiprows=6)
    return np.where(a > NODATA + 1.0, a, np.nan)


def find_max(run_dir):
    hits = sorted(glob.glob(str(Path(run_dir) / "results_*" / "*.max")))
    return hits[0] if hits else None


def pick_members(report, n_mid=1):
    """Best, median and worst held-out members by RMSE, from the training report.

    Reading the report rather than re-scoring keeps the figure consistent with the number the
    chapter quotes: a member picked by a second, slightly different metric could disagree with
    the reported ranking and invite a question about which is right.
    """
    runs = report.get("runs") or []
    per = None
    for r in runs:
        for k in ("val_members", "held_out", "members", "per_member"):
            if isinstance(r.get(k), list) and r[k]:
                per = r[k]; break
        if per:
            break
    if not per:
        raise SystemExit("no per-member block in the report; cannot choose members")
    key = next((k for k in ("rmse", "rmse_m", "wet_rmse", "RMSE") if k in per[0]), None)
    if key is None:
        raise SystemExit(f"no rmse field in per-member records: {list(per[0])}")
    per = sorted(per, key=lambda x: x[key])
    mid = per[len(per) // 2]
    return [("best", per[0], key), ("median", mid, key), ("worst", per[-1], key)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--ens", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--waterline", type=float, required=True)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--out", default="reports/figures/emulator_vs_physics.png")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from coral.emulator.dataset import build_manifest
    from coral.emulator.inference import load_model, predict

    report = json.load(open(a.report))
    chosen = pick_members(report)
    manifest = {e["name"]: e for e in json.load(open(Path(a.ens) / "manifest.json"))}
    dem = rd(a.dem)
    land = np.isfinite(dem) & (dem > a.waterline)

    model, stats, device = load_model(a.ckpt)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from coral.viz.pinpoint_style import make_flood_cmap
    FLOOD = make_flood_cmap()
    DIV = LinearSegmentedColormap.from_list("err", ["#2c7fb8", "#ffffff", "#a63f22"])

    fig, ax = plt.subplots(len(chosen), 3, figsize=(13.5, 4.0 * len(chosen)), squeeze=False)
    for i, (label, rec, key) in enumerate(chosen):
        name = rec.get("name") or rec.get("member")
        e = manifest.get(name)
        if e is None:
            continue
        truth = np.nan_to_num(rd(find_max(e["run_dir"])), nan=0.0)
        sample = build_manifest([e], skip_missing=True)[0]
        pred = np.asarray(predict(model, stats, sample, device), dtype=float)
        if pred.shape != truth.shape:
            raise SystemExit(f"prediction {pred.shape} does not match parent {truth.shape}")
        err = np.where(land, pred - truth, np.nan)
        v = float(np.nanpercentile(np.where(land, truth, np.nan), 99)) or 0.5
        e99 = float(np.nanpercentile(np.abs(err[np.isfinite(err)]), 99.5)) or 0.05

        for j, (arr, ttl, cm, kw) in enumerate([
                (np.where(land & (truth > a.threshold), truth, np.nan),
                 "parent model (LISFLOOD-FP)", FLOOD, dict(vmin=0, vmax=v)),
                (np.where(land & (pred > a.threshold), pred, np.nan),
                 "emulator", FLOOD, dict(vmin=0, vmax=v)),
                (np.where(np.abs(err) > 0.005, err, np.nan),
                 "emulator minus parent", DIV, dict(vmin=-e99, vmax=e99))]):
            im = ax[i][j].imshow(arr, cmap=cm, interpolation="none", **kw)
            ax[i][j].set_xticks([]); ax[i][j].set_yticks([])
            ax[i][j].set_title(f"{label}: {ttl}" if j == 0 else ttl, fontsize=10)
            fig.colorbar(im, ax=ax[i][j], fraction=0.046,
                         label="depth (m)" if j < 2 else "error (m)")
        ax[i][2].set_xlabel(f"{name}   RMSE {rec[key]:.3f} m", fontsize=8.5, color="0.35")

    fig.suptitle("Emulator prediction against the parent model, on held-out members", fontsize=13)
    fig.text(0.5, 0.005, "Blue: emulator shallower than the parent.  Red: deeper.",
             ha="center", fontsize=9, color="0.4")
    fig.tight_layout(rect=[0, 0.012, 1, 0.97])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
