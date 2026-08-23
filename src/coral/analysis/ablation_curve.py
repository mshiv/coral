"""Read the marsh-roughness ablation and draw the response curve.

The question is whether marsh roughness has purchase on the flood at all in this regime, and a
single summary number cannot answer it: a mean over the whole domain dilutes the platform's
contribution with upland that the edit never touched, and a maximum reports one cell. So the
response is reported as several quantities over several zones, against roughness, stratified by
sea level.

Flat curves across a twenty-fold roughness range would say that still-water peak depth in this
regime is insensitive to marsh friction, and that the near-null intervention response measured in
the production ensemble is a property of the regime rather than of the sampled range. Curves that
bend say the opposite, and the bend point locates where friction stops mattering.

The control state (baseline roughness, zero offset) must reproduce the run the ablation was staged
from. It is checked first and reported loudly, because if it does not, nothing below means
anything.

Usage:
  python -m coral.analysis.ablation_curve --root <rough_ablation dir> \
      --dem <30 m DEM .asc> --control <compound_tide_30m results .max> \
      --waterline 1.114 --mlw -1.091 --cell-m 30 \
      --out reports/figures/roughness_ablation.png --out-json reports/compound/ablation.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

NODATA = -9999.0
THRESHOLDS = (0.05, 0.10, 0.30, 0.50, 1.00)


def rd(path):
    a = np.loadtxt(path, skiprows=6)
    return np.where(a > NODATA + 1.0, a, np.nan)


def find_max(run_dir):
    hits = sorted(glob.glob(str(Path(run_dir) / "results_*" / "*.max")))
    return hits[0] if hits else None


def zones(dem, waterline, mlw):
    """Three zones, because the same edit means different things in each.

    The platform is where the roughness was changed. Land above the waterline is where a
    community sits and where a depth change is a consequence rather than a direct edit.
    Channels drain both, and an edit that slows the platform can raise levels in them.
    """
    return {
        "marsh platform (MLW to MHW)": np.isfinite(dem) & (dem >= mlw) & (dem <= waterline),
        "land above MHW": np.isfinite(dem) & (dem > waterline),
        "channels (below MLW)": np.isfinite(dem) & (dem < mlw),
    }


def metrics(depth, zone, cell_m):
    """Depth statistics and flooded area at several thresholds, over one zone.

    Percentiles rather than a maximum: one cell at the edge of a channel should not stand for
    the response of six hundred thousand. Area at several thresholds rather than one, because a
    shallow pathway and a metre of water are different decisions.
    """
    d = depth[zone & np.isfinite(depth)]
    if d.size == 0:
        return None
    a = cell_m ** 2
    out = {"cells": int(d.size),
           "mean_m": float(d.mean()),
           "p50_m": float(np.percentile(d, 50)),
           "p90_m": float(np.percentile(d, 90)),
           "p99_m": float(np.percentile(d, 99)),
           "volume_m3": float(d.sum() * a)}
    for t in THRESHOLDS:
        out[f"area_ge_{t:g}m_km2"] = float((d > t).sum()) * a / 1e6
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--control", default=None,
                    help="the .max the ablation was staged from; the n=baseline slr=0 state "
                         "must reproduce it")
    ap.add_argument("--waterline", type=float, required=True)
    ap.add_argument("--mlw", type=float, required=True)
    ap.add_argument("--cell-m", type=float, default=30.0)
    ap.add_argument("--out", default="reports/figures/roughness_ablation.png")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    manifest = json.load(open(root / "manifest.json"))
    dem = rd(a.dem)
    Z = zones(dem, a.waterline, a.mlw)
    for k, v in Z.items():
        print(f"{k:30s} {int(v.sum()):,} cells")

    rows, missing = [], []
    for m in manifest:
        f = find_max(m["run_dir"])
        if f is None:
            missing.append(m["name"]); continue
        d = np.nan_to_num(rd(f), nan=0.0)
        rec = {"name": m["name"], "n": m.get("n_target", m.get("multiplier")),
               "slr_m": m["slr_m"], "max": f}
        for zn, zm in Z.items():
            rec[zn] = metrics(d, zm, a.cell_m)
        rows.append(rec)
    if missing:
        print(f"\n{len(missing)} run(s) with no .max yet: {', '.join(missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}")
    if not rows:
        raise SystemExit("no finished runs")

    # The control. Everything downstream is conditional on this.
    if a.control:
        ctrl = [r for r in rows if r["slr_m"] == 0 and r["n"] is not None and r["n"] < 0]
        if ctrl:
            c = np.nan_to_num(rd(ctrl[0]["max"]), nan=0.0)
            b = np.nan_to_num(rd(a.control), nan=0.0)
            diff = np.abs(c - b)
            print(f"\ncontrol {ctrl[0]['name']} against the staged base:"
                  f"  max|diff| {diff.max():.3e} m  mean {diff.mean():.3e} m")
            if diff.max() > 1e-3:
                print("  WARNING: the control does not reproduce its base. The staging changed "
                      "something other than roughness, and the curves below are not a clean "
                      "ablation.")
        else:
            print("\nno untouched-grid control state in this ablation. Setting the band to its "
                  "own median is not a control: the band holds open-water and channel-edge cells "
                  "below that median, so overwriting them changes the run. Restage with a "
                  "negative state to add one.")

    print(f"\n{'state':22s} {'n':>6} {'slr':>6}  "
          f"{'platform p90':>12} {'land p90':>10} {'land >0.3m km2':>15} {'land vol Mm3':>13}")
    for r in sorted(rows, key=lambda x: (x["slr_m"], x["n"])):
        pf = r["marsh platform (MLW to MHW)"]; ld = r["land above MHW"]
        if not (pf and ld):
            continue
        print(f"{r['name']:22s} {r['n']:6.3f} {r['slr_m']:6.3f}  "
              f"{pf['p90_m']:12.3f} {ld['p90_m']:10.3f} "
              f"{ld['area_ge_0.3m_km2']:15.3f} {ld['volume_m3']/1e6:13.2f}")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out_json}")

    # ---- figure: response against roughness, one line per sea level
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slrs = sorted({r["slr_m"] for r in rows})
    panels = [("land above MHW", "p90_m", "land depth p90 (m)"),
              ("land above MHW", "area_ge_0.3m_km2", "land area >0.3 m (km$^2$)"),
              ("land above MHW", "volume_m3", "land flood volume (m$^3$)"),
              ("marsh platform (MLW to MHW)", "p90_m", "platform depth p90 (m)")]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    cmap = plt.get_cmap("viridis")
    for j, (zn, key, lab) in enumerate(panels):
        for i, s in enumerate(slrs):
            pts = sorted([(r["n"], r[zn][key]) for r in rows
                          if r["slr_m"] == s and r.get(zn) and r["n"] and r["n"] > 0],
                         key=lambda x: x[0])
            if len(pts) < 2:
                continue
            x, y = zip(*pts)
            ax[j].plot(x, y, "o-", color=cmap(i / max(len(slrs) - 1, 1)),
                       label=f"SLR {s:.2f} m", lw=1.6, ms=4)
        ax[j].set_xscale("log")
        # Explicit ticks: log minor labels collide at five closely spaced states and the axis
        # becomes unreadable.
        xs = sorted({r["n"] for r in rows if r["n"] and r["n"] > 0})
        ax[j].set_xticks(xs)
        ax[j].set_xticklabels([f"{x:g}" for x in xs], fontsize=8.5)
        ax[j].minorticks_off()
        ax[j].set_xlabel("marsh Manning's $n$")
        ax[j].set_ylabel(lab)
        ax[j].grid(alpha=0.25, lw=0.5)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Flood response to marsh roughness, everything else held fixed", fontsize=13)
    fig.text(0.5, 0.005, "A flat line means still-water peak depth in this regime is insensitive "
             "to marsh friction across the plotted range.", ha="center", fontsize=9, color="0.4")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
