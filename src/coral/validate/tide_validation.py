"""Validate the modelled tide/surge boundary against observations.

The tide is imposed from NOAA CO-OPS Fort Pulaski (8670870). NOAA publishes both the
observed water level and the astronomical prediction for the event, so the modelled
boundary can be checked against each:

    observed = prediction (astronomical tide) + residual (surge + post-storm anomaly)

The model boundary is surge + tide. Recovering its tidal part as (full - surge_only)
and comparing to the NOAA prediction tests the tide phase and amplitude; comparing the
full boundary to observed water level tests the surge.

There was no in-situ water-level gauge at Pin Point during Matthew (Oct 2016). The
nearest Vernon/Skidaway station, USGS 315651081035601 at Diamond Causeway, was installed
2019-08-31 and cannot validate the event. It is included here only for post-2019 events.

Model time zero is MODEL_T0_UTC (see build_bdy). All times are on the model clock.

Usage:
    python -m coral.validate.tide_validation \\
        --full runs/compound30m/matthew_tide.bdy \\
        --surge runs/compound30m/matthew_savannah.bdy \\
        --block bc1 --out reports/tide_validation.png
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from datetime import datetime, timezone

import numpy as np

from ..couple.build_bdy import MODEL_T0_UTC

FORT_PULASKI = "8670870"


def _coops(product, begin, end, station=FORT_PULASKI):
    """NOAA CO-OPS series -> (model_seconds, level_m). product = predictions | water_level."""
    t0 = datetime.fromisoformat(MODEL_T0_UTC).replace(tzinfo=timezone.utc)
    u = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
         f"product={product}&application=coral&begin_date={begin}&end_date={end}"
         f"&datum=NAVD&station={station}&time_zone=gmt&units=metric&interval=h&format=json")
    d = json.load(urllib.request.urlopen(u, timeout=90))
    rows = d["predictions"] if product == "predictions" else d["data"]
    t, v = [], []
    for r in rows:
        if r.get("v") in ("", None):
            continue
        dt = datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        t.append((dt - t0).total_seconds()); v.append(float(r["v"]))
    return np.array(t), np.array(v)


def _block(bdy_path, name):
    """Read one .bdy block -> (values, model_seconds)."""
    tok = open(bdy_path).read().split()
    i = 0
    while i < len(tok):
        if tok[i] == name:
            n = int(tok[i + 1]); i += 3
            return np.array(tok[i:i + 2 * n:2], dtype=float), np.array(tok[i + 1:i + 2 * n:2], dtype=float)
        i += 1
    raise SystemExit(f"block {name!r} not found in {bdy_path}")


def validate(full_bdy, surge_bdy, block="bc1", *, begin="20161005", end="20161016",
             out=None, t_lo=90000.0, t_hi=250000.0):
    """Compare the modelled boundary to NOAA observed and predicted water level.

    Returns a dict of statistics. Writes a figure if `out` is given.
    """
    fv, ft = _block(full_bdy, block)
    sv, st = _block(surge_bdy, block)
    pred_t, pred_v = _coops("predictions", begin, end)
    obs_t, obs_v = _coops("water_level", begin, end)

    # Common model-time grid over the GeoClaw window.
    T = np.arange(max(t_lo, ft.min(), st.min()), min(t_hi, ft.max(), st.max()), 1800.0)
    full = np.interp(T, ft, fv)
    surge = np.interp(T, st, sv)
    tide_model = full - surge                       # recovered tidal component
    pred = np.interp(T, pred_t, pred_v)
    obs = np.interp(T, obs_t, obs_v)

    def _stats(a, b):
        return dict(r=float(np.corrcoef(a, b)[0, 1]),
                    rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                    bias=float(np.mean(a - b)))

    tlf = 172800.0
    res = {
        "tide_vs_pred": _stats(tide_model, pred),
        "full_vs_obs": _stats(full, obs),
        "landfall": dict(model_tide=float(np.interp(tlf, ft, fv) - np.interp(tlf, st, sv)),
                         noaa_pred=float(np.interp(tlf, pred_t, pred_v)),
                         model_full=float(np.interp(tlf, ft, fv)),
                         noaa_obs=float(np.interp(tlf, obs_t, obs_v))),
    }

    print(f"tide (model) vs NOAA prediction : r={res['tide_vs_pred']['r']:.4f}  "
          f"rmse={res['tide_vs_pred']['rmse']:.3f} m  bias={res['tide_vs_pred']['bias']:+.3f} m")
    print(f"full boundary   vs NOAA observed : r={res['full_vs_obs']['r']:.4f}  "
          f"rmse={res['full_vs_obs']['rmse']:.3f} m  bias={res['full_vs_obs']['bias']:+.3f} m")
    lf = res["landfall"]
    print(f"at landfall: model tide {lf['model_tide']:+.3f} vs NOAA pred {lf['noaa_pred']:+.3f} m "
          f"(low water: Matthew did not hit at high tide)")
    print(f"            model total {lf['model_full']:+.3f} vs NOAA obs {lf['noaa_obs']:+.3f} m")

    if out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hr = (T - tlf) / 3600.0
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax[0].plot(hr, obs, "k", lw=1.5, label="NOAA observed (Fort Pulaski)")
        ax[0].plot(hr, full, "C3", lw=1.2, label="model boundary (surge+tide)")
        ax[0].axvline(0, color="0.6", ls="--", lw=0.8)
        ax[0].set_ylabel("water level (m NAVD88)")
        ax[0].set_title(f"Boundary vs observed water level  "
                        f"(r={res['full_vs_obs']['r']:.3f}, RMSE {res['full_vs_obs']['rmse']:.2f} m)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].plot(hr, pred, "k", lw=1.5, label="NOAA astronomical prediction")
        ax[1].plot(hr, tide_model, "C0", lw=1.2, label="model tide (full − surge)")
        ax[1].axvline(0, color="0.6", ls="--", lw=0.8)
        ax[1].axhline(lf["noaa_pred"], color="0.7", ls=":", lw=0.8)
        ax[1].annotate(f"landfall tide {lf['noaa_pred']:+.2f} m (near low water)",
                       (0, lf["noaa_pred"]), fontsize=8, xytext=(5, -12),
                       textcoords="offset points")
        ax[1].set_ylabel("tidal elevation (m)")
        ax[1].set_xlabel("hours from landfall (2016-10-08 13:00 UTC)")
        ax[1].set_title(f"Tide phase  (r={res['tide_vs_pred']['r']:.4f})")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="surge+tide .bdy (e.g. matthew_tide.bdy)")
    ap.add_argument("--surge", required=True, help="surge-only .bdy (e.g. matthew_savannah.bdy)")
    ap.add_argument("--block", default="bc1")
    ap.add_argument("--begin", default="20161005"); ap.add_argument("--end", default="20161016")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    validate(a.full, a.surge, a.block, begin=a.begin, end=a.end, out=a.out)


if __name__ == "__main__":
    main()
