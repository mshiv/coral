"""GeoClaw surge against observed surge at the NOAA stations, for one or more runs.

Observed surge is water level minus the tidal prediction, so it is the residual the model is
actually solving for. With geoclaw.sea_level at 0.0 the modelled eta is the same quantity and
the two are directly comparable; on an older run with the tide in the datum, pass
--datum-offset to remove it.

Model time is mapped to UTC through coupling.landfall_utc. That mapping is the thing this
figure is most sensitive to, so it is also measured: for each station the script cross
correlates the modelled and observed series and reports the lag that best aligns them. A lag
that is consistently one hour across every station is a clock error, not physics.

Stations are the 9001-9005 block written by setrun. Older runs used 64-68; pass --ids for those.

    python -m coral.validate.noaa_surge_validation \\
        --runs <l6>/_output <l7>/_output --labels "level 6" "level 7" \\
        --scenario configs/scenarios/savannah_matthew_compound.yaml \\
        --out reports/figures/noaa_surge_validation.png
"""
from __future__ import annotations
import argparse
import datetime
import json
import urllib.request
from pathlib import Path

import numpy as np

from ..viz.gauge_io import read_gauge
from ..viz.pinpoint_style import PALETTE

# setrun writes these in order; Fort Pulaski is 9002.
STATIONS = {9001: ("8720218", "Mayport, FL"),
            9002: ("8670870", "Fort Pulaski, GA"),
            9003: ("8665530", "Charleston, SC"),
            9004: ("8658163", "Wrightsville Beach, NC"),
            9005: ("8658120", "Wilmington, NC")}

API = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={b}"
       "&end_date={e}&station={s}&product={p}&datum=MSL&time_zone=GMT&units=metric"
       "&application=coral&format=json")


def observed_surge(station, begin="20161005", end="20161012"):
    """(datetime64, surge m) = water level minus tidal prediction."""
    def get(product):
        url = API.format(b=begin, e=end, s=station, p=product)
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode())
    try:
        wl = get("water_level").get("data", [])
        pr = {p["t"]: float(p["v"])
              for p in get("predictions").get("predictions", []) if p.get("v") not in ("", None)}
    except Exception as exc:
        print(f"  station {station}: NOAA fetch failed ({exc})")
        return None, None
    t, s = [], []
    for w in wl:
        v, p = w.get("v"), pr.get(w["t"])
        if v in ("", None) or p is None:
            continue
        t.append(np.datetime64(w["t"].replace(" ", "T")))
        s.append(float(v) - p)
    if not t:
        return None, None
    return np.array(t), np.array(s)


def best_lag(t_mod, y_mod, t_obs, y_obs, max_lag_h=6.0, step_min=10):
    """Lag in hours that best aligns model to observation, by minimising RMSE.

    Positive means the model is LATE: shifting it earlier improves the fit.
    """
    lags = np.arange(-max_lag_h * 60, max_lag_h * 60 + 1, step_min) / 60.0
    lo, hi = max(t_mod.min(), t_obs.min()), min(t_mod.max(), t_obs.max())
    grid = np.arange(lo, hi, 600.0)              # 10 min, in model seconds
    if grid.size < 6:
        return np.nan, np.nan
    o = np.interp(grid, t_obs, y_obs)
    best, best_r = np.nan, np.inf
    for L in lags:
        m = np.interp(grid, t_mod - L * 3600.0, y_mod)
        r = float(np.sqrt(np.mean((m - o) ** 2)))
        if r < best_r:
            best, best_r = L, r
    return best, best_r


def build(run_dirs, out, *, labels=None, scenario=None, landfall_utc=None,
          datum_offset=0.0, ids=None, begin="20161005", end="20161012",
          baseline_window_h=(-24.0, -6.0), max_lag_h=6.0, out_json=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if scenario:
        from ..config import load
        cfg = load(scenario)
        landfall_utc = landfall_utc or cfg.coupling.landfall_utc
        print(f"scenario {cfg.name}: landfall_utc {landfall_utc}, "
              f"geoclaw.sea_level {cfg.geoclaw.sea_level}")
    if not landfall_utc:
        raise SystemExit("need --scenario or --landfall-utc to map model time to UTC")
    t0 = np.datetime64(datetime.datetime.fromisoformat(landfall_utc))

    labels = list(labels or [Path(d).parent.name for d in run_dirs])
    ids = ids or sorted(STATIONS)
    colours = [PALETTE["muted"], PALETTE["flood"], PALETTE["intervention"]]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes = axes.flatten()
    lag_table = {lab: [] for lab in labels}

    for ax, gid in zip(axes, ids):
        stn, name = STATIONS.get(gid, (None, f"gauge {gid}"))
        to, so = (observed_surge(stn, begin, end) if stn else (None, None))
        if to is not None:
            ax.plot(to, so, lw=1.8, color=PALETTE["text"], label="observed", zorder=3)

        for k, (d, lab) in enumerate(zip(run_dirs, labels)):
            p = Path(d) / f"gauge{gid:05d}.txt"
            if not p.exists():
                print(f"  {lab}: {p.name} missing")
                continue
            _, _, tm, eta = read_gauge(p)
            if tm.size < 2:
                continue
            eta = eta - datum_offset
            tdt = t0 + (tm * 1e9).astype("timedelta64[ns]")
            ax.plot(tdt, eta, lw=1.3, color=colours[k % len(colours)], label=lab)

            if to is not None:
                tos = (to - t0) / np.timedelta64(1, "s")
                tos = tos.astype(float)
                L, r = best_lag(tm, eta, tos, so, max_lag_h=max_lag_h)
                lo, hi = max(tm.min(), tos.min()), min(tm.max(), tos.max())
                grid = np.arange(lo, hi, 600.0)
                mod = np.interp(grid, tm, eta)
                obs = np.interp(grid, tos, so)
                bias = float(np.mean(mod - obs)) if grid.size else np.nan
                bm = ((grid >= baseline_window_h[0] * 3600.0) &
                      (grid <= baseline_window_h[1] * 3600.0))
                # Positive means the unmodelled observed setup is above GeoClaw and should be
                # added to the boundary. It is diagnostic, not automatically a calibration.
                setup = float(np.mean(obs[bm] - mod[bm])) if bm.any() else np.nan
                lag_table[lab].append(dict(station=name, lag_h=L, rmse_m=r,
                                           model_peak_m=float(mod.max()),
                                           observed_peak_m=float(obs.max()),
                                           model_minus_observed_bias_m=bias,
                                           prestorm_observed_minus_model_m=setup))

        ax.set_title(name, fontsize=10.5, color=PALETTE["text"])
        ax.set_ylabel("surge (m)", fontsize=9)
        ax.tick_params(labelsize=7, axis="x", rotation=25)
        ax.grid(alpha=0.25)
        ax.axhline(0, color=PALETTE["muted"], lw=0.7)
    axes[0].legend(fontsize=8, frameon=False)
    for ax in axes[len(ids):]:
        ax.axis("off")

    print("\nstation                    label            peak mod  peak obs   lag h   rmse  bias m  pre-setup")
    for lab, rows in lag_table.items():
        for row in rows:
            print(f"  {row['station']:26s} {lab:14s} {row['model_peak_m']:+7.3f}  "
                  f"{row['observed_peak_m']:+7.3f}  {row['lag_h']:+6.2f}  "
                  f"{row['rmse_m']:6.3f}  {row['model_minus_observed_bias_m']:+7.3f}  "
                  f"{row['prestorm_observed_minus_model_m']:+9.3f}")
        if rows:
            med = float(np.median([x["lag_h"] for x in rows]))
            print(f"  {'':26s} {lab:14s} median lag across stations: {med:+.2f} h")

    if out_json:
        report = {"scenario": scenario, "landfall_utc": landfall_utc,
                  "begin": begin, "end": end,
                  "prestorm_window_h": list(baseline_window_h),
                  "max_lag_h": max_lag_h, "runs": lag_table}
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out_json}")

    fig.suptitle("Modelled surge against observed, NOAA stations",
                 fontsize=14, y=0.98, color=PALETTE["text"])
    fig.text(0.5, 0.005,
             "Observed surge is water level minus tidal prediction. A lag that is the same "
             "sign and size at every station is a clock error in landfall_utc, not a "
             "modelling error.", ha="center", fontsize=8.6, color=PALETTE["muted"])
    fig.subplots_adjust(hspace=0.42, wspace=0.2, top=0.93)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="one or more _output directories")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--landfall-utc", default=None)
    ap.add_argument("--datum-offset", type=float, default=0.0,
                    help="subtract from eta; 0.81 for runs with the old static tide")
    ap.add_argument("--ids", nargs="+", type=int, default=None)
    ap.add_argument("--begin", default="20161005")
    ap.add_argument("--end", default="20161012")
    ap.add_argument("--prestorm-window-h", nargs=2, type=float, default=(-24.0, -6.0))
    ap.add_argument("--max-lag-h", type=float, default=6.0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out", default="reports/figures/noaa_surge_validation.png")
    a = ap.parse_args()
    build(a.runs, a.out, labels=a.labels, scenario=a.scenario,
          landfall_utc=a.landfall_utc, datum_offset=a.datum_offset, ids=a.ids,
          begin=a.begin, end=a.end, baseline_window_h=tuple(a.prestorm_window_h),
          max_lag_h=a.max_lag_h, out_json=a.out_json)


if __name__ == "__main__":
    main()
