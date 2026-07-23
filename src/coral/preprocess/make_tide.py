"""Build a tidal water-level series for the LISFLOOD coastal boundary.

The coral surge boundary comes from GeoClaw with a *static* sea_level baked in.
To model the time-varying astronomical tide (and its phase relative to the surge
peak), we fetch the tide separately from NOAA CO-OPS and add it onto the surge in
build_bdy. This is the linear-superposition approach to compound coastal water
level: TWL(t) = tide(t) + surge_residual(t).

Two products (NOAA CO-OPS datagetter):
  - "predictions"  = astronomical tide only (harmonic), any past/future time.
    Use for scenarios/forecasts and the surge+tide compound run.
  - "water_level"  = observed verified total level (tide + surge + everything),
    past only. Use as a hindcast/validation boundary (the real total at the gauge).

Critical: fetched on datum=NAVD (NAVD88) to match the DEM/.bdy vertical datum, and
returned on the MODEL clock (seconds, t=0 = landfall) so it lines up with the surge.

Two-tier per repo convention:
  fetch_tide(...)   -- pure worker (station, window, datum, product)
  from_config(cfg)  -- adapter: station + window from the config, model-clock aligned

Deps: stdlib only (urllib + json).
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import json
import urllib.request
import urllib.parse

COOPS = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def fetch_tide(station, t0, t1, *, datum="NAVD", product="predictions", interval=None):
    """Fetch NOAA CO-OPS levels for `station` over [t0, t1] (UTC datetimes).

    Returns (times_utc, levels_m) as parallel lists. `product` is "predictions"
    (tide) or "water_level" (observed total). `interval` e.g. "h" or "6" (min);
    None = product default (6-min).
    """
    params = {
        "begin_date": t0.strftime("%Y%m%d %H:%M"),
        "end_date": t1.strftime("%Y%m%d %H:%M"),
        "station": str(station),
        "product": product,
        "datum": datum,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "application": "coral",
    }
    if interval:
        params["interval"] = interval
    url = f"{COOPS}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read())
    if "error" in payload:
        raise SystemExit(f"NOAA CO-OPS error: {payload['error'].get('message')}")
    key = "predictions" if product == "predictions" else "data"
    rows = payload.get(key, [])
    if not rows:
        raise SystemExit(f"NOAA returned no {product} for station {station} {t0}..{t1}")
    from datetime import datetime
    times, levels = [], []
    for row in rows:
        v = row.get("v")
        if v in (None, ""):
            continue                                   # gaps in observed data
        times.append(datetime.strptime(row["t"], "%Y-%m-%d %H:%M"))
        levels.append(float(v))
    print(f"tide: {len(levels)} pts, {product}@{datum}, "
          f"range {min(levels):.2f}..{max(levels):.2f} m")
    return times, levels


def to_model_seconds(times_utc, landfall_dt, landfall_s):
    """Map UTC datetimes to the shifted model clock (t=0 at landfall -> landfall_s)."""
    return [(t - landfall_dt).total_seconds() + landfall_s for t in times_utc]


def from_config(cfg, out=None, product="predictions"):
    """Adapter: station + [landfall-window] from config; write model-clock tide series.

    Writes a 2-column text file (model_seconds  level_m) to data/interim/ for
    build_bdy to superpose onto the surge gauges. Returns the path.
    """
    if cfg.forcing.tide is None:
        raise SystemExit(f"{cfg.name}: forcing.tide is null (static sea_level run)")
    if cfg.landfall_dt is None:
        raise SystemExit("coupling.landfall_utc required to align tide to the model clock")
    lf = cfg.landfall_dt
    t0 = lf + timedelta(hours=cfg.lisflood.sim_window_h[0])
    t1 = lf + timedelta(hours=cfg.lisflood.sim_window_h[1])
    times, levels = fetch_tide(cfg.forcing.tide_station, t0, t1, product=product)
    secs = to_model_seconds(times, lf, cfg.coupling.landfall_s)

    out = out or f"data/interim/tide_{cfg.name}.txt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# NOAA CO-OPS station {cfg.forcing.tide_station} {product} NAVD88\n")
        f.write("# model_seconds    level_m\n")
        for s, v in zip(secs, levels):
            f.write(f"{s:.1f}\t{v:.4f}\n")
    print(f"wrote {out}: {len(secs)} pts, model-clock {secs[0]:.0f}..{secs[-1]:.0f} s")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="NOAA CO-OPS tide -> model-clock series")
    ap.add_argument("--config", required=True)
    ap.add_argument("--product", default="predictions", choices=["predictions", "water_level"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from coral import config
    from_config(config.load(a.config), a.out, product=a.product)
