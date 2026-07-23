"""Fetch AORC rainfall for a scenario's bbox + simulation window.

AORC v1.1 (NOAA Analysis of Record for Calibration) is hosted as yearly Zarr
stores on AWS Open Data, anonymous access:
  s3://noaa-nws-aorc-v1-1-1km/<YEAR>.zarr
We read the total-precip variable (APCP_surface, hourly accumulation in mm),
clip to the domain bbox and the [landfall-24h, landfall+24h] window, and write a
small netCDF to data/interim/ for make_rain.py to turn into LISFLOOD forcing.

Two-tier per repo convention:
  fetch_aorc(...)   -- pure worker (bbox, time window, out path)
  from_config(cfg)  -- adapter pulling bbox + rain_window_utc() from the config

Deps: xarray, zarr, s3fs (anonymous S3).
"""
from __future__ import annotations
import os
from pathlib import Path

AORC_BUCKET = "noaa-nws-aorc-v1-1-1km"   # us-east-1, anonymous
PRECIP_VAR = "APCP_surface"              # total precip, hourly accumulation (mm)


def fetch_aorc(bbox, t0, t1, out_path, *, var=PRECIP_VAR):
    """Clip AORC precip to bbox=[W,E,S,N] and [t0,t1] (datetimes); write netCDF.

    Returns the path written. Spans year boundaries by opening each year's Zarr.
    """
    import xarray as xr
    import s3fs

    W, E, S, N = bbox
    fs = s3fs.S3FileSystem(anon=True)
    years = sorted({t0.year, t1.year})
    stores = []
    for y in years:
        m = s3fs.S3Map(f"{AORC_BUCKET}/{y}.zarr", s3=fs, check=False)
        stores.append(xr.open_zarr(m, consolidated=True)[[var]])
    ds = xr.concat(stores, dim="time") if len(stores) > 1 else stores[0]

    # AORC coords are 'latitude'/'longitude'; latitude may be ascending.
    lat_asc = bool(ds.latitude[0] < ds.latitude[-1])
    ds = ds.sel(
        time=slice(t0, t1),
        longitude=slice(W, E),
        latitude=slice(S, N) if lat_asc else slice(N, S),
    )
    if ds.time.size == 0:
        raise SystemExit(f"AORC returned no timesteps for {t0}..{t1}")
    if ds.longitude.size == 0 or ds.latitude.size == 0:
        raise SystemExit(f"AORC returned no cells for bbox {bbox}; check W/E sign")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds = ds.load()
    ds.attrs["source"] = f"AORC v1.1 {AORC_BUCKET}"
    ds.to_netcdf(out_path)
    print(f"wrote {out_path}: {ds.time.size} hrs, "
          f"{ds.latitude.size}x{ds.longitude.size} cells, "
          f"max {float(ds[var].max()):.1f} mm/hr")
    return out_path


def from_config(cfg, out_path=None):
    """Adapter: bbox + window from config; dispatch by forcing.rainfall (aorc here,
    mrms -> fetch_mrms). Both write the same APCP_surface schema for make_rain."""
    if cfg.forcing.rainfall is None:
        raise SystemExit(f"{cfg.name}: forcing.rainfall is null (surge-only run)")
    if cfg.forcing.rainfall == "mrms":
        from .fetch_mrms import from_config as _mrms_from_config
        return _mrms_from_config(cfg, out_path)
    if cfg.forcing.rainfall != "aorc":
        raise SystemExit(f"unknown rainfall source {cfg.forcing.rainfall!r}")
    t0, t1 = cfg.rain_window_utc()
    out_path = out_path or f"data/interim/rain_aorc_{cfg.name}.nc"
    return fetch_aorc(cfg.domain.bbox, t0, t1, out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch AORC rainfall for a scenario")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from coral import config
    from_config(config.load(a.config), a.out)
