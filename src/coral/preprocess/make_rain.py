"""Turn fetched AORC precip into LISFLOOD rainfall forcing.

Two LISFLOOD input modes (this build, verified against source):

  uniform  -> a 'rainfile' time series. Rate is in mm/hr (the HPC LISFLOOD build
              converts mm/hr->m/s internally, factor 3.6e6 -- verified empirically;
              writing m/s made rain 3.6e6x too weak). Times are model-clock SECONDS.
              File format (LoadTimeSeries, skipFirstLine=ON):
                  <label line, skipped>
                  <N>  seconds
                  <rate_m_s>  <time_s>
                  ...                       (times strictly increasing)

  dynamic  -> a 'dynamicrainfile' netCDF with variable 'rainfall_depth' = mm
              accumulated over each interval; rain.tpp converts (/dt, /1000) to
              m/s. Time axis in HOURS on the model clock. The rain grid must
              share the DEM origin and have a cell size that is an integer
              multiple of the DEM cell -> we resample onto the exact DEM grid.

All times are placed on the same model clock as the surge .bdy and .par:
    t_model_s = (utc - landfall_utc) + landfall_s
so rain, surge, and sim_time share the landfall origin.

Deps: xarray, numpy; rasterio (dynamic mode only).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _to_model_seconds(times_utc, landfall_dt, landfall_s):
    """Vectorised: pandas/xarray datetimes -> model-clock seconds."""
    import pandas as pd
    lf = pd.Timestamp(landfall_dt)
    secs = (pd.to_datetime(times_utc) - lf).total_seconds().to_numpy()
    return secs + landfall_s


def make_uniform(rain_nc, out_txt, landfall_dt, landfall_s, *, var="APCP_surface"):
    """Domain-mean hyetograph -> LISFLOOD rainfile (rate m/s, time s)."""
    import xarray as xr
    ds = xr.open_dataset(rain_nc)
    rate_mm_hr = ds[var].mean(dim=("latitude", "longitude")).values  # mm/hr
    # LISFLOOD-FP rainfile rate is in mm/hr (the build converts mm/hr->m/s
    # internally, factor 3.6e6). Verified empirically: writing m/s made rain
    # 3.6e6x too weak. Do NOT pre-convert to m/s here.
    t_s = _to_model_seconds(ds.time.values, landfall_dt, landfall_s)

    order = np.argsort(t_s)
    t_s, rate_mm_hr = t_s[order], rate_mm_hr[order]
    keep = np.concatenate(([True], np.diff(t_s) > 0))                # strictly increasing
    t_s, rate_mm_hr = t_s[keep], rate_mm_hr[keep]

    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write("# AORC domain-mean rainfall (rate mm/hr, time model-seconds)\n")
        f.write(f"{len(t_s)}\tseconds\n")
        for r, t in zip(rate_mm_hr, t_s):
            f.write(f"{r:.5f}\t{t:.1f}\n")
    print(f"wrote {out_txt}: {len(t_s)} steps, peak "
          f"{rate_mm_hr.max():.1f} mm/hr, t {t_s.min():.0f}-{t_s.max():.0f}s")
    return out_txt


def make_dynamic(rain_nc, dem_path, out_nc, landfall_dt, landfall_s,
                 *, var="APCP_surface"):
    """Resample AORC onto the DEM grid -> netCDF 'rainfall_depth' (mm/interval),
    time in model-clock hours, origin matching the DEM (LISFLOOD requirements)."""
    import xarray as xr
    import rasterio
    from rasterio.transform import xy

    with rasterio.open(dem_path) as s:
        tr, nx, ny = s.transform, s.width, s.height
    xs = np.array([xy(tr, 0, i)[0] for i in range(nx)])   # cell-centre lon
    ys = np.array([xy(tr, j, 0)[1] for j in range(ny)])   # cell-centre lat (descending)

    ds = xr.open_dataset(rain_nc)
    rain = ds[var].interp(longitude=("x", xs), latitude=("y", ys),
                          method="linear", kwargs={"fill_value": None})
    rain = rain.transpose("time", "y", "x").values.astype("f4")  # (time, y, x)
    rain = np.nan_to_num(rain, nan=0.0)   # edge cells beyond AORC centers -> 0 rain
    # AORC hourly accumulation (mm) IS the per-interval depth; keep as mm.
    t_hr = (_to_model_seconds(ds.time.values, landfall_dt, landfall_s) / 3600.0).astype("f4")

    # Write with netCDF4 directly, mirroring LISFLOOD's own reference generator
    # (testing/T032_DynamicRain_SGM/generate_rain_netcdf.py): dims time/y/x read
    # BY NAME, coords with axis+units attrs, float32 time & rainfall, float64 x/y,
    # rainfall_depth ordered (time, y, x), y descending. No xarray _FillValue cruft.
    import netCDF4
    Path(out_nc).parent.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(out_nc, "w", format="NETCDF4_CLASSIC")
    nc.createDimension("time", len(t_hr))
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)
    vt = nc.createVariable("time", "f4", ("time",)); vt.units = "hour"; vt.axis = "T"
    vx = nc.createVariable("x", "f8", ("x",)); vx.units = "degrees_east"; vx.axis = "X"
    vy = nc.createVariable("y", "f8", ("y",)); vy.units = "degrees_north"; vy.axis = "Y"
    vr = nc.createVariable("rainfall_depth", "f4", ("time", "y", "x")); vr.units = "mm"
    vt[:] = t_hr; vx[:] = xs; vy[:] = ys; vr[:, :, :] = rain
    nc.close()
    print(f"wrote {out_nc}: {len(t_hr)} hrs on DEM grid {ny}x{nx} (NETCDF4_CLASSIC), "
          f"t {t_hr.min():.1f}-{t_hr.max():.1f} hr, peak {float(rain.max()):.1f} mm")
    return out_nc


def from_config(cfg, rain_nc=None, out=None):
    """Adapter: choose mode from cfg.forcing.rain_mode; share the landfall clock."""
    rain_nc = rain_nc or f"data/interim/rain_{cfg.forcing.rainfall}_{cfg.name}.nc"
    if cfg.landfall_dt is None:
        raise SystemExit("coupling.landfall_utc required to time-align rainfall")
    if cfg.forcing.rain_mode == "uniform":
        out = out or f"data/interim/rain_{cfg.name}.txt"
        return make_uniform(rain_nc, out, cfg.landfall_dt, cfg.coupling.landfall_s)
    else:  # dynamic
        out = out or f"data/interim/rain_{cfg.name}.nc"
        return make_dynamic(rain_nc, cfg.domain.dem, out,
                            cfg.landfall_dt, cfg.coupling.landfall_s)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AORC netCDF -> LISFLOOD rainfall")
    ap.add_argument("--config", required=True)
    ap.add_argument("--rain-nc", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from coral import config
    from_config(config.load(a.config), a.rain_nc, a.out)
