"""Fetch MRMS GaugeCorr_QPE_01H rainfall for a scenario's bbox + window.

MRMS (Multi-Radar Multi-Sensor) QPE is radar+gauge precipitation at ~1 km
(0.01 deg, WGS84) and hourly accumulation — higher spatial/temporal fidelity than
AORC for convective / hurricane rainbands. For Hurricane Matthew (Oct 2016) the
data live in the Iowa Environmental Mesonet historical mirror (the AWS
noaa-mrms-pds bucket is only a ~30-day rolling real-time archive):

  https://mtarchive.geol.iastate.edu/YYYY/MM/DD/mrms/ncep/GaugeCorr_QPE_01H/
      GaugeCorr_QPE_01H_00.00_YYYYMMDD-HH0000.grib2.gz

GaugeCorr_QPE_01H (gauge-bias-corrected) is the best land product for 2016
(MultiSensor Pass2 postdates this era). Output matches fetch_rainfall (AORC):
APCP_surface(time, latitude, longitude) in mm, so make_rain works unchanged.

Caveats (see the CORAL notes): coastal/marsh gauge sparsity + radar beam
overshoot at range from KCLX/KVAX can undercatch tropical rain; 2016 predates
later MRMS QPE algorithm upgrades. Cross-check against AORC / Stage IV.

Deps: xarray, cfgrib (eccodes), numpy. conda install -c conda-forge cfgrib eccodes
"""
from __future__ import annotations
import gzip, os, tempfile, urllib.request
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

IEM = "https://mtarchive.geol.iastate.edu"
PRODUCT = "GaugeCorr_QPE_01H"


def _hours(t0, t1):
    h0 = t0.replace(minute=0, second=0, microsecond=0)
    out, t = [], h0
    while t <= t1:
        out.append(t); t += timedelta(hours=1)
    return out


def _url(dt):
    return (f"{IEM}/{dt:%Y}/{dt:%m}/{dt:%d}/mrms/ncep/{PRODUCT}/"
            f"{PRODUCT}_00.00_{dt:%Y%m%d}-{dt:%H}0000.grib2.gz")


def _read_hour(dt, tmp, bbox):
    """Download one hourly GRIB2, clip to bbox=[W,E,S,N], return a DataArray (or None)."""
    import xarray as xr
    W, E, S, N = bbox
    gz = os.path.join(tmp, "m.grib2.gz"); gr = os.path.join(tmp, "m.grib2")
    try:
        urllib.request.urlretrieve(_url(dt), gz)
        with gzip.open(gz, "rb") as fi, open(gr, "wb") as fo:
            fo.write(fi.read())
    except Exception as e:
        print(f"  {dt:%Y-%m-%d %HZ}: fetch failed ({e})"); return None
    ds = xr.open_dataset(gr, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]                       # MRMS = one field per file
    if float(da.longitude.max()) > 180:                  # 0-360 -> -180..180
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")
    da = da.sortby("latitude")
    da = da.sel(longitude=slice(W, E), latitude=slice(S, N))
    da = da.where(da >= 0, 0.0)                           # MRMS -3 no-coverage / -1 missing -> 0
    return da.assign_coords(time=dt)


def fetch_mrms(bbox, t0, t1, out_path):
    """Clip MRMS GaugeCorr_QPE_01H to bbox + [t0,t1]; write AORC-schema netCDF."""
    import xarray as xr
    tmp = tempfile.mkdtemp()
    das = [d for d in (_read_hour(h, tmp, bbox) for h in _hours(t0, t1)) if d is not None]
    if not das:
        raise SystemExit(f"MRMS returned no hours for {t0}..{t1}")
    da = xr.concat(das, dim="time").rename("APCP_surface")
    da.attrs.update(units="mm", long_name="MRMS GaugeCorr_QPE_01H hourly accumulation")
    ds = da.to_dataset()
    ds.attrs["source"] = f"MRMS {PRODUCT} (IEM mtarchive)"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"wrote {out_path}: {ds.time.size} hrs, {ds.latitude.size}x{ds.longitude.size} "
          f"cells, max {float(ds.APCP_surface.max()):.1f} mm/hr")
    return out_path


def from_config(cfg, out_path=None):
    """Adapter: bbox from domain, time window from rain_window_utc()."""
    if cfg.forcing.rainfall != "mrms":
        raise SystemExit(f"{cfg.name}: forcing.rainfall is {cfg.forcing.rainfall!r}, not mrms")
    t0, t1 = cfg.rain_window_utc()
    out_path = out_path or f"data/interim/rain_mrms_{cfg.name}.nc"
    return fetch_mrms(cfg.domain.bbox, t0, t1, out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch MRMS rainfall for a scenario")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from coral import config
    from_config(config.load(a.config), a.out)
