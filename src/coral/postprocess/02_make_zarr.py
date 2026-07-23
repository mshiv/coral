#!/usr/bin/env python3
"""
Deliverable 2: Zarr time-series cube -- (time, y, x) water-depth stack for
animation and point queries.

Builds an xarray Dataset from a stack of LISFLOOD water-depth snapshots and
writes it to a chunked Zarr store with CF-style metadata.

LISFLOOD writes time-varying grids as `<resroot>-NNNN.wd` (water depth) when
`saveint` is small enough to produce multiple frames. The frame number NNNN
maps to simulation time via saveint (set in the .par).  ==> IMPORTANT: your
production run used saveint=302399 (one frame). To get a real cube, re-run with
a small saveint (e.g. 3600 s = hourly) so you get a series of .wd files. See
README for the one-line .par change.

Deps:  pip install xarray zarr numpy rioxarray
Usage:
    python 02_make_zarr.py --dir results_dir --root res_rain_Sorted_1198.bdy \
        --saveint 3600 --t0 64800 --out flood_depth.zarr
"""
import argparse
import glob
import os
import re

import numpy as np
import xarray as xr


def read_ascii_grid(path):
    header = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            header[k.lower()] = float(v)
        data = np.loadtxt(f).astype("float32")
    nx, ny = int(header["ncols"]), int(header["nrows"])
    cs = header["cellsize"]
    x0, y0 = header["xllcorner"], header["yllcorner"]
    nodata = header.get("nodata_value", -9999.0)
    data = np.where((data == nodata) | (data <= -9990), np.nan, data)
    # cell-center coordinates; rows run N->S (top row first)
    x = x0 + (np.arange(nx) + 0.5) * cs
    y = (y0 + ny * cs) - (np.arange(ny) + 0.5) * cs
    return data, x, y, nodata


def convert_netcdf(ncpath, out):
    """LISFLOOD-FP netcdf_out file (<resroot>.nc) -> chunked Zarr.

    The file has an unlimited `time` dim and a 3D `depth(time, dimY, dimX)`
    variable (deflate-compressed), with lat/lon (latlong runs) or x/y coords.
    We standardize to water_depth(time, y, x), EPSG:4326, CF-1.10.
    """
    ds = xr.open_dataset(ncpath)
    if "depth" not in ds:
        raise SystemExit(f"no 'depth' variable in {ncpath}; vars: {list(ds.data_vars)}")
    da = ds["depth"]
    spatial = [d for d in da.dims if d != "time"]   # (rows, cols) = (y, x)
    da = da.rename({spatial[0]: "y", spatial[1]: "x"}).rename("water_depth")
    # carry coordinate values if present (lat/lon or x/y), else leave index coords
    ren = {}
    for c in ds.coords:
        if c in ("lat",) or c == spatial[0]:
            ren[c] = "y"
        elif c in ("lon",) or c == spatial[1]:
            ren[c] = "x"
    da = da.rename({k: v for k, v in ren.items() if k in da.coords})
    da.attrs.update(long_name="water depth above ground", units="m",
                    standard_name="depth_of_surface_liquid_water",
                    grid_mapping="crs")
    out_ds = da.to_dataset()
    out_ds["crs"] = xr.DataArray(0, attrs={
        "grid_mapping_name": "latitude_longitude", "epsg_code": "EPSG:4326"})
    out_ds.attrs = {"title": "CHORUS flood depth time series (LISFLOOD-FP netcdf_out)",
                    "source": "LISFLOOD-FP 8.0.3", "Conventions": "CF-1.10"}
    chunks = {"time": 1, "y": min(512, out_ds.sizes["y"]), "x": min(512, out_ds.sizes["x"])}
    out_ds = out_ds.chunk(chunks)
    if os.path.exists(out):
        import shutil; shutil.rmtree(out)
    out_ds.to_zarr(out, mode="w", consolidated=True)
    print(f"wrote {out}  dims={dict(out_ds.sizes)}  chunks={chunks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netcdf", default=None,
                    help="LISFLOOD-FP netcdf_out file (<resroot>.nc); preferred path")
    ap.add_argument("--dir", help="(ASCII path) results dir")
    ap.add_argument("--root", help="(ASCII path) resroot prefix")
    ap.add_argument("--ext", default="wd", help="snapshot extension (wd, etc.)")
    ap.add_argument("--saveint", type=float, default=3600.0,
                    help="seconds between snapshots (from .par)")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="simulation time of first frame (s); often tstart")
    ap.add_argument("--out", default="flood_depth.zarr")
    args = ap.parse_args()

    if args.netcdf:
        convert_netcdf(args.netcdf, args.out)
        return
    if not (args.dir and args.root):
        raise SystemExit("ASCII mode needs --dir and --root (or use --netcdf)")

    base = os.path.join(args.dir, args.root)
    files = sorted(glob.glob(f"{base}-*.{args.ext}"))
    if not files:
        # fall back to the single peak grid so devs still get a (degenerate) cube
        single = f"{base}.max"
        if os.path.exists(single):
            print(f"No -NNNN.{args.ext} frames found; using single {single} "
                  f"as a 1-step cube (re-run with small saveint for a real series).")
            files = [single]
        else:
            raise SystemExit(f"no {base}-*.{args.ext} and no {single}")

    def frame_num(p):
        m = re.search(r"-(\d+)\.", os.path.basename(p))
        return int(m.group(1)) if m else 0

    files.sort(key=frame_num)
    arrs, times = [], []
    x = y = None
    for p in files:
        data, gx, gy, _ = read_ascii_grid(p)
        if x is None:
            x, y = gx, gy
        arrs.append(data)
        times.append(args.t0 + frame_num(p) * args.saveint)

    cube = np.stack(arrs, axis=0)  # (time, y, x)
    da = xr.DataArray(
        cube,
        dims=("time", "y", "x"),
        coords={"time": np.array(times, dtype="float64"), "y": y, "x": x},
        name="water_depth",
        attrs={
            "long_name": "water depth above ground",
            "units": "m",
            "standard_name": "depth_of_surface_liquid_water",
            "_FillValue": np.nan,
            "grid_mapping": "crs",
        },
    )
    ds = da.to_dataset()
    ds["time"].attrs = {"long_name": "simulation time", "units": "seconds"}
    ds["x"].attrs = {"long_name": "longitude", "units": "degrees_east",
                     "standard_name": "longitude"}
    ds["y"].attrs = {"long_name": "latitude", "units": "degrees_north",
                     "standard_name": "latitude"}
    # CF grid mapping for EPSG:4326
    ds["crs"] = xr.DataArray(0, attrs={
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": "EPSG:4326",
        "crs_wkt": 'GEOGCS["WGS 84",DATUM["WGS_1984"]]',
    })
    ds.attrs = {
        "title": "CHORUS coastal flood depth time series (LISFLOOD-FP)",
        "source": "LISFLOOD-FP 8.0.3, GeoClaw HVAR boundary forcing",
        "Conventions": "CF-1.10",
    }

    chunks = {"time": 1,
              "y": min(512, ds.dims["y"]),
              "x": min(512, ds.dims["x"])}
    ds = ds.chunk(chunks)
    enc = {"water_depth": {"chunks": tuple(chunks[d] for d in ("time", "y", "x"))}}
    if os.path.exists(args.out):
        import shutil; shutil.rmtree(args.out)
    ds.to_zarr(args.out, mode="w", encoding=enc, consolidated=True)
    print(f"wrote {args.out}  dims={dict(ds.dims)}  chunks={chunks}")


if __name__ == "__main__":
    main()
