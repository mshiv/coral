#!/usr/bin/env python3
"""
Animate LISFLOOD-FP flood depth over time from the netcdf_out file (<resroot>.nc).

Reads the 3D `depth(time, y, x)` variable and writes an mp4 (or gif) with the
flood depth over a faint DEM backdrop. Dry/zero cells are transparent so the
land shows through.

Deps: xarray, numpy, matplotlib (mp4 needs ffmpeg; gif works without).
Usage:
  python 04_animate_flood.py --netcdf results_matthew_sav/res_matthew_sav.nc \
      --dem ../inputs/SUB_DEM_SAV.asc --out flood.mp4 [--fps 6] [--depth-floor 0.05]
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
try:
    import contextily as ctx
    HAVE_CTX = True
except ImportError:
    HAVE_CTX = False

PIN = (-81.0903, 31.9522)   # Pin Point, Savannah
SAV_BOX = (-81.111, -80.819, 31.804, 32.100)   # Pin Point / coupling box


def read_ascii_grid(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.where((a == h.get("nodata_value", -9999)) | (a <= -9990), np.nan, a)
    nx, ny, cs = int(h["ncols"]), int(h["nrows"]), h["cellsize"]
    return a, [h["xllcorner"], h["xllcorner"] + nx * cs,
               h["yllcorner"], h["yllcorner"] + ny * cs]


def load_netcdf(path):
    ds = xr.open_dataset(path)
    depth = ds["depth"]
    spatial = [d for d in depth.dims if d != "time"]
    depth = depth.transpose("time", *spatial)
    cube = depth.values.astype("float32")
    cube = np.where(cube <= -9990, np.nan, cube)
    try:
        yc = ds[spatial[0]].values; xc = ds[spatial[1]].values
        extent = [xc.min(), xc.max(), yc.min(), yc.max()]
    except Exception:
        extent = None
    tvals = ds["time"].values if "time" in ds else np.arange(cube.shape[0])
    return cube, extent, np.asarray(tvals, dtype=float)


def load_ascii(d, root, ext, saveint, t0):
    import glob, re, os
    base = os.path.join(d, root)
    files = sorted(glob.glob(f"{base}-*.{ext}"),
                   key=lambda p: int(re.search(r"-(\d+)\.", os.path.basename(p)).group(1)))
    if not files:
        raise SystemExit(f"no {base}-*.{ext} frames found")
    arrs, times, extent = [], [], None
    for p in files:
        a, ext_ = read_ascii_grid(p)
        if extent is None:
            extent = ext_
        arrs.append(a)
        n = int(re.search(r"-(\d+)\.", os.path.basename(p)).group(1))
        times.append(t0 + n * saveint)
    return np.stack(arrs).astype("float32"), extent, np.array(times, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netcdf", default=None, help="LISFLOOD .nc (if finalized OK)")
    ap.add_argument("--dir", default=None, help="ASCII: results dir")
    ap.add_argument("--root", default=None, help="ASCII: resroot prefix")
    ap.add_argument("--ext", default="wd", help="ASCII frame extension (wd/wdfp)")
    ap.add_argument("--saveint", type=float, default=1800.0, help="ASCII: seconds/frame")
    ap.add_argument("--dem", default=None, help="DEM .asc for the backdrop")
    ap.add_argument("--out", default="flood.mp4")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--depth-floor", type=float, default=0.05,
                    help="hide depths below this (m)")
    ap.add_argument("--t0", type=float, default=86400.0,
                    help="sim time (s) of frame 0 (shifted clock; tstart)")
    ap.add_argument("--landfall", type=float, default=172800.0,
                    help="sim time (s) of landfall on the shifted clock")
    ap.add_argument("--sea-level", type=float, default=0.81,
                    help="DEM elevation above which a cell is land (for masking)")
    ap.add_argument("--basemap", action="store_true",
                    help="satellite backdrop (Esri WorldImagery) instead of DEM")
    ap.add_argument("--zoom", action="store_true",
                    help="zoom to the Pin Point / Savannah box")
    args = ap.parse_args()

    if args.netcdf:
        cube, extent, tvals = load_netcdf(args.netcdf)
    elif args.dir and args.root:
        cube, extent, tvals = load_ascii(args.dir, args.root, args.ext,
                                         args.saveint, args.t0)
    else:
        raise SystemExit("give --netcdf OR (--dir --root)")
    nframes = cube.shape[0]
    print(f"{nframes} frames loaded")

    dem = dem_ext = None
    if args.dem:
        dem, dem_ext = read_ascii_grid(args.dem)
        if extent is None:
            extent = dem_ext

    # mask to LAND so ocean/channel depths don't dominate the color scale
    if dem is not None:
        land = dem > args.sea_level
        cube = np.where(land[None, :, :], cube, np.nan)
        vmax = float(np.nanpercentile(cube[np.isfinite(cube)], 99)) if np.isfinite(cube).any() else 2.0
    else:
        vmax = float(np.nanpercentile(cube, 99))
    vmax = max(vmax, 0.3)
    print(f"land flood color scale: {args.depth_floor}-{vmax:.2f} m")

    fig, ax = plt.subplots(figsize=(10, 8))
    # view window: zoom to Pin Point box or full domain
    if args.zoom:
        m = 0.02
        view = (SAV_BOX[0] - m, SAV_BOX[1] + m, SAV_BOX[2] - m, SAV_BOX[3] + m)
    else:
        view = (extent[0], extent[1], extent[2], extent[3])
    ax.set_xlim(view[0], view[1]); ax.set_ylim(view[2], view[3])
    # static backdrop (drawn once): satellite if requested, else faint DEM
    if args.basemap and HAVE_CTX:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", zorder=0,
                            source=ctx.providers.Esri.WorldImagery)
        except Exception as e:
            print(f"basemap skipped ({e})")
    elif dem is not None:
        ax.imshow(dem, extent=dem_ext, origin="upper", cmap="Greys",
                  norm=TwoSlopeNorm(vmin=max(np.nanmin(dem), -6), vcenter=0,
                                    vmax=min(np.nanmax(dem), 8)), alpha=0.4, zorder=0)

    d0 = np.where(cube[0] < args.depth_floor, np.nan, cube[0])
    im = ax.imshow(d0, extent=extent, origin="upper", cmap="Blues",
                   vmin=args.depth_floor, vmax=vmax, zorder=2)
    ax.scatter(*PIN, s=120, marker="*", c="red", ec="k", zorder=5, label="Pin Point")
    ax.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("flood depth on land (m)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_xlim(view[0], view[1]); ax.set_ylim(view[2], view[3])  # imshow can reset; re-apply
    title = ax.set_title("")

    def update(k):
        d = cube[k]
        im.set_data(np.where(d < args.depth_floor, np.nan, d))
        hrs = (float(tvals[k]) - args.landfall) / 3600.0 + 6.0
        title.set_text(f"Hurricane Matthew flood depth — Savannah   "
                       f"({hrs:+.1f} h from closest approach)")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=nframes, blit=False)
    if args.out.endswith(".gif"):
        anim.save(args.out, writer="pillow", fps=args.fps)
    else:
        anim.save(args.out, fps=args.fps, dpi=130)
    print(f"wrote {args.out}  ({nframes} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
