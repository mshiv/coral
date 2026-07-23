#!/usr/bin/env python3
"""
Peak flood depth over a satellite basemap (publication-grade). Land-masked depth
in Blues over Esri WorldImagery, so you see WHAT is flooded (roads, neighborhoods).

Deps: numpy, matplotlib, contextily
Usage:
  python 06_flood_map_satellite.py --max ../lisflood/results_matthew_sav/res_matthew_sav.max \
      --dem ../inputs/SUB_DEM_SAV.asc --out fig_flood_satellite.png [--zoom]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import contextily as ctx
    HAVE = True
except ImportError:
    HAVE = False

PIN = (-81.137, 31.944)
SAV_BOX = (-81.111, -80.819, 31.804, 32.100)


def read_grid(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.where((a == h.get("nodata_value", -9999)) | (a <= -9990), np.nan, a)
    nx, ny, cs = int(h["ncols"]), int(h["nrows"]), h["cellsize"]
    ext = [h["xllcorner"], h["xllcorner"] + nx * cs,
           h["yllcorner"], h["yllcorner"] + ny * cs]
    return a, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", default="fig_flood_satellite.png")
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--depth-floor", type=float, default=0.05)
    ap.add_argument("--zoom", action="store_true", help="zoom to Pin Point box")
    args = ap.parse_args()

    depth, ext = read_grid(args.max)
    dem, _ = read_grid(args.dem)
    d = np.where((dem > args.sea_level) & (depth >= args.depth_floor), depth, np.nan)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(d, extent=ext, origin="upper", cmap="Blues",
                   vmin=args.depth_floor, vmax=np.nanpercentile(d, 99),
                   alpha=0.85, zorder=2)
    ax.scatter(*PIN, s=160, marker="*", c="red", ec="k", zorder=5, label="Pin Point")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("peak flood depth (m)")
    if args.zoom:
        ax.set_xlim(SAV_BOX[0] - 0.02, SAV_BOX[1] + 0.02)
        ax.set_ylim(SAV_BOX[2] - 0.02, SAV_BOX[3] + 0.02)
    else:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    if HAVE:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", zorder=1,
                            source=ctx.providers.Esri.WorldImagery)
        except Exception as e:
            print(f"basemap skipped ({e})")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(loc="upper right")
    ax.set_title("Hurricane Matthew peak flood depth — Savannah (LISFLOOD-FP)",
                 fontsize=13, fontweight="bold")
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
