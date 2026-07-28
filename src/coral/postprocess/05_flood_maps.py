#!/usr/bin/env python3
"""
Headline static flood figures from LISFLOOD-FP summary grids (.max/.mxe/.inittm/
.totaltm). Masks to LAND (DEM > sea level) so the maps show inland inundation,
not the permanently-wet ocean/channels.

Makes a 4-panel summary + a Pin Point zoom of peak depth.

Deps: numpy, matplotlib
Usage:
  python 05_flood_maps.py --dir ../lisflood/results_matthew_sav --root res_matthew_sav \
      --dem ../inputs/SUB_DEM_SAV.asc [--sea-level 0.81]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


PIN = (-81.0903, 31.9522)
SAV_BOX = (-81.111, -80.819, 31.804, 32.100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--sea-level", type=float, default=0.81,
                    help="DEM elevation above which a cell counts as land")
    ap.add_argument("--depth-floor", type=float, default=0.05)
    ap.add_argument("--out-prefix", default="fig_flood")
    args = ap.parse_args()

    base = os.path.join(args.dir, args.root)
    dem, ext = read_grid(args.dem)
    land = dem > args.sea_level

    def land_only(a, floor=None):
        a = np.where(land, a, np.nan)
        if floor is not None:
            a = np.where(a < floor, np.nan, a)
        return a

    mx, _   = read_grid(base + ".max")
    mxe, _  = read_grid(base + ".mxe")
    itm, _  = read_grid(base + ".inittm")     # arrival time (hours)
    ttm, _  = read_grid(base + ".totaltm")    # duration (hours)

    depth = land_only(mx, args.depth_floor)
    elev  = land_only(mxe, None)
    arr   = land_only(itm, None)
    dur   = land_only(ttm, args.depth_floor)

    # ---- 4-panel summary ----
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    panels = [
        (ax[0, 0], depth, "Peak flood depth (m)", "Blues",
         np.nanpercentile(depth, 99) if np.isfinite(depth).any() else 1),
        (ax[0, 1], elev, "Max water-surface elev (m)", "viridis", None),
        (ax[1, 0], arr, "Arrival time (h)", "magma_r", None),
        (ax[1, 1], dur, "Inundation duration (h)", "cividis", None),
    ]
    for a, data, title, cmap, vmax in panels:
        # faint land backdrop
        a.imshow(np.where(land, dem, np.nan), extent=ext, origin="upper",
                 cmap="Greys", alpha=0.25, zorder=0)
        im = a.imshow(data, extent=ext, origin="upper", cmap=cmap,
                      vmax=vmax, zorder=2)
        a.scatter(*PIN, s=120, marker="*", c="red", ec="k", zorder=5)
        a.set_title(title, fontsize=13); a.set_xlabel("lon"); a.set_ylabel("lat")
        plt.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle("Hurricane Matthew flood — Savannah (matthew_SAV_LR, LISFLOOD-FP)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out1 = f"{args.out_prefix}_summary.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight"); print("wrote", out1)

    # ---- Pin Point zoom of peak depth ----
    fig2, a2 = plt.subplots(figsize=(9, 8))
    a2.imshow(np.where(land, dem, np.nan), extent=ext, origin="upper",
              cmap="Greys", alpha=0.3, zorder=0)
    im2 = a2.imshow(depth, extent=ext, origin="upper", cmap="Blues",
                    vmax=np.nanpercentile(depth, 99), zorder=2)
    a2.scatter(*PIN, s=160, marker="*", c="red", ec="k", zorder=5, label="Pin Point")
    a2.set_xlim(SAV_BOX[0] - 0.03, SAV_BOX[1] + 0.03)
    a2.set_ylim(SAV_BOX[2] - 0.03, SAV_BOX[3] + 0.03)
    plt.colorbar(im2, ax=a2, shrink=0.8, label="peak flood depth (m)")
    a2.set_title("Peak flood depth — Pin Point / Savannah", fontsize=14)
    a2.set_xlabel("lon"); a2.set_ylabel("lat"); a2.legend(loc="upper right")
    out2 = f"{args.out_prefix}_pinpoint.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); print("wrote", out2)

    # stats
    fin = depth[np.isfinite(depth)]
    print(f"LAND flood: {fin.size:,} cells > {args.depth_floor} m, "
          f"max {np.nanmax(depth):.2f} m, mean {np.nanmean(depth):.2f} m, "
          f"area ~{fin.size * 30 * 30 / 1e6:.1f} km^2 (at 30 m)")


if __name__ == "__main__":
    main()
