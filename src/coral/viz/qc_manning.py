#!/usr/bin/env python3
"""
QC plot for the Manning's-n grid built by make_manning.py.

Shows the roughness map (and optionally the DEM coastline + the boundary points)
so you can eyeball that high n falls on marsh/forest and low n on water/developed,
and that the grid aligns with the DEM. Also prints a histogram of n values.

Deps: numpy, rasterio, matplotlib
Usage:
  python qc_manning.py --manning inputs/Manning_SAV.asc --dem inputs/SUB_DEM_SAV.asc \
      [--bci inputs/savannah_coastline.bci] [--out qc/manning_QC.png]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_grid(path):
    """Plain ESRI-ASCII reader (avoids GDAL AAIGrid driver quirks on .asc)."""
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
        a = np.loadtxt(f).astype(float)
    nod = hdr.get("nodata_value", -9999)
    a = np.where((a == nod) | (a <= -9990), np.nan, a)
    nx, ny, cs = int(hdr["ncols"]), int(hdr["nrows"]), hdr["cellsize"]
    x0, y0 = hdr["xllcorner"], hdr["yllcorner"]
    ext = [x0, x0 + nx * cs, y0, y0 + ny * cs]
    return a, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manning", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--bci", default=None)
    ap.add_argument("--out", default="qc/manning_QC.png")
    args = ap.parse_args()

    n, ext = read_grid(args.manning)
    z, _ = read_grid(args.dem)

    # boundary points (optional)
    blon, blat = [], []
    if args.bci and os.path.exists(args.bci):
        for line in open(args.bci):
            p = line.split()
            if p and p[0] == "P":
                blon.append(float(p[1])); blat.append(float(p[2]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # left: Manning's n
    im0 = ax[0].imshow(n, extent=ext, origin="upper", cmap="YlGnBu",
                       vmin=np.nanmin(n), vmax=np.nanmax(n))
    ax[0].contour(np.where(np.isfinite(z), z, 1e3), levels=[0.81],
                  extent=ext, origin="upper", colors="k", linewidths=0.4)
    fig.colorbar(im0, ax=ax[0], shrink=0.8, label="Manning's n")
    if blon:
        ax[0].scatter(blon, blat, s=4, c="red", label="BC points")
        ax[0].legend(loc="lower left", fontsize=8)
    ax[0].set_title("Manning's n (black = 0.81 m shoreline)")
    ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat")

    # right: histogram of n
    vals = n[np.isfinite(n)]
    ax[1].hist(vals, bins=40, color="teal")
    ax[1].set_title("Distribution of Manning's n")
    ax[1].set_xlabel("n"); ax[1].set_ylabel("cells"); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"n: min={np.nanmin(n):.3f} max={np.nanmax(n):.3f} "
          f"mean={np.nanmean(n):.3f}")
    # top values by frequency
    u, c = np.unique(np.round(vals, 4), return_counts=True)
    order = np.argsort(c)[::-1][:8]
    print("most common n values (n: cells):")
    for i in order:
        print(f"  {u[i]:.4f}: {int(c[i]):,}")


if __name__ == "__main__":
    main()
