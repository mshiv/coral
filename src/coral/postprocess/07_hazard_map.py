"""
Flood hazard map: hazard to people = depth x velocity (the standard DV product),
optionally the UK FD2320 hazard rating HR = d*(v + 0.5) + DF.

Needs a velocity grid. LISFLOOD only writes velocity/hazard to ASCII if the .par
has `voutput_max` (-> .maxVc) and/or `hazard` (-> .maxHaz). If those grids are
absent, re-run with those keywords (see note below). If .maxHaz exists, we plot
it directly; else hazard = .max(depth) x .maxVc(velocity).

  add to savannah.par:   voutput_max    and    hazard
  then re-run LISFLOOD (the .maxVc / .maxHaz grids will appear)

Deps: numpy, matplotlib
Usage:
  python 07_hazard_map.py --dir ../lisflood/results_matthew_sav --root res_matthew_sav \
      --dem ../inputs/SUB_DEM_SAV.asc --out fig_hazard.png
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

PIN = (-81.0903, 31.9522)


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
    ap.add_argument("--dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", default="fig_hazard.png")
    ap.add_argument("--sea-level", type=float, default=0.81)
    args = ap.parse_args()
    base = os.path.join(args.dir, args.root)

    dem, ext = read_grid(args.dem)
    land = dem > args.sea_level

    haz_path = base + ".maxHaz"
    v_path = base + ".maxVc"
    if os.path.exists(haz_path):
        haz, _ = read_grid(haz_path)
        src = "LISFLOOD .maxHaz"
    elif os.path.exists(v_path):
        depth, _ = read_grid(base + ".max")
        vel, _ = read_grid(v_path)
        haz = depth * vel                       # DV product (m^2/s)
        src = "depth x velocity (.max x .maxVc)"
    else:
        raise SystemExit(
            "no velocity/hazard grid found.\n"
            "Add `voutput_max` and `hazard` to savannah.par and re-run LISFLOOD,\n"
            "then this script will find .maxVc / .maxHaz.")

    haz = np.where(land & (haz > 0), haz, np.nan)

    # DV hazard classes (m^2/s): low/moderate/significant/extreme (after common
    # flood-hazard-to-people guidance; adjust thresholds to your standard)
    bounds = [0, 0.25, 0.5, 1.0, 1e9]
    labels = ["Low (<0.25)", "Moderate (0.25-0.5)", "Significant (0.5-1.0)", "Extreme (>1.0)"]
    colors = ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"]
    cmap = ListedColormap(colors); norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(np.where(land, dem, np.nan), extent=ext, origin="upper",
              cmap="Greys", alpha=0.3, zorder=0)
    im = ax.imshow(haz, extent=ext, origin="upper", cmap=cmap, norm=norm, zorder=2)
    ax.scatter(*PIN, s=150, marker="*", c="blue", ec="k", zorder=5, label="Pin Point")
    cb = fig.colorbar(im, ax=ax, ticks=[0.125, 0.375, 0.75, 2], shrink=0.8)
    cb.ax.set_yticklabels(labels)
    cb.set_label("flood hazard (depth x velocity, m^2/s)")
    ax.set_title(f"Flood hazard — Savannah, Matthew 2016\n({src})", fontsize=13)
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(loc="upper right")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)
    fin = haz[np.isfinite(haz)]
    print(f"hazard cells {fin.size:,}, max {fin.max():.2f}, "
          f"extreme (>1) {int((fin>1).sum()):,}")


if __name__ == "__main__":
    main()
