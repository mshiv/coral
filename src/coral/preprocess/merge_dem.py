"""Composite a lidar DEM with a topobathymetric one.

Airborne lidar at 1064 nm does not penetrate water, so tidal channels carry no ground returns and
gap-filling raises them to the surrounding marsh level. Over the Pin Point clip this leaves 1.4% of
cells at or below -1 m against 13.9% in CoNED: the creek network, which is the conveyance path that
drains the marsh, is missing.

CoNED is topobathymetric, lidar topography merged with bathymetric survey, so it has the channels.
It is coarser on land.

Each source is used where it is valid: bathymetry from the topobathy product below the water
threshold, lidar elsewhere. The topobathy DEM is resampled to the lidar grid, so channel geometry
carries its own resolution and is not improved by the merge.
"""
import argparse

import numpy as np

from .marsh_corrections import read_asc, write_asc


def merge(lidar_asc, topobathy_asc, out, water_thresh=-0.5, nodata=-9999.0, floor=None):
    lid, hl = read_asc(lidar_asc)
    bat, hb = read_asc(topobathy_asc)

    # resample the coarse grid onto the fine one by cell-centre lookup
    nr, nc, cs = hl["nrows"], hl["ncols"], hl["cellsize"]
    x = hl["xllcorner"] + (np.arange(nc) + 0.5) * cs
    y = (hl["yllcorner"] + nr * cs) - (np.arange(nr) + 0.5) * cs
    bx = hb["xllcorner"] + (np.arange(hb["ncols"]) + 0.5) * hb["cellsize"]
    by = (hb["yllcorner"] + hb["nrows"] * hb["cellsize"]) - (np.arange(hb["nrows"]) + 0.5) * hb["cellsize"]
    i = np.clip(np.searchsorted(bx, x) - 1, 0, len(bx) - 1)
    j = np.clip(np.searchsorted(-by, -y) - 1, 0, len(by) - 1)
    bat_on_fine = bat[np.ix_(j, i)]

    lid = np.where(lid == nodata, np.nan, lid)
    bat_on_fine = np.where(bat_on_fine == nodata, np.nan, bat_on_fine)

    use_bat = np.isfinite(bat_on_fine) & (bat_on_fine < water_thresh)
    dem = np.where(use_bat, bat_on_fine, lid)
    dem = np.where(np.isfinite(dem), dem, bat_on_fine)     # lidar gaps fall back to topobathy
    if floor is not None:
        dem = np.where(np.isfinite(dem), np.maximum(dem, floor), dem)
    dem = np.where(np.isfinite(dem), dem, nodata)

    write_asc(out, dem, hl, nodata, fmt="%.3f")
    v = dem[dem > nodata]
    print(f"merged -> {out}: {hl['ncols']}x{hl['nrows']}, {v.min():.2f} to {v.max():.2f} m")
    print(f"  bathymetry used on {100*use_bat.mean():.1f}% of cells (below {water_thresh} m)")
    print(f"  at/below -1 m: {100*np.mean(v <= -0.99):.1f}%   median {np.median(v):.2f} m")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lidar", required=True, help="lidar-derived DEM, fine grid")
    ap.add_argument("--topobathy", required=True, help="topobathymetric DEM with channels")
    ap.add_argument("--water-thresh", type=float, default=-0.5,
                    help="use bathymetry below this elevation (m)")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a.lidar, a.topobathy, a.out, a.water_thresh, floor=a.floor)


if __name__ == "__main__":
    main()
