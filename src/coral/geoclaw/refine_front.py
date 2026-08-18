"""Build a GeoClaw ruled rectangle that follows the coupling gauge front.

The coupling gauges sit on a curve running north to south along the seaward edge of the 30 m
domain. Refining the bounding box of that curve refines mostly open ocean that no gauge reads.
The box is about 33 by 63 km; a band a few km wide around the curve is several times smaller,
and at level 7 the saving is the difference between a run that finishes and one that does not.

A ruled rectangle with `ixy='y'` is exactly this shape: for each latitude, one longitude range.
That is the same structure `kml2slu` produces for the Mayport and Pulaski regions, so the
downstream handling in setrun is unchanged.

    python -m coral.geoclaw.refine_front --csv inputs/boundary_points.csv --half-width-km 2.5
"""
from __future__ import annotations

import argparse

import numpy as np

EARTH_KM_PER_DEG = 111.32


def front_slu(csv_path, half_width_km=2.5, lat_pad_deg=0.02):
    """(n, 3) array of [lat, lon_west, lon_east] for RuledRectangle(slu=...).

    Rows are one per distinct gauge latitude, sorted south to north. Width is converted from km
    at each row's own latitude, so the band keeps a constant physical width rather than a
    constant angular one. `lat_pad_deg` extends the first and last rows so the band does not
    stop exactly on the end gauges.
    """
    rows = np.atleast_2d(np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                                       usecols=(2, 3)))
    lon, lat = rows[:, 0], rows[:, 1]
    if lon.size == 0:
        raise SystemExit(f"refine_front: no gauges read from {csv_path}")

    # Several gauges can share a latitude where the front doubles back. Take the full longitude
    # span at each latitude so the band covers all of them.
    order = np.argsort(lat)
    lat, lon = lat[order], lon[order]
    uniq, idx = np.unique(lat, return_inverse=True)
    west = np.full(uniq.size, np.inf)
    east = np.full(uniq.size, -np.inf)
    np.minimum.at(west, idx, lon)
    np.maximum.at(east, idx, lon)

    half_deg = half_width_km / (EARTH_KM_PER_DEG * np.cos(np.radians(uniq)))
    slu = np.column_stack([uniq, west - half_deg, east + half_deg])

    # Pad the ends by repeating the terminal rows outward. method=1 interpolates linearly
    # between rows, so without this the band tapers to nothing at the first and last gauge.
    first, last = slu[0].copy(), slu[-1].copy()
    first[0] -= lat_pad_deg
    last[0] += lat_pad_deg
    return np.vstack([first, slu, last])


def write(csv_path, out_path, half_width_km=2.5):
    """Write the ruled rectangle .data file. Returns the slu for inspection."""
    from clawpack.amrclaw import region_tools
    slu = front_slu(csv_path, half_width_km)
    rr = region_tools.RuledRectangle(slu=slu)
    rr.ixy = "y"
    rr.method = 1
    rr.write(out_path)
    return slu


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="boundary_points.csv")
    ap.add_argument("--half-width-km", type=float, default=2.5)
    ap.add_argument("--out", default="RuledRectangle_front.data")
    a = ap.parse_args()
    slu = write(a.csv, a.out, a.half_width_km)

    lat0, lat1 = slu[0, 0], slu[-1, 0]
    wid = (slu[:, 2] - slu[:, 1]) * EARTH_KM_PER_DEG * np.cos(np.radians(slu[:, 0]))
    box_km2 = ((slu[:, 2].max() - slu[:, 1].min()) * EARTH_KM_PER_DEG
               * np.cos(np.radians(slu[:, 0].mean())) * (lat1 - lat0) * EARTH_KM_PER_DEG)
    band_km2 = float(np.trapezoid(wid, slu[:, 0]) * EARTH_KM_PER_DEG)
    print(f"wrote {a.out}: {slu.shape[0]} rows, lat {lat0:.4f} to {lat1:.4f}")
    print(f"  band width {wid.min():.1f} to {wid.max():.1f} km")
    print(f"  band {band_km2:.0f} km2 against bounding box {box_km2:.0f} km2 "
          f"({box_km2 / max(band_km2, 1e-9):.1f}x smaller)")
    print("  AMR adds buffer cells and enforces proper nesting, so the realised saving will be "
          "less than this ratio.")


if __name__ == "__main__":
    main()
