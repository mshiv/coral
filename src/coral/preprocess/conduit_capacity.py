"""Per-inlet drainage capacity from SAGIS conduit geometry.

make_drainage_proxy applies one uniform capacity (default 0.05 m^3/s) to all 276 inlets. The
SAGIS conduit layer carries DIAMETER, MATERIAL and INVERT_IN/OUT, which is enough for a
spatially varying estimate from Manning full-pipe flow:

    Q_full = (1/n) * A * R^(2/3) * S^(1/2),   A = pi D^2/4,  R = D/4,  S = dz/L

Slope is a DIFFERENCE of inverts, so the unknown SAGIS vertical datum cancels. 
absolute invert elevations are recorded in feet with no stated datum and a 999 missing-value
sentinel, so they cannot be used directly, but drops between adjacent nodes can.

Q_full is pipe conveyance. The constraint on a storm drain is
usually grate CAPTURE, which depends on gutter flow and grate geometry (FHWA HEC-22) and is
generally smaller. So this is an UPPER BOUND on the capacity an inlet can achieve, and should
set the top of a sensitivity range rather than be used as a point estimate. Sweeping capacity
from a low fraction of it up to it converts the weakest drainage input into a stated uncertainty.

Each inlet takes the capacity of its nearest conduit. 
"""
import argparse
import json
import math

import numpy as np

# Manning's n by pipe material. Standard design values; concrete dominates this network.
MATERIAL_N = {
    "RCP": 0.013, "CONCRETE": 0.013, "CONC": 0.013,
    "CMP": 0.024, "CORRUGATED METAL": 0.024, "METAL": 0.024,
    "PVC": 0.010, "PLASTIC": 0.010, "HDPE": 0.012,
    "DIP": 0.012, "DUCTILE IRON": 0.012, "BRICK": 0.016, "CLAY": 0.013,
}
DEFAULT_N = 0.013
SENTINEL = 999.0          # SAGIS missing-value marker in the invert fields
FT_TO_M = 0.3048
IN_TO_M = 0.0254
MIN_SLOPE, MAX_SLOPE = 1e-4, 0.10   # below: not gravity flow; above: bad data, not a real sewer


def _first(props, *names, default=None):
    for n in names:
        for k in props:
            if k.upper() == n.upper() and props[k] not in (None, ""):
                return props[k]
    return default


def _line_length_m(coords):
    """Planar length of a lon/lat linestring, in metres. Adequate over a few hundred metres."""
    tot = 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        lat = math.radians((y0 + y1) / 2)
        dx = (x1 - x0) * 111320 * math.cos(lat)
        dy = (y1 - y0) * 110540
        tot += math.hypot(dx, dy)
    return tot


def conduit_capacities(conduits_geojson, diameter_units="in"):
    """Return (lons, lats, Q_full) at conduit midpoints. Q in m^3/s."""
    d = json.load(open(conduits_geojson))
    xs, ys, qs, skipped = [], [], [], 0
    for f in d["features"]:
        p = f.get("properties", {}) or {}
        g = f.get("geometry") or {}
        if g.get("type") != "LineString":
            skipped += 1; continue
        coords = g["coordinates"]
        D = _first(p, "DIAMETER", "DIA", "PIPE_SIZE")
        zi = _first(p, "INVERT_IN", "INVERTIN", "INV_IN")
        zo = _first(p, "INVERT_OUT", "INVERTOUT", "INV_OUT")
        try:
            D, zi, zo = float(D), float(zi), float(zo)
        except (TypeError, ValueError):
            skipped += 1; continue
        if SENTINEL in (zi, zo) or D <= 0:
            skipped += 1; continue
        L = _line_length_m(coords)
        if L <= 0:
            skipped += 1; continue
        D_m = D * (IN_TO_M if diameter_units == "in" else 1.0)
        S = abs(zi - zo) * FT_TO_M / L
        if not (MIN_SLOPE <= S <= MAX_SLOPE):
            skipped += 1; continue
        n = MATERIAL_N.get(str(_first(p, "MATERIAL", "MAT", default="")).strip().upper(), DEFAULT_N)
        A = math.pi * D_m ** 2 / 4
        R = D_m / 4
        qs.append((1 / n) * A * R ** (2 / 3) * math.sqrt(S))
        mid = coords[len(coords) // 2]
        xs.append(mid[0]); ys.append(mid[1])
    print(f"{len(qs)} conduits resolved, {skipped} skipped (missing geometry, sentinel, or "
          f"slope outside {MIN_SLOPE}-{MAX_SLOPE})")
    if qs:
        q = np.array(qs)
        print(f"  Q_full: min {q.min():.3f}, median {np.median(q):.3f}, max {q.max():.3f} m^3/s")
    return np.array(xs), np.array(ys), np.array(qs)


def capacity_per_inlet(inlets_geojson, conduits_geojson, out_csv=None, diameter_units="in"):
    """Assign each inlet the capacity of its nearest conduit; median fallback."""
    cx, cy, cq = conduit_capacities(conduits_geojson, diameter_units)
    if cq.size == 0:
        raise SystemExit("no conduits resolved; check field names and units")
    d = json.load(open(inlets_geojson))
    rows = []
    fallback = float(np.median(cq))
    for f in d["features"]:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        x, y = g["coordinates"][:2]
        k = int(np.argmin((cx - x) ** 2 + (cy - y) ** 2))
        rows.append((x, y, float(cq[k])))
    q = np.array([r[2] for r in rows]) if rows else np.array([fallback])
    print(f"{len(rows)} inlets; capacity median {np.median(q):.3f}, "
          f"p10 {np.percentile(q,10):.3f}, p90 {np.percentile(q,90):.3f} m^3/s")
    print("  UPPER BOUND: pipe conveyance, not grate capture (HEC-22). Use as the top of a "
          "sensitivity range, not a point estimate.")
    if out_csv:
        with open(out_csv, "w") as fh:
            fh.write("lon,lat,capacity_m3s\n")
            for x, y, c in rows:
                fh.write(f"{x:.7f},{y:.7f},{c:.4f}\n")
        print(f"  -> {out_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conduits", required=True); ap.add_argument("--inlets", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--diameter-units", default="in", choices=("in", "m"))
    a = ap.parse_args()
    capacity_per_inlet(a.inlets, a.conduits, a.out_csv, a.diameter_units)


if __name__ == "__main__":
    main()
