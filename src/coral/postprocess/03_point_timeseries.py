#!/usr/bin/env python3
"""
Deliverable 3: point time series as CSV + JSON, for quick graphing.

Two sources supported:
  --gauge  : a GeoClaw gauge#####.txt  (sea-surface elevation eta vs time)
  --mass   : a LISFLOOD .mass file     (domain-integrated series; pick a column)

GeoClaw gauge columns (0-indexed): level, time, h, hu, hv, eta
We export time + eta (water-surface elevation, m, relative to MSL) and h (depth).

Usage:
    python 03_point_timeseries.py --gauge "GoeClaw - Dorian/gauge00005a.txt" \
        --lon -80.9017 --lat 32.0367 --name savannah_gauge5 --out savannah_gauge5
    python 03_point_timeseries.py --mass results/res_rain_Sorted_1198.bdy.mass \
        --col Vol --out domain_volume
"""
import argparse
import csv
import json
import os


def from_gauge(path, lon, lat, name):
    rows = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 6:
                continue
            t = float(p[1]); h = float(p[2]); eta = float(p[5])
            rows.append({"time_s": t, "eta_m": eta, "depth_m": h})
    meta = {
        "name": name, "type": "geoclaw_gauge", "source_file": os.path.basename(path),
        "lon": lon, "lat": lat, "crs": "EPSG:4326",
        "variables": {
            "time_s": {"units": "s", "desc": "time relative to storm reference"},
            "eta_m": {"units": "m", "desc": "sea-surface elevation above MSL datum"},
            "depth_m": {"units": "m", "desc": "water column depth at gauge"},
        },
    }
    return rows, meta


def from_mass(path, col):
    lines = open(path).read().splitlines()
    hi = next(i for i, ln in enumerate(lines) if "Time" in ln.split())
    names = lines[hi].split()
    rows = []
    for ln in lines[hi + 1:]:
        p = ln.split()
        if len(p) != len(names):
            continue
        vals = [float(x) for x in p]
        d = dict(zip(names, vals))
        rows.append({"time_s": d["Time"], col: d[col]})
    meta = {"name": col, "type": "lisflood_mass", "source_file": os.path.basename(path),
            "variables": {"time_s": {"units": "s"}, col: {"units": "varies (see data dictionary)"}}}
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge"); ap.add_argument("--mass")
    ap.add_argument("--col", default="Vol")
    ap.add_argument("--lon", type=float); ap.add_argument("--lat", type=float)
    ap.add_argument("--name", default="point")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.gauge:
        rows, meta = from_gauge(args.gauge, args.lon, args.lat, args.name)
    elif args.mass:
        rows, meta = from_mass(args.mass, args.col)
    else:
        raise SystemExit("give --gauge or --mass")

    # CSV
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # JSON (metadata + data, dev-friendly)
    with open(args.out + ".json", "w") as f:
        json.dump({"metadata": meta, "data": rows}, f, indent=2)
    print(f"wrote {args.out}.csv and {args.out}.json  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
