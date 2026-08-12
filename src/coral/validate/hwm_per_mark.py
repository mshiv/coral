"""Per-mark high-water-mark comparison, stratified by mark type.

Aggregate bias and RMSE hide which marks a model misses. Muñoz et al. 2020 report the
Savannah-Matthew comparison per observation and split it by class: their three seed-line
high-water marks are all positive (+0.15 to +0.38 m, mean 0.29), while nine peak-stage
observations average +0.04 m with both signs. That split matters because seed and debris
lines record the maximum water surface reached, including wave runup and debris deposition,
so they sit high relative to a still-water surface, which is what a depth-averaged model
computes.

This reports bias per mark and by stratum, using the USGS STN metadata already attached to
each mark: type (seed line, debris, stain), environment (coastal or riverine), quality, and
the stillwater flag.

    python -m coral.validate.hwm_per_mark --mxe runs/.../res.mxe --dem .../SUB_DEM_SAV.asc \\
        --max-quality 2 --max-dem-diff 1.0 --csv reports/hwm_per_mark.csv
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from collections import defaultdict

import numpy as np

from ..analysis.physics_ab import _read_grid, _sample, _extent, USGS_EVENT, FT2M

HWM_TYPES = {1: "Mud", 2: "Debris", 3: "Clear water", 4: "Vegetation line",
             5: "Seed line", 6: "Stain line", 7: "Melted snow line", 8: "Presence",
             9: "Other"}
VDATUMS = {2: "NAVD88", 4: "NGVD29"}


def fetch_hwms_full(bbox, event=USGS_EVENT):
    """USGS STN marks in bbox, keeping the metadata needed to stratify.

    Returns a list of dicts. Elevations converted to metres; the vertical datum is kept so
    a mixed-datum set can be flagged rather than silently averaged.
    """
    url = f"https://stn.wim.usgs.gov/STNServices/Events/{event}/HWMs.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = json.load(r)
    w, e, s, n = bbox
    out = []
    for h in raw:
        lon, lat, ev = h.get("longitude_dd"), h.get("latitude_dd"), h.get("elev_ft")
        if lon is None or lat is None or ev is None:
            continue
        if not (w <= lon <= e and s <= lat <= n):
            continue
        out.append(dict(
            lon=lon, lat=lat, obs=ev * FT2M,
            hwm_id=h.get("hwm_id"),
            type=HWM_TYPES.get(h.get("hwm_type_id"), f"id{h.get('hwm_type_id')}"),
            env=h.get("hwm_environment"),
            quality=h.get("hwm_quality_id"),
            stillwater=bool(h.get("stillwater")),
            vdatum=VDATUMS.get(h.get("vdatum_id"), f"id{h.get('vdatum_id')}"),
            desc=(h.get("hwm_locationdescription") or "")[:40],
        ))
    return out


def _summary(rows):
    d = np.array([r["resid"] for r in rows])
    return dict(n=len(d), bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                lo=float(d.min()), hi=float(d.max()))


def compare(mxe, dem, *, max_quality=None, max_dem_diff=None, csv=None):
    """Sample the modelled maximum water surface at each mark; report per mark and by stratum."""
    grid, gh = _read_grid(mxe)
    dg, dh = _read_grid(dem)
    marks = fetch_hwms_full(_extent(dh))

    rows, dropped = [], defaultdict(int)
    for m in marks:
        if max_quality is not None and (m["quality"] is None or m["quality"] > max_quality):
            dropped["quality"] += 1; continue
        mod = _sample(grid, gh, m["lon"], m["lat"])
        if not np.isfinite(mod):
            dropped["outside/dry"] += 1; continue
        if max_dem_diff is not None:
            z = _sample(dg, dh, m["lon"], m["lat"])
            if not np.isfinite(z) or abs(z - m["obs"]) > max_dem_diff:
                dropped["dem-diff"] += 1; continue
        rows.append({**m, "model": float(mod), "resid": float(mod - m["obs"])})

    if not rows:
        raise SystemExit("no marks survived the filters")

    print(f"{len(rows)} marks scored" + (f"  (dropped: {dict(dropped)})" if dropped else ""))
    vd = {r["vdatum"] for r in rows}
    if len(vd) > 1:
        print(f"  WARNING: mixed vertical datums in the scored set: {vd}")

    print(f"\n{'id':>7} {'type':14s} {'env':9s} {'q':>2} {'sw':>3} "
          f"{'obs':>6} {'model':>6} {'resid':>7}  location")
    for r in sorted(rows, key=lambda r: -r["resid"]):
        print(f"{str(r['hwm_id']):>7} {r['type']:14s} {str(r['env']):9s} {r['quality']:>2} "
              f"{'Y' if r['stillwater'] else 'n':>3} {r['obs']:6.2f} {r['model']:6.2f} "
              f"{r['resid']:+7.2f}  {r['desc']}")

    allr = _summary(rows)
    print(f"\nall marks           n={allr['n']:3d}  bias {allr['bias']:+.3f}  "
          f"RMSE {allr['rmse']:.3f}  range {allr['lo']:+.2f} to {allr['hi']:+.2f}")

    for label, key in (("type", "type"), ("environment", "env")):
        print(f"\nby {label}:")
        groups = defaultdict(list)
        for r in rows:
            groups[str(r[key])].append(r)
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            st = _summary(rs)
            print(f"  {g:16s} n={st['n']:3d}  bias {st['bias']:+.3f}  RMSE {st['rmse']:.3f}  "
                  f"range {st['lo']:+.2f} to {st['hi']:+.2f}")

    print("\nby stillwater flag:")
    for flag in (True, False):
        rs = [r for r in rows if r["stillwater"] is flag]
        if rs:
            st = _summary(rs)
            print(f"  {'stillwater' if flag else 'not stillwater':16s} n={st['n']:3d}  "
                  f"bias {st['bias']:+.3f}  RMSE {st['rmse']:.3f}")

    # Seed/debris lines record the maximum surface including runup; a model computing a
    # still-water surface is expected to sit below them, not above.
    runup = [r for r in rows if r["type"] in ("Seed line", "Debris")]
    other = [r for r in rows if r["type"] not in ("Seed line", "Debris")]
    if runup and other:
        a, b = _summary(runup), _summary(other)
        print(f"\nseed/debris vs rest: bias {a['bias']:+.3f} (n={a['n']}) "
              f"vs {b['bias']:+.3f} (n={b['n']})  ->  difference {a['bias'] - b['bias']:+.3f} m")

    if csv:
        import csv as _csv
        with open(csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {csv}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mxe", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--max-quality", type=int, default=None)
    ap.add_argument("--max-dem-diff", type=float, default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    compare(a.mxe, a.dem, max_quality=a.max_quality, max_dem_diff=a.max_dem_diff, csv=a.csv)


if __name__ == "__main__":
    main()
