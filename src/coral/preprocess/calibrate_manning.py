"""Calibrate Manning's n against Hurricane Matthew observations.

Instead of trusting textbook NLCD->n values, estimate the per-class n that best
reproduces the observed high-water marks (HWMs) + gauge. This follows Talea Mayo's
data-assimilation idea for spatially-varying friction: the field n(x) is
under-determined, so we DON'T estimate every cell — we keep the NLCD classes as the
spatial structure and estimate the handful of dominant per-class values (a low-dim
inverse problem). The rest stay at NLCD_N.

Because each objective evaluation is a full LISFLOOD run (expensive, on HPC), the
workflow is propose -> run -> rank, not an in-loop optimizer:

  1. propose(): reproject NLCD once, draw a Latin-hypercube of candidate n-vectors
     over the TUNABLE classes, write Manning_cand{i}.asc + candidates.json.
  2. (you) run LISFLOOD for each candidate grid on HPC (surge+rain reused).
  3. rank(): score each candidate's .mxe against the quality-filtered HWMs, print
     the ranking, write the best per-class n + its Manning grid.

Tunable classes default to Savannah's dominant roughness controls (marsh, water,
developed, forest); edit TUNABLE to change the reduced parameter set / ranges.

Deps: numpy, scipy, rasterio, requests. Reuses make_manning (single source of truth).
"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path
import numpy as np

from .make_manning import NLCD_N, classes_on_dem, classes_to_n, write_ascii

FT2M, EVENT = 0.3048, 135

# reduced parameter set (Mayo dim-reduction): label -> (nlcd codes, lo, hi)
TUNABLE = {
    "marsh":     ([95],          0.03, 0.15),   # emergent herbaceous wetland (dominant)
    "woody_wet": ([90],          0.05, 0.15),
    "water":     ([11],          0.02, 0.04),
    "developed": ([21, 22, 23, 24], 0.03, 0.10),
    "forest":    ([41, 42, 43],  0.08, 0.16),
}


def _read_grid(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.where((a == h.get("nodata_value", -9999)) | (a <= -9990), np.nan, a)
    nx, ny, cs = int(h["ncols"]), int(h["nrows"]), h["cellsize"]
    x = h["xllcorner"] + (np.arange(nx) + .5) * cs
    y = (h["yllcorner"] + ny * cs) - (np.arange(ny) + .5) * cs
    ext = [h["xllcorner"], h["xllcorner"] + nx * cs, h["yllcorner"], h["yllcorner"] + ny * cs]
    return a, x, y, ext


def fetch_hwms(bbox, max_quality=3):
    """Quality-filtered HWMs in bbox=[W,E,S,N] -> [(lon,lat,elev_m)]."""
    url = f"https://stn.wim.usgs.gov/STNServices/Events/{EVENT}/HWMs.json"
    d = json.loads(urllib.request.urlopen(url, timeout=60).read())
    W, E, S, N = bbox
    out = []
    for h in d:
        lon, lat, ev, q = (h.get("longitude_dd"), h.get("latitude_dd"),
                           h.get("elev_ft"), h.get("hwm_quality_id") or 9)
        if lon and lat and ev and W < lon < E and S < lat < N and q < max_quality:
            out.append((lon, lat, ev * FT2M))
    return out


def score(mxe_path, hwm_pts):
    """HWM RMSE (m) for a modelled max water-surface-elevation grid."""
    mxe, x, y, _ = _read_grid(mxe_path)
    res = []
    for lon, lat, obs in hwm_pts:
        i, j = int(np.argmin(np.abs(x - lon))), int(np.argmin(np.abs(y - lat)))
        m = mxe[j, i]
        if np.isfinite(m):
            res.append(m - obs)
    res = np.array(res)
    return float(np.sqrt(np.mean(res**2))), float(res.mean()), len(res)


def _class_n(sample):
    """A candidate n-vector (dict label->value) -> full {code: n} override map."""
    cn = dict(NLCD_N)
    for label, val in sample.items():
        for code in TUNABLE[label][0]:
            cn[code] = val
    return cn


def propose(nlcd, dem, out_dir, n_candidates=16, water_level=0.81, seed=0):
    """Write n_candidates Manning grids over the TUNABLE space (Latin hypercube)."""
    from scipy.stats import qmc
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    classes, z = classes_on_dem(nlcd, dem)                 # reproject ONCE
    labels = list(TUNABLE)
    lo = np.array([TUNABLE[l][1] for l in labels])
    hi = np.array([TUNABLE[l][2] for l in labels])
    u = qmc.LatinHypercube(d=len(labels), seed=seed).random(n_candidates)
    samples = lo + u * (hi - lo)
    manifest = []
    for i, row in enumerate(samples):
        s = {l: round(float(v), 4) for l, v in zip(labels, row)}
        n = classes_to_n(classes, z, _class_n(s), water_level=water_level)
        grid = out / f"Manning_cand{i:02d}.asc"
        write_ascii(str(grid), n, dem, nodata=-9999)
        manifest.append({"i": i, "grid": grid.name, "sample": s})
    json.dump(manifest, open(out / "candidates.json", "w"), indent=2)
    print(f"proposed {n_candidates} candidates -> {out}/candidates.json")
    print("run LISFLOOD for each Manning_cand*.asc on HPC, then call rank().")
    return manifest


def rank(candidates_json, runs_root, bbox, *, root="res_matthew_sav"):
    """Score each candidate's LISFLOOD run against HWMs; print + return sorted.

    Expects each candidate i run under runs_root/cand{i:02d}/results_matthew_sav/<root>.mxe.
    """
    manifest = json.load(open(candidates_json))
    hwms = fetch_hwms(bbox)
    print(f"scoring {len(manifest)} candidates on {len(hwms)} quality-filtered HWMs")
    scored = []
    for c in manifest:
        mxe = Path(runs_root) / f"cand{c['i']:02d}/results_matthew_sav/{root}.mxe"
        if not mxe.exists():
            print(f"  cand{c['i']:02d}: no run yet ({mxe})"); continue
        rmse, bias, n = score(str(mxe), hwms)
        c.update(rmse=round(rmse, 3), bias=round(bias, 3), n_hwm=n)
        scored.append(c)
    scored.sort(key=lambda c: c["rmse"])
    print(f"{'rank':>4} {'cand':>4} {'RMSE':>6} {'bias':>6}  sample")
    for r, c in enumerate(scored):
        print(f"{r:>4} {c['i']:>4} {c['rmse']:>6.3f} {c['bias']:>+6.3f}  {c['sample']}")
    if scored:
        best = scored[0]
        json.dump(best, open(Path(candidates_json).parent / "best_manning.json", "w"), indent=2)
        print(f"\nbest: cand{best['i']:02d} RMSE {best['rmse']:.3f} -> best_manning.json")
    return scored


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Calibrate Manning's n vs Matthew HWMs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--nlcd", required=True)
    p.add_argument("--dem", required=True); p.add_argument("--out", required=True)
    p.add_argument("-n", "--n-candidates", type=int, default=16)
    p.add_argument("--water-level", type=float, default=0.81)
    r = sub.add_parser("rank"); r.add_argument("--candidates", required=True)
    r.add_argument("--runs-root", required=True)
    r.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "E", "S", "N"))
    a = ap.parse_args()
    if a.cmd == "propose":
        propose(a.nlcd, a.dem, a.out, a.n_candidates, a.water_level)
    else:
        rank(a.candidates, a.runs_root, a.bbox)
