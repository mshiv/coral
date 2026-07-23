"""Tier-1 storm-drain proxy: tide-conditioned point-source sinks (native LISFLOOD-FP).

Approximates the buried pipe network without a second model (no SWMM). Rationale and the
full three-tier plan: wiki `syntheses/Drainage backup and SWMM coupling`.

Each storm-drain inlet is a point source that removes water at the pipe capacity
(negative Q) while the network can discharge, and stops removing (or optionally pushes a
little back, positive Q = surcharge) while the outfall is drowned by the surge+tide
stage. The stage timeseries is known a priori (it is the forcing), so the drowned
window is pre-computed here and baked into the point-source schedule. This reproduces the
first-order compound signature: the neighbourhood loses its drainage when the tide
is high. It does not capture true network routing/redistribution; that needs SWMM (Tier 2).

Inputs: inlet points (SAGIS inlets_sav), the stage series (make_tide / .bdy-style
`seconds level_m`), an outfall drown-threshold (m, same datum as the stage). Outputs a
`.bci` (P lon lat QVAR drainpx) referencing one shared `.bdy` schedule block.

Deps: numpy, geopandas (points only). Sign convention: LISFLOOD point-source Q>0 = inflow
into the domain, Q<0 = extraction, so draining is negative.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

TAG = "drainpx"


def _read_stage(path):
    """Read a `seconds level_m` series (make_tide / coupling stage) -> (t, stage)."""
    t, v = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        s = line.split()
        if len(s) >= 2:
            t.append(float(s[0])); v.append(float(s[1]))
    return np.array(t), np.array(v)


def schedule(stage_t, stage, *, capacity=0.05, outfall_invert=0.0, backup_frac=0.0):
    """Per-timestep point-source Q (m^3/s) for one inlet given the stage series.

    q = -capacity while stage < outfall_invert (draining); when the outfall is drowned
    (stage >= invert) q = +backup_frac*capacity (0 = simply stops draining; >0 mimics
    surcharge back onto the street).
    """
    drowned = stage >= outfall_invert
    q = np.where(drowned, backup_frac * capacity, -capacity)
    return stage_t, q.astype(float)


def make_drainage_proxy(inlets, stage_series, out_bci, out_bdy, *, capacity=0.05,
                        outfall_invert=0.0, backup_frac=0.0, max_inlets=None):
    """Write a .bci (one QVAR point source per inlet, shared tag) + a .bdy schedule block.

    All inlets share one tide-driven schedule (same capacity + drown threshold), so the
    .bdy holds a single block and every P-line references it. Returns the inlet count.
    """
    import geopandas as gpd
    gdf = gpd.read_file(inlets).to_crs(4326)
    pts = [(g.x, g.y) for g in gdf.geometry if g is not None and not g.is_empty]
    if max_inlets:
        pts = pts[:max_inlets]

    st, q = schedule(*_read_stage(stage_series), capacity=capacity,
                     outfall_invert=outfall_invert, backup_frac=backup_frac)

    Path(out_bdy).parent.mkdir(parents=True, exist_ok=True)
    with open(out_bdy, "w") as f:
        f.write("comment\n")
        f.write(f"{TAG}\n{len(st)}\t\tseconds\n")
        for v, tt in zip(q, st):
            f.write(f"{v:.7g}\t{tt:.5f}\t\n")

    with open(out_bci, "w") as f:
        for x, y in pts:
            f.write(f"P\t{x:.7f}\t{y:.7f}\tQVAR\t{TAG}\n")

    frac_drowned = float((q >= 0).mean())
    print(f"drainage proxy: {len(pts)} inlets @ {capacity} m^3/s, invert {outfall_invert} m; "
          f"outfall drowned {frac_drowned:.0%} of the run -> {out_bci}, {out_bdy}")
    return len(pts)


def from_config(cfg, inlets, stage_series, out_dir=None, **kw):
    out_dir = Path(out_dir or f"data/interim/{cfg.name}")
    return make_drainage_proxy(inlets, stage_series, out_dir / "drainage_proxy.bci",
                               out_dir / "drainage_proxy.bdy", **kw)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tier-1 tide-conditioned storm-drain proxy")
    ap.add_argument("--inlets", required=True, help="inlet point vector (SAGIS inlets_sav)")
    ap.add_argument("--stage", required=True, help="stage series file: `seconds level_m`")
    ap.add_argument("--out-bci", default="drainage_proxy.bci")
    ap.add_argument("--out-bdy", default="drainage_proxy.bdy")
    ap.add_argument("--capacity", type=float, default=0.05, help="drain rate per inlet (m^3/s)")
    ap.add_argument("--outfall-invert", type=float, default=0.0,
                    help="stage (m, same datum) above which the outfall is drowned -> drainage off")
    ap.add_argument("--backup-frac", type=float, default=0.0,
                    help="0 = just stop draining when drowned; >0 = push back (surcharge) as a fraction of capacity")
    ap.add_argument("--max-inlets", type=int, default=None)
    a = ap.parse_args()
    make_drainage_proxy(a.inlets, a.stage, a.out_bci, a.out_bdy, capacity=a.capacity,
                        outfall_invert=a.outfall_invert, backup_frac=a.backup_frac,
                        max_inlets=a.max_inlets)
