"""Transfer one representative 4 m intervention into a paired 30 m regional diagnostic.

This is a scale-transfer sensitivity, not an exact replica of the native-grid member.  Narrow
raised features use the maximum absolute crest so they remain connected.  Roughness uses the
area-average change in n**2 over every valid 4 m subcell, then applies that change to the 30 m
baseline; this retains fractional treatment coverage and is closer to the friction term than
maximum-n resampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .roughness_ablation import par_inputs, read_asc, write_like

DEM_KINDS = {"road_raise", "living_shoreline"}
MANNING_KINDS = {"living_shoreline", "marsh_restoration", "marsh_migration"}


def transform(h):
    from rasterio.transform import from_origin
    return from_origin(h["xllcorner"], h["yllcorner"] + h["nrows"] * h["cellsize"],
                       h["cellsize"], h["cellsize"])


def aligned(a, ha, b, hb):
    return a.shape == b.shape and all(ha[k] == hb[k] for k in
                                      ("nrows", "ncols", "xllcorner", "yllcorner", "cellsize"))


def grid(directory, prefix):
    hits = sorted(Path(directory).glob(f"{prefix}*.asc"))
    if not hits:
        raise SystemExit(f"no {prefix}*.asc in {directory}")
    return hits[0]


def transfer_dem(source_base, source_edit, target_base, out, tol=0.01):
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject
    b4, h4 = read_asc(source_base); e4, he = read_asc(source_edit)
    b30, h30 = read_asc(target_base)
    if not aligned(b4, h4, e4, he):
        raise SystemExit("source DEMs do not align")
    changed4 = np.isfinite(b4) & np.isfinite(e4) & (e4 > b4 + tol)
    crest4 = np.where(changed4, e4, -9999).astype("float32")
    crest30 = np.full(b30.shape, -9999, "float32")
    reproject(crest4, crest30, src_transform=transform(h4), dst_transform=transform(h30),
              src_crs=CRS.from_epsg(4326), dst_crs=CRS.from_epsg(4326),
              src_nodata=-9999, dst_nodata=-9999, resampling=Resampling.max)
    hit = np.isfinite(b30) & (crest30 > -9990)
    result = np.where(hit, np.maximum(b30, crest30), b30)
    changed30 = hit & (result > b30 + tol)
    write_like(out, result, target_base)
    return {"source_dem_cells": int(changed4.sum()), "target_dem_cells": int(changed30.sum())}


def transfer_manning(source_base, source_edit, target_base, out, tol=1e-6):
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject
    b4, h4 = read_asc(source_base); e4, he = read_asc(source_edit)
    b30, h30 = read_asc(target_base)
    if not aligned(b4, h4, e4, he):
        raise SystemExit("source Manning grids do not align")
    valid = np.isfinite(b4) & np.isfinite(e4)
    changed4 = valid & (np.abs(e4 - b4) > tol)
    # Zero on untreated valid subcells is intentional: it makes treatment coverage fractional.
    dn2 = np.where(valid, e4 ** 2 - b4 ** 2, -9999).astype("float32")
    avg_dn2 = np.full(b30.shape, -9999, "float32")
    reproject(dn2, avg_dn2, src_transform=transform(h4), dst_transform=transform(h30),
              src_crs=CRS.from_epsg(4326), dst_crs=CRS.from_epsg(4326),
              src_nodata=-9999, dst_nodata=-9999, resampling=Resampling.average)
    covered = np.isfinite(b30) & (avg_dn2 > -9990)
    result = b30.copy()
    result[covered] = np.sqrt(np.maximum(0, b30[covered] ** 2 + avg_dn2[covered]))
    changed30 = covered & (np.abs(result - b30) > tol)
    write_like(out, result, target_base)
    return {"source_manning_cells": int(changed4.sum()),
            "target_manning_cells": int(changed30.sum()),
            "target_manning_delta_median": (float(np.median((result-b30)[changed30]))
                                               if changed30.any() else 0.0)}


def stage_dir(source, dest, par, omit, snapshot_s):
    dest.mkdir(parents=True)
    for p in source.iterdir():
        if p.name.startswith("results_") or p.name == par.name or p.name in omit:
            continue
        (dest / p.name).symlink_to(p.resolve())
    lines = [ln for ln in par.read_text().splitlines()
             if ln.split()[:1] not in (["saveint"], ["qoutput"])]
    lines += [f"saveint        {snapshot_s:g}", "qoutput"]
    (dest / par.name).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", required=True, choices=sorted(DEM_KINDS | MANNING_KINDS))
    ap.add_argument("--source-base", required=True)
    ap.add_argument("--source-member", required=True)
    ap.add_argument("--target-control", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-s", type=float, default=1800)
    ap.add_argument("--selection-rule", default="targeted Int2050 member nearest median footprint")
    a = ap.parse_args()

    root = Path(a.out)
    if root.exists():
        raise SystemExit(f"output already exists: {root}; move it aside explicitly")
    target = Path(a.target_control)
    par, want = par_inputs(target)
    dem_name, man_name = want["demfile"].name, want["manningfile"].name
    omit = {dem_name, man_name}
    stage_dir(target, root / "control", par, omit, a.snapshot_s)
    stage_dir(target, root / "intervention", par, omit, a.snapshot_s)
    for arm in (root / "control", root / "intervention"):
        (arm / dem_name).symlink_to(want["demfile"].resolve())
        (arm / man_name).symlink_to(want["manningfile"].resolve())

    stats = {}
    if a.kind in DEM_KINDS:
        (root / "intervention" / dem_name).unlink()
        stats.update(transfer_dem(grid(a.source_base, "SUB_DEM"),
                                  grid(a.source_member, "SUB_DEM"), want["demfile"],
                                  root / "intervention" / dem_name))
    if a.kind in MANNING_KINDS:
        (root / "intervention" / man_name).unlink()
        stats.update(transfer_manning(grid(a.source_base, "Manning"),
                                      grid(a.source_member, "Manning"), want["manningfile"],
                                      root / "intervention" / man_name))
    if not any(v for k, v in stats.items() if k.startswith("target_") and k.endswith("_cells")):
        raise SystemExit("transfer produced no changed 30 m cells")
    report = {"kind": a.kind, "source_member": str(Path(a.source_member).resolve()),
              "target_control": str(target.resolve()), "selection_rule": a.selection_rule,
              "snapshot_s": a.snapshot_s, "aggregation": {
                  "DEM": "maximum absolute crest", "Manning": "area-average delta n^2"},
              **stats}
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); print(f"staged regional pair -> {root}")


if __name__ == "__main__":
    main()
