"""Transfer a verified 4 m floodwall system to a paired 30 m diagnostic.

Only cells where the source member raised its DEM are transferred. Absolute crest elevations
are aggregated with a maximum operator, because averaging a line into a 30 m cell lowers or
erases the barrier. The target control supplies every forcing and input other than the DEM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .roughness_ablation import par_inputs, read_asc, write_like


def _transform(h):
    from rasterio.transform import from_origin
    return from_origin(h["xllcorner"], h["yllcorner"] + h["nrows"] * h["cellsize"],
                       h["cellsize"], h["cellsize"])


def transfer(source_base, source_wall, target_base, out, tol=0.01):
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject
    b4, h4 = read_asc(source_base); w4, hw = read_asc(source_wall)
    b30, h30 = read_asc(target_base)
    if b4.shape != w4.shape or any(h4[k] != hw[k] for k in
                                   ("nrows", "ncols", "xllcorner", "yllcorner", "cellsize")):
        raise SystemExit("source baseline and wall grids are not aligned")
    raised = np.isfinite(b4) & np.isfinite(w4) & (w4 > b4 + tol)
    if not raised.any():
        raise SystemExit("source wall contains no raised cells")
    crest4 = np.where(raised, w4, -9999.0).astype("float32")
    crest30 = np.full(b30.shape, -9999.0, "float32")
    reproject(crest4, crest30, src_transform=_transform(h4), dst_transform=_transform(h30),
              src_crs=CRS.from_epsg(4326), dst_crs=CRS.from_epsg(4326),
              src_nodata=-9999.0, dst_nodata=-9999.0, resampling=Resampling.max)
    hit = (crest30 > -9990) & np.isfinite(b30)
    edited = np.where(hit, np.maximum(b30, crest30), b30)
    changed = hit & (edited > b30 + tol)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_like(out, edited, target_base)
    return {"source_raised_cells_4m": int(raised.sum()),
            "target_touched_cells_30m": int(hit.sum()),
            "target_raised_cells_30m": int(changed.sum()),
            "target_crest_min_m": float(edited[changed].min()),
            "target_crest_median_m": float(np.median(edited[changed])),
            "target_crest_max_m": float(edited[changed].max())}


def _stage_dir(source, dest, par, dem_name, snapshot_s):
    dest.mkdir(parents=True)
    for p in source.iterdir():
        if p.name.startswith("results_") or p.name == par.name or p.name == dem_name:
            continue
        (dest / p.name).symlink_to(p.resolve())
    text = par.read_text()
    lines = [ln for ln in text.splitlines()
             if ln.split()[:1] not in (["saveint"], ["qoutput"])]
    lines += [f"saveint        {snapshot_s:g}", "qoutput"]
    (dest / par.name).write_text("\n".join(lines) + "\n")


def stage(a):
    target = Path(a.target_control)
    par, want = par_inputs(target)
    dem_name = want["demfile"].name
    root = Path(a.out)
    if root.exists():
        raise SystemExit(f"output already exists: {root}; move it aside explicitly")
    control, wall = root / "control", root / "wall"
    _stage_dir(target, control, par, dem_name, a.snapshot_s)
    _stage_dir(target, wall, par, dem_name, a.snapshot_s)
    (control / dem_name).symlink_to(want["demfile"].resolve())
    stats = transfer(a.source_base, a.source_wall, want["demfile"], wall / dem_name)
    report = {"selection_rule": a.selection_rule, "source_member": a.source_member,
              "target_control": str(target.resolve()), "snapshot_s": a.snapshot_s,
              **stats}
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"staged control and wall -> {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-base", required=True)
    ap.add_argument("--source-wall", required=True)
    ap.add_argument("--target-control", required=True,
                    help="validated 30 m run at the desired SLR, including its own startfile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-s", type=float, default=1800.0)
    ap.add_argument("--source-member", default="slrInt2050_floodwall3")
    ap.add_argument("--selection-rule", default="largest total benefit among targeted Int2050 walls")
    a = ap.parse_args(); stage(a)


if __name__ == "__main__":
    main()
