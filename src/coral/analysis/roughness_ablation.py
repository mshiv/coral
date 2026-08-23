"""Stage a controlled marsh-roughness ablation on the 30 m domain.

The production ensemble shows that interventions changing elevation move peak depth by up to
1.8 m while interventions changing roughness alone move it by millimetres, even when they edit a
hundred times more cells. The candidate explanation is that the local-inertial friction term,

    1 + g dt n^2 |q| / h^(7/3),

loses purchase once the marsh platform is deeply submerged. That is an assertion about a
denominator. This turns it into a measurement.

Everything except marsh roughness is held fixed: same domain, same boundary, same rainfall, same
infiltration, same initial state. Only the Manning grid changes, and only over the marsh band.
Multipliers deliberately reach beyond the sourced range: the members already applied about 2.2x
and produced 3 mm, so testing 2x again proves nothing. If 8x still does nothing, friction is not
the lever, and the result is a statement about the regime rather than about the sampled range.

The 30 m domain is used rather than the 4 m Pin Point clip because the clip barely contains marsh
-- restoration there edits 79,000 cells and moves 0.003 m partly because there is so little
platform in frame. At 30 m the estuary's marsh is actually present.

Usage:
  python -m coral.analysis.roughness_ablation stage \
      --base <compound_tide_30m dir> --out <runs root> \
      --waterline 1.114 --mlw -1.091 --multipliers 1 2 4 8 --slr 0 0.301 1.098 2.043
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

NODATA = -9999.0


def read_asc(path):
    hdr, n = {}, 0
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 2 and not p[0][:1].isdigit() and p[0][:1] != "-":
                hdr[p[0].lower()] = float(p[1]); n += 1
            else:
                break
    a = np.loadtxt(path, skiprows=n)
    return np.where(a > NODATA + 1.0, a, np.nan), hdr


def write_like(out, arr, template):
    """Copy the template's six header lines verbatim so the grids stay byte-aligned."""
    with open(template) as f:
        head = [f.readline() for _ in range(6)]
    with open(out, "w") as f:
        f.writelines(head)
        np.savetxt(f, np.where(np.isfinite(arr), arr, NODATA), fmt="%.4f")


def marsh_band(dem, waterline, mlw):
    """Cells between MLW and MHW: the tidal platform the ablation acts on.

    Not "all land": raising roughness on upland would confound the platform's contribution with
    overland routing far from the estuary, and the question is specifically what marsh does.
    """
    return np.isfinite(dem) & (dem >= mlw) & (dem <= waterline)


def stage(a):
    base = Path(a.base)
    dem_p = next(iter(sorted(base.glob("SUB_DEM_*.asc"))))
    man_p = next(iter(sorted(base.glob("Manning_*.asc"))))
    bdy_p = next(iter(sorted(base.glob("*.bdy"))))
    dem, _ = read_asc(dem_p)
    man, _ = read_asc(man_p)
    band = marsh_band(dem, a.waterline, a.mlw)
    n_band = int(band.sum())
    if n_band == 0:
        raise SystemExit("marsh band is empty; check --waterline and --mlw against this DEM")

    v = man[band & np.isfinite(man)]
    print(f"marsh band: {n_band:,} cells, n p50={np.median(v):.3f} "
          f"p90={np.percentile(v, 90):.3f}")

    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    scenario = man_p.stem.split("_", 1)[1]
    manifest = []
    for mult in a.multipliers:
        for slr in a.slr:
            tag = f"rough{mult:g}x_slr{slr:g}".replace(".", "p")
            d = out_root / tag
            d.mkdir(exist_ok=True)
            # Everything except the Manning grid is shared. Symlinked rather than copied: the
            # DEM alone is 55 MB and 16 members would carry a gigabyte of identical files.
            for p in base.glob("*"):
                if p.name.startswith("results_") or p.name == man_p.name:
                    continue
                if p.name == bdy_p.name and slr:
                    continue                       # written below with the offset applied
                dest = d / p.name
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                dest.symlink_to(p.resolve())
            if slr:
                from ..emulator.sweep import apply_slr_to_bdy
                apply_slr_to_bdy(bdy_p, d / bdy_p.name, slr)
            edited = np.where(band, man * mult, man)
            write_like(d / man_p.name, edited, man_p)
            manifest.append({"name": tag, "run_dir": str(d.resolve()),
                             "multiplier": float(mult), "slr_m": float(slr),
                             "band_cells": n_band,
                             "n_median_after": float(np.median(edited[band]))})
            print(f"  {tag:26s} n p50 {np.median(edited[band]):.3f}")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_root / "run_dirs.txt").write_text(
        "\n".join(m["run_dir"] for m in manifest) + "\n")
    print(f"\nstaged {len(manifest)} runs -> {out_root}")
    print("  each needs its startfile primed at its own boundary level before submission:")
    print("  the .bdy is offset per member, so a shared startfile puts the domain below its")
    print("  own boundary and the run floods in at the first step.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="write the ablation run directories")
    s.add_argument("--base", required=True, help="a validated 30 m run directory")
    s.add_argument("--out", required=True)
    s.add_argument("--waterline", type=float, required=True)
    s.add_argument("--mlw", type=float, required=True)
    s.add_argument("--multipliers", type=float, nargs="+", default=[1, 2, 4, 8])
    s.add_argument("--slr", type=float, nargs="+", default=[0.0, 0.301, 1.098, 2.043])
    s.set_defaults(func=stage)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
