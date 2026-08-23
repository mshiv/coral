"""Stage a controlled marsh-roughness ablation on the 30 m domain.

The production ensemble shows that interventions changing elevation move peak depth by up to
1.8 m while interventions changing roughness alone move it by millimetres, even when they edit a
hundred times more cells. The candidate explanation is that the local-inertial friction term,

    1 + g dt n^2 |q| / h^(7/3),

loses purchase once the marsh platform is deeply submerged. That is an assertion about a
denominator. This turns it into a measurement.

Everything except marsh roughness is held fixed: same domain, same boundary, same rainfall, same
infiltration, same initial state. Only the Manning grid changes, and only over the marsh band.

States are absolute roughness values rather than multipliers. The 30 m marsh band is uniform at
n = 0.110, so a multiplier ladder reaches 0.88 by 8x, an order of magnitude past anything physical
-- Chow puts the densest brush near 0.15 to 0.20. Absolute states also let the ladder run DOWNWARD,
which matters because the strongest published counterexample does exactly that: replacing vegetated
marsh roughness with a uniform open-water value raised mean inundated area by 59.2 percent across
ten synthetic storms in Apalachicola Bay. Removing the marsh is a sharper test of whether roughness
has purchase than adding implausible amounts of it.

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


def par_inputs(base):
    """The files the par actually declares, not whatever sorts first in a glob.

    A validated run directory can hold several boundary files and several grids from earlier
    configurations. Globbing `*.bdy` there returned savannah_matthew_compound.bdy while the par
    read savannah_matthew_compound_tide.bdy, so the sea-level offset was written to a file
    nothing used, the real boundary passed through unchanged, and every member was primed for a
    rise its boundary never received. Errors then grew with the offset and looked like a physics
    problem at the high sea levels.

    This is the third time in this project that selecting a file by pattern rather than by
    declaration has produced a plausible wrong answer. Read the par.
    """
    pars = sorted(base.glob("*.par"))
    if len(pars) != 1:
        raise SystemExit(f"{base} holds {len(pars)} .par files; cannot tell which run this is. "
                         "Point --base at a directory with exactly one.")
    want = {}
    for line in pars[0].read_text().splitlines():
        s = line.split()
        if len(s) == 2 and s[0].lower() in ("demfile", "manningfile", "bdyfile", "startfile"):
            want[s[0].lower()] = base / s[1]
    for k in ("demfile", "manningfile", "bdyfile"):
        if k not in want:
            raise SystemExit(f"{pars[0].name} declares no {k}")
        if not want[k].exists():
            raise SystemExit(f"{pars[0].name} declares {k} = {want[k].name}, which is not in {base}")
    return pars[0], want


def stage(a):
    base = Path(a.base)
    par_p, want = par_inputs(base)
    dem_p, man_p, bdy_p = want["demfile"], want["manningfile"], want["bdyfile"]
    print(f"par {par_p.name} declares:")
    for k in ("demfile", "manningfile", "bdyfile", "startfile"):
        if k in want:
            print(f"  {k:12s} {want[k].name}")
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
    start_p = want.get("startfile")
    manifest = []
    for mult in a.multipliers:
        for slr in a.slr:
            tag = ("base" if mult < 0 else
                   (f"n{mult:g}" if a.absolute else f"rough{mult:g}x")) + f"_slr{slr:g}"
            tag = tag.replace(".", "p")
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
            # A negative state is the sentinel for "leave the grid alone". The true control is
            # the untouched base grid: setting the band to its own median is NOT a control,
            # because the band contains open-water and channel-edge cells well below that
            # median and overwriting them changes the run.
            if mult < 0:
                edited = man.copy()
            elif a.absolute:
                edited = np.where(band, mult, man)
            else:
                edited = np.where(band, man * mult, man)
            write_like(d / man_p.name, edited, man_p)
            manifest.append({"name": tag, "run_dir": str(d.resolve()),
                             "n_target" if a.absolute else "multiplier": float(mult),
                             "slr_m": float(slr),
                             "band_cells": n_band,
                             "n_median_after": float(np.median(edited[band]))})
            print(f"  {tag:26s} n p50 {np.median(edited[band]):.3f}")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_root / "run_dirs.txt").write_text(
        "\n".join(m["run_dir"] for m in manifest) + "\n")
    print(f"\nstaged {len(manifest)} runs -> {out_root}")
    if start_p:
        print(f"  startfile is {start_p.name}; prime one per offset before submitting.")
    print("  The .bdy is offset per member, so a shared startfile leaves the domain out of")
    print("  hydrostatic balance with its own boundary and the run floods in at the first step.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="write the ablation run directories")
    s.add_argument("--base", required=True, help="a validated 30 m run directory")
    s.add_argument("--out", required=True)
    s.add_argument("--waterline", type=float, required=True)
    s.add_argument("--mlw", type=float, required=True)
    s.add_argument("--multipliers", type=float, nargs="+",
                   default=[-1, 0.02, 0.055, 0.11, 0.22, 0.44],
                   help="roughness states. Absolute n by default; multipliers with --relative. "
                        "A negative value means leave the grid untouched, which is the control.")
    s.add_argument("--relative", dest="absolute", action="store_false",
                   help="treat the values as multipliers of the existing grid")
    s.set_defaults(absolute=True)
    s.add_argument("--slr", type=float, nargs="+", default=[0.0, 0.301, 1.098, 2.043])
    s.set_defaults(func=stage)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
