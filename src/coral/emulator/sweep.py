"""Stage-0 scenario-sweep driver: generate the emulator training ensemble.

Fixing the storm (Matthew) makes this cheap: GeoClaw runs once. Everything swept is
on the LISFLOOD side:
  * SLR  = bathtub offset added to the surge boundary (.bdy), no GeoClaw rerun.
  * interventions = edits to the DEM / Manning / infiltration grids (coral.interventions).
So each sample is one LISFLOOD run reusing the Matthew surge + rainfall.

plan_sweep()  -> a list of scenario specs (SLR x interventions), sampled.
build_sweep() -> for each spec: write a run dir with modified grids + SLR-shifted
                 .bdy + a copy of the .par/rain, and a manifest.json the emulator
                 dataset consumes (emulator.dataset.build_manifest).
Running LISFLOOD is a pluggable step (default: emit an sbatch submit list for HPC).

Deps: numpy, scipy, rasterio (via interventions), stdlib.
"""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
import numpy as np

from ..interventions import sample_intervention, apply_intervention
from ..preprocess.make_manning import write_ascii
from .dataset import read_asc


def apply_slr_to_bdy(bdy_in, bdy_out, slr_m):
    """Add a uniform SLR offset to a LISFLOOD .bdy (bathtub). HVAR blocks list
    `value time` data rows; we add slr_m to the value column of numeric rows and
    copy everything else verbatim. Verify against your .bdy format on first use."""
    out = []
    for line in Path(bdy_in).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                v, t = float(parts[0]), float(parts[1])
                out.append(f"{v + slr_m:.4f}\t{t:.1f}"); continue
            except ValueError:
                pass
        out.append(line)
    Path(bdy_out).write_text("\n".join(out) + "\n")


def plan_sweep(slr_levels, kinds, n_per_kind=4, include_combos=True, seed=0):
    """Build the sweep spec: baseline + single interventions x SLR (+ optional
    pairwise combos). Returns a list of {name, slr_m, slr_label, interventions:[knobs,...]}.

    `slr_levels` is either a list of floats (name f"slr{slr}", slr_label=None) or a
    list of (label, float) tuples (name f"slr{label}", e.g. slrInt2050), for SLR levels
    resolved from NOAA scenarios via preprocess.fetch_slr.slr_levels. Mixed lists are
    fine; each entry is normalized independently.
    """
    rng = np.random.default_rng(seed)
    specs = []
    for entry in slr_levels:
        if isinstance(entry, (tuple, list)):
            label, slr = entry
        else:
            label, slr = None, entry
        tag = label if label is not None else slr
        specs.append({"name": f"slr{tag}_base", "slr_m": slr, "slr_label": label,
                      "interventions": []})
        for kind in kinds:
            for j in range(n_per_kind):
                kb = sample_intervention(kind, rng)
                specs.append({"name": f"slr{tag}_{kind}{j}", "slr_m": slr, "slr_label": label,
                              "interventions": [kb]})
        if include_combos:                       # a few multi-intervention configs
            for j in range(n_per_kind):
                combo = [sample_intervention(k, rng) for k in rng.choice(kinds, 2, replace=False)]
                specs.append({"name": f"slr{tag}_combo{j}", "slr_m": slr, "slr_label": label,
                              "interventions": combo})
    return specs


def _asc_ext(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    nx, ny, cs = int(h["ncols"]), int(h["nrows"]), h["cellsize"]
    return [h["xllcorner"], h["xllcorner"]+nx*cs, h["yllcorner"], h["yllcorner"]+ny*cs]


def build_sweep(base_dir, specs, out_root, *, root="res_matthew_sav",
                bdy_glob="*.bdy", sea_level=0.81,
                focus_center=None, focus_radius_km=None, focus_mask=None, nlcd=None,
                wetlands=None, soil_ksat=None, buildings=None,
                place="random", flood_depth=None, flood_zone=None, res_m=30.0,
                job_array=True, lisflood_bin="lisflood", account="gts-arobel3-atlas",
                partition="cpu-medium", throttle=50,
                modules="gcc/12.3.0 netcdf-c/4.9.2-gszew36", cpus_per_task=8,
                mem="4G", walltime="01:30:00"):
    """Materialize each spec as a LISFLOOD-ready run dir + a training manifest.

    base_dir must hold the Matthew inputs: SUB_DEM*.asc, Manning*.asc, infil_*.asc,
    infilcap_*.asc (optional), the .bdy, .bci, .par, and rain input. Static grids are
    edited per intervention; the .bdy is SLR-shifted; everything else is copied.

    SAGIS-conditioned siting (Phase 2): `wetlands` (NWI bool mask), `soil_ksat` (SSURGO
    Ksat grid mm/hr), `buildings` (FEMA footprint bool mask), all on the DEM grid, e.g.
    from interventions.context_rasters. Submission: `job_array` emits a single SLURM array
    script (run_array.sbatch + run_dirs.txt) instead of one sbatch per dir.
    """
    base = Path(base_dir); out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    dem_p = next(base.glob("SUB_DEM*.asc")); man_p = next(base.glob("Manning*.asc"))
    ksat_p = next(iter(base.glob("infil_*.asc")), None)
    cap_p = next(iter(base.glob("infilcap_*.asc")), None)
    bdy_p = next(base.glob(bdy_glob))
    dem0, _ = read_asc(dem_p); man0, _ = read_asc(man_p)
    ksat0 = read_asc(ksat_p)[0] if ksat_p else np.zeros_like(dem0)
    awc0 = read_asc(cap_p)[0] if cap_p else np.zeros_like(dem0)
    # context for intervention siting: Pin Point focus + NLCD land cover (optional)
    from ..interventions import focus_region
    focus = focus_mask                                  # precomputed (e.g. hydraulic connectivity)
    if focus is None and focus_center is not None and focus_radius_km:
        focus = focus_region(dem0.shape, _asc_ext(dem_p), focus_center, focus_radius_km)
    classes = None
    if nlcd is not None:
        from ..preprocess.make_manning import classes_on_dem
        classes, _ = classes_on_dem(nlcd, str(dem_p))
    passthrough = [p for p in base.glob("*")
                   if p.suffix in (".bci", ".par", ".txt", ".nc", ".sbatch", ".sh")
                   or p.suffix == ""]   # "" = the lisflood binary; .sbatch = job script

    # Shared, deduplicated inputs. Copying every grid into every member costs about 120 MB
    # per member, and most of it is redundant: the .bdy depends only on the SLR level, so a
    # 306-member sweep over 6 levels writes 300 needless copies of a 69 MB file, and a grid
    # an intervention does not touch is bit-identical to the base. Both are symlinked instead.
    shared = out_root / "_shared"; shared.mkdir(exist_ok=True)

    def shared_bdy(slr_m):
        """One .bdy per distinct SLR offset, written once and reused."""
        tag = f"{slr_m:+.4f}".replace(".", "p")
        p = shared / f"{bdy_p.stem}_slr{tag}{bdy_p.suffix}"
        if not p.exists():
            apply_slr_to_bdy(bdy_p, p, slr_m)
        return p

    def link(target, dest):
        """Absolute symlink, replacing anything already at dest."""
        dest = Path(dest)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(Path(target).resolve())

    def stage_grid(arr, base_arr, base_path, dest):
        """Write the edited grid, or symlink the base file when the intervention left it
        untouched. Returns True if a new file was written."""
        if np.array_equal(np.nan_to_num(arr, nan=-9999.0),
                          np.nan_to_num(base_arr, nan=-9999.0)):
            link(base_path, dest); return False
        write_ascii(str(dest), arr, str(dem_p)); return True

    manifest = []
    for spec in specs:
        run = out_root / spec["name"]; run.mkdir(exist_ok=True)
        dem, man, ksat, awc = dem0.copy(), man0.copy(), ksat0.copy(), awc0.copy()
        for kb in spec["interventions"]:
            dem, man, ksat, awc, _ = apply_intervention(kb, dem, man, ksat, awc,
                                                        sea_level=sea_level,
                                                        classes=classes, focus=focus,
                                                        wetlands=wetlands, soil_ksat=soil_ksat,
                                                        buildings=buildings, place=place,
                                                        flood_depth=flood_depth, flood_zone=flood_zone,
                                                        res_m=res_m)
        stage_grid(dem, dem0, dem_p, run / dem_p.name)
        stage_grid(man, man0, man_p, run / man_p.name)
        if ksat_p:
            stage_grid(ksat, ksat0, ksat_p, run / ksat_p.name)
        if cap_p:
            stage_grid(awc, awc0, cap_p, run / cap_p.name)
        link(shared_bdy(spec["slr_m"]), run / bdy_p.name)
        for p in passthrough:
            if p.is_file():                       # .bci, .par, rain: never edited, so link
                link(p, run / p.name)
        forcing = {"slr_m": spec["slr_m"]}
        if spec.get("slr_label"):
            forcing["slr_label"] = spec["slr_label"]
        for kb in spec["interventions"]:                    # flatten knobs as features
            forcing.update({f"{kb['kind']}_{k}": v for k, v in kb.items()
                            if k not in ("kind", "seed")})
        manifest.append({"name": spec["name"], "run_dir": str(run), "root": root,
                         "forcing": forcing, "interventions": spec["interventions"]})

    json.dump(manifest, open(out_root / "manifest.json", "w"), indent=2)
    # pick the .par deterministically: prefer an ASCII-output one. A base dir often holds
    # several (savannah.par, test.par) that enable netcdf_out, which segfaults on
    # finalisation in this build, and Path.glob order is not guaranteed.
    pars = sorted(p for p in passthrough if p.suffix == ".par")
    ascii_pars = [p for p in pars
                  if not any(l.strip().startswith("netcdf_out") for l in p.read_text().splitlines())]
    if not pars:
        raise SystemExit(f"no .par file found in {base}")
    if not ascii_pars:
        raise SystemExit(f"every .par in {base} sets netcdf_out, which segfaults; "
                         "stage an ASCII-output .par")
    par_name = ascii_pars[0].name
    if job_array:
        _emit_job_array(out_root, manifest, lisflood_bin, par_name,
                        account, partition, throttle,
                        modules=modules, cpus_per_task=cpus_per_task,
                        mem=mem, walltime=walltime)
        print(f"built {len(manifest)} runs -> {out_root}")
        print(f"  manifest.json (for emulator.dataset.build_manifest)")
        print(f"  run_array.sbatch + run_dirs.txt  ->  sbatch run_array.sbatch")
    else:
        subs = "\n".join(f"( cd {m['run_dir']} && sbatch run_lisflood.sbatch )" for m in manifest)
        (out_root / "submit_all.sh").write_text("#!/usr/bin/env bash\n" + subs + "\n")
        print(f"built {len(manifest)} runs -> {out_root}\n  manifest.json + submit_all.sh")
    return manifest


def _emit_job_array(out_root, manifest, lisflood_bin, par_name, account, partition, throttle,
                    modules="gcc/12.3.0 netcdf-c/4.9.2-gszew36", cpus_per_task=8,
                    mem="4G", walltime="01:30:00"):
    """One SLURM array over all run dirs: task i cd's into run_dirs.txt line i and runs
    LISFLOOD. Concurrency throttled with %throttle. Replaces N separate sbatch calls."""
    out_root = Path(out_root)
    # absolute paths: the array job cd's into these, and SLURM's working directory is the
    # submit directory, which is not necessarily where the sweep was generated from
    (out_root / "run_dirs.txt").write_text(
        "\n".join(str(Path(m["run_dir"]).resolve()) for m in manifest) + "\n")
    n = len(manifest)
    # A bad binary path here fails every member identically, so refuse to emit rather than
    # let 1506 tasks die with "command not found". The bare default "lisflood" is not on PATH
    # on a compute node, and the config default is a literal placeholder.
    if not str(lisflood_bin).startswith("/") or "/path/to/" in str(lisflood_bin):
        raise SystemExit(
            f"lisflood_bin is {lisflood_bin!r}, which will not resolve on a compute node. "
            "Pass --config so hpc.lisflood_bin from configs/base.yaml is used, or set an "
            "absolute path there.")
    # the default directive is only used when submitting with no explicit --array, so cap it
    arr_n = min(n, 1000)
    sbatch = f"""#!/usr/bin/env bash
#SBATCH -J coral_sweep
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH --array=1-{arr_n}%{throttle}
#SBATCH -N 1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH -t {walltime}
#SBATCH -o %x_%A_%a.out
# The binary is dynamically linked against these; a compute node starts with a clean module
# environment and the loader fails without them. Loaded before `set -e` because module init
# is not always clean under `set -u`.
module load {modules}
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited
# SLURM's MaxArraySize caps the maximum task INDEX, not the number of tasks: with
# MaxArraySize=1001 the largest legal index is 1000, so `--array=1002-1506` is rejected
# outright. Ensembles larger than that are submitted as several arrays that all use low
# indices, with IDX_OFFSET selecting which block of run_dirs.txt each one reads.
OFFSET=${{IDX_OFFSET:-0}}
LINE=$((SLURM_ARRAY_TASK_ID + OFFSET))
RUN=$(sed -n "${{LINE}}p" "{out_root.resolve()}/run_dirs.txt")
[ -n "$RUN" ] && [ -d "$RUN" ] || {{ echo "no run dir for line $LINE (task $SLURM_ARRAY_TASK_ID + offset $OFFSET)"; exit 1; }}
echo "task $SLURM_ARRAY_TASK_ID + offset $OFFSET -> line $LINE -> $RUN"
cd "$RUN"
{lisflood_bin} {par_name}
# A nonzero exit is not conclusive in this build (it can segfault at finalisation after
# writing output), and a zero exit does not prove the target field exists. Check the artefact.
ls results_*/*.max >/dev/null 2>&1 || {{ echo "FAILED: no .max in $RUN"; exit 1; }}
# Report the output size. A correct member writes about 55 MB of end-of-run fields. If this
# reads in the GB, the par is writing per-timestep snapshots: absent `saveint` does NOT mean
# off, LISFLOOD falls back to a 1000 s default, so the keyword must be present and larger
# than the run length. Checking only that .max exists passes either way, which is how a
# 5.7 TB overrun went unnoticed through a green smoke test.
SZ=$(du -sm results_* 2>/dev/null | awk '{{s+=$1}} END{{print s+0}}')
NSNAP=$(ls results_*/*-[0-9]*.wd 2>/dev/null | wc -l)
echo "ok: $(ls results_*/*.max)  outputs ${{SZ}} MB, ${{NSNAP}} snapshots"
[ "$NSNAP" -gt 0 ] && echo "WARNING: $NSNAP snapshots written; check saveint in the par"
exit 0
"""
    (out_root / "run_array.sbatch").write_text(sbatch)


def _read_flood_depth(path, dem_path):
    """Read a baseline `.max` for targeted siting, on the DEM's grid.

    The path may contain shell variables (`${BASE}/...`) so a config can name the base run
    without hard-coding an absolute cluster path. The grid must match the DEM cell for cell:
    siting indexes the depth array with DEM-shaped masks, so a mismatch would silently
    misplace every intervention rather than fail. That matters most for the 4 m ensemble,
    whose baseline `.max` comes from the 4 m baseline run and not from the 30 m one.
    """
    p = Path(os.path.expandvars(str(path)))
    if not p.exists():
        raise SystemExit(f"interventions.flood_depth does not exist: {p}\n"
                         "  (unexpanded ${BASE}? export it, or write the full path)")
    dem_h = _asc_header(dem_path)
    dep_h = _asc_header(p)
    for k in ("ncols", "nrows"):
        if dem_h[k] != dep_h[k]:
            raise SystemExit(
                f"flood_depth grid {p.name} is {dep_h['ncols']:g}x{dep_h['nrows']:g} but the "
                f"DEM {Path(dem_path).name} is {dem_h['ncols']:g}x{dem_h['nrows']:g}. Targeted "
                "siting needs the baseline .max from a run on this same grid.")
    a = np.loadtxt(p, skiprows=6)
    return np.where(a <= -9990, 0.0, a)


def _asc_header(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    return h


def from_config(cfg, base_dir, out_root, *, root="res_matthew_sav", nlcd=None):
    """Build the ensemble from a scenario's `interventions` config (Phase 3). Resolves the
    SAGIS context rasters (wetlands/soil_ksat/buildings) from the config paths and the DEM."""
    iv = cfg.interventions
    if iv is None:
        raise SystemExit(f"scenario {cfg.name!r} has no `interventions:` block")
    base = Path(base_dir); dem_p = next(base.glob("SUB_DEM*.asc"))
    wet = sk = bld = None
    from ..interventions.context_rasters import wetlands_mask, soil_ksat_grid, buildings_mask
    if iv.wetlands:
        wet = wetlands_mask(iv.wetlands, str(dem_p))
    if iv.soils_geojson and iv.ssurgo_table:
        sk = soil_ksat_grid(iv.soils_geojson, iv.ssurgo_table, str(dem_p))
    if iv.buildings:
        bld = buildings_mask(iv.buildings, str(dem_p))
    fdep = fz = None
    if iv.siting == "targeted":                          # resolve the targeting drivers
        if not iv.flood_depth:
            raise SystemExit(
                f"scenario {cfg.name!r} sets siting: targeted but no interventions.flood_depth. "
                "Targeted siting ranks candidate cells by baseline peak depth, so it needs the "
                "no-adaptation .max on this same grid, e.g.\n"
                "  flood_depth: ${BASE}/results_matthew_sav/res_matthew_sav.max")
        fdep = _read_flood_depth(iv.flood_depth, dem_p)
        if iv.flood_zone:
            fz = buildings_mask(os.path.expandvars(iv.flood_zone), str(dem_p))
    slr_levels_all = list(iv.slr_levels)
    if iv.slr_scenarios:                          # additive: resolve scenarios onto slr_levels
        from ..preprocess.fetch_slr import slr_levels as resolve_slr_scenarios
        targets = [(scen, year) for scen, year in iv.slr_scenarios]
        slr_levels_all += resolve_slr_scenarios(targets, station=cfg.forcing.tide_station)
    specs = plan_sweep(slr_levels_all, iv.kinds, iv.n_per_kind, iv.include_combos, iv.seed)
    return build_sweep(base_dir, specs, out_root, root=root, sea_level=cfg.geoclaw.sea_level,
                       focus_center=cfg.domain.ref_point,
                       focus_radius_km=iv.focus_radius_km or cfg.domain.focus_radius_km,
                       nlcd=nlcd, wetlands=wet, soil_ksat=sk, buildings=bld,
                       place=iv.siting, flood_depth=fdep, flood_zone=fz,
                       res_m=cfg.domain.res_m,
                       modules=cfg.hpc.modules, cpus_per_task=cfg.hpc.cpus_per_task,
                       mem=cfg.hpc.mem, walltime=cfg.hpc.walltime,
                       throttle=cfg.hpc.throttle,
                       lisflood_bin=cfg.hpc.lisflood_bin, account=cfg.hpc.account,
                       partition=cfg.hpc.partition)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the emulator training sweep")
    ap.add_argument("--base", required=True, help="Matthew run dir with inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="scenario YAML with an `interventions:` block (Phase 3)")
    ap.add_argument("--nlcd", default=None)
    ap.add_argument("--slr", nargs="+", type=float, default=[0.0, 0.3, 0.6, 1.0, 1.5])
    ap.add_argument("--kinds", nargs="+",
                    default=["seawall", "marsh", "mangrove", "permeable", "retreat"])
    ap.add_argument("--n-per-kind", type=int, default=4)
    a = ap.parse_args()
    if a.config:                                   # config-driven (SAGIS-conditioned) ensemble
        from ..config import load
        from_config(load(a.config), a.base, a.out, nlcd=a.nlcd)
    else:                                          # ad-hoc generic sweep
        specs = plan_sweep(a.slr, a.kinds, a.n_per_kind)
        build_sweep(a.base, specs, a.out, nlcd=a.nlcd)
