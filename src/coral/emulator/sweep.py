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
    pairwise combos). Returns a list of {name, slr_m, interventions:[knobs,...]}."""
    rng = np.random.default_rng(seed)
    specs = []
    for slr in slr_levels:
        specs.append({"name": f"slr{slr}_base", "slr_m": slr, "interventions": []})
        for kind in kinds:
            for j in range(n_per_kind):
                kb = sample_intervention(kind, rng)
                specs.append({"name": f"slr{slr}_{kind}{j}", "slr_m": slr,
                              "interventions": [kb]})
        if include_combos:                       # a few multi-intervention configs
            for j in range(n_per_kind):
                combo = [sample_intervention(k, rng) for k in rng.choice(kinds, 2, replace=False)]
                specs.append({"name": f"slr{slr}_combo{j}", "slr_m": slr,
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
                place="random", flood_depth=None, flood_zone=None,
                job_array=True, lisflood_bin="lisflood", account="gts-arobel3-atlas",
                partition="cpu-medium", throttle=20):
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
                                                        flood_depth=flood_depth, flood_zone=flood_zone)
        write_ascii(str(run / dem_p.name), dem, str(dem_p))
        write_ascii(str(run / man_p.name), man, str(dem_p))
        if ksat_p:
            write_ascii(str(run / ksat_p.name), ksat, str(dem_p))
        if cap_p:
            write_ascii(str(run / cap_p.name), awc, str(dem_p))
        apply_slr_to_bdy(bdy_p, run / bdy_p.name, spec["slr_m"])
        for p in passthrough:
            if p.is_file():
                shutil.copy2(p, run / p.name)
        forcing = {"slr_m": spec["slr_m"]}
        for kb in spec["interventions"]:                    # flatten knobs as features
            forcing.update({f"{kb['kind']}_{k}": v for k, v in kb.items()
                            if k not in ("kind", "seed")})
        manifest.append({"name": spec["name"], "run_dir": str(run), "root": root,
                         "forcing": forcing, "interventions": spec["interventions"]})

    json.dump(manifest, open(out_root / "manifest.json", "w"), indent=2)
    par_name = next((p.name for p in passthrough if p.suffix == ".par"), "savannah.par")
    if job_array:
        _emit_job_array(out_root, manifest, lisflood_bin, par_name,
                        account, partition, throttle)
        print(f"built {len(manifest)} runs -> {out_root}")
        print(f"  manifest.json (for emulator.dataset.build_manifest)")
        print(f"  run_array.sbatch + run_dirs.txt  ->  sbatch run_array.sbatch")
    else:
        subs = "\n".join(f"( cd {m['run_dir']} && sbatch run_lisflood.sbatch )" for m in manifest)
        (out_root / "submit_all.sh").write_text("#!/usr/bin/env bash\n" + subs + "\n")
        print(f"built {len(manifest)} runs -> {out_root}\n  manifest.json + submit_all.sh")
    return manifest


def _emit_job_array(out_root, manifest, lisflood_bin, par_name, account, partition, throttle):
    """One SLURM array over all run dirs: task i cd's into run_dirs.txt line i and runs
    LISFLOOD. Concurrency throttled with %throttle. Replaces N separate sbatch calls."""
    out_root = Path(out_root)
    (out_root / "run_dirs.txt").write_text("\n".join(m["run_dir"] for m in manifest) + "\n")
    n = len(manifest)
    sbatch = f"""#!/usr/bin/env bash
#SBATCH -J coral_sweep
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH --array=1-{n}%{throttle}
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=12G
#SBATCH -t 08:00:00
#SBATCH -o %x_%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
RUN=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{out_root.resolve()}/run_dirs.txt")
echo "task $SLURM_ARRAY_TASK_ID -> $RUN"
cd "$RUN"
{lisflood_bin} {par_name}
"""
    (out_root / "run_array.sbatch").write_text(sbatch)


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
        if iv.flood_depth:
            a = np.loadtxt(iv.flood_depth, skiprows=6); fdep = np.where(a <= -9990, 0.0, a)
        if iv.flood_zone:
            fz = buildings_mask(iv.flood_zone, str(dem_p))
    specs = plan_sweep(iv.slr_levels, iv.kinds, iv.n_per_kind, iv.include_combos, iv.seed)
    return build_sweep(base_dir, specs, out_root, root=root, sea_level=cfg.geoclaw.sea_level,
                       focus_center=cfg.domain.ref_point,
                       focus_radius_km=iv.focus_radius_km or cfg.domain.focus_radius_km,
                       nlcd=nlcd, wetlands=wet, soil_ksat=sk, buildings=bld,
                       place=iv.siting, flood_depth=fdep, flood_zone=fz,
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
