"""Tier-0 vs Tier-1 drainage sensitivity test: the gate on whether SWMM (Tier 2) is worth it.

Test whether representing storm-drainage changes the Pin Point compound flood enough to
justify a coupled SWMM model. We run the same scenario twice and compare:
  - Tier 0: no drainage (baseline). SGC channels off, no drain point sources.
  - Tier 1: open channels as SGC (burn_drainage) + tide-conditioned inlet sinks
            (make_drainage_proxy). Native LISFLOOD only.
If Tier-1 barely moves the flood relative to Tier-0, SWMM is unjustified. If it moves it a
lot, that is the evidence to build Tier 2. Full plan: wiki `Drainage backup and SWMM coupling`.

Two entry points:
  assemble(...)  - stage the two run dirs (needs a prepared base run + the SAGIS vectors).
                   The LISFLOOD binary run itself happens on HPC (prints the commands).
  compare(...)   - load the two flood-max rasters, compute metrics + a diff figure + a
                   go/no-go verdict. Runnable standalone on any pair of .max/.asc rasters.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np

from ..preprocess.burn_drainage import burn_drainage
from ..preprocess.make_drainage_proxy import make_drainage_proxy


# ---------------------------------------------------------------- raster io
def _read_asc(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    return np.where(a <= -9990, np.nan, a), h


# ---------------------------------------------------------------- .bci/.bdy merge
def _merge_bci(base_bci, drain_bci, out_bci):
    """Concatenate point-source P-lines (coastline HVAR + drainage QVAR)."""
    txt = Path(base_bci).read_text().rstrip("\n") + "\n" + Path(drain_bci).read_text()
    Path(out_bci).write_text(txt)


def _merge_bdy(base_bdy, drain_bdy, out_bdy):
    """One .bdy holds all series blocks; both files start with a 'comment' header line
    that must appear only once."""
    base = Path(base_bdy).read_text().splitlines()
    drain = Path(drain_bdy).read_text().splitlines()
    merged = base + drain[1:]                      # drop drain's duplicate header line
    Path(out_bdy).write_text("\n".join(merged) + "\n")


# ---------------------------------------------------------------- assemble variants
def assemble(base_run, out_root, *, name, channels, inlets, stage_series,
             capacity=0.05, outfall_invert=0.0, backup_frac=0.0, sgc_kind="ditch",
             lisflood_bin="lisflood"):
    """Stage tier0/ and tier1/ run dirs from a prepared `base_run` dir.

    `base_run` must already contain the standard staged inputs the .par expects:
    SUB_DEM_{name}.asc, Manning_{name}.asc, {name}_coastline.bci, {name}.bdy, the .par
    (named {name}.par), and any rain/infil files. Tier 0 is a straight copy. Tier 1 adds
    the SGC width raster + merged drainage point sources and enables SGCwidth in the .par.
    Returns the two run-dir paths and the shell commands to run each on HPC.
    """
    base = Path(base_run); out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
    dem = base / f"SUB_DEM_{name}.asc"
    par = base / f"{name}.par"
    t0, t1 = out / "tier0", out / "tier1"

    # --- Tier 0: verbatim copy (baseline, no drainage) ---
    if t0.exists():
        shutil.rmtree(t0)
    shutil.copytree(base, t0)

    # --- Tier 1: copy, then add drainage ---
    if t1.exists():
        shutil.rmtree(t1)
    shutil.copytree(base, t1)
    sgc = t1 / f"SGCwidth_{name}.asc"
    burn_drainage(str(dem), channels, str(sgc), kind=sgc_kind, method="sgc")
    make_drainage_proxy(inlets, stage_series, str(t1 / "_drain.bci"), str(t1 / "_drain.bdy"),
                        capacity=capacity, outfall_invert=outfall_invert, backup_frac=backup_frac)
    _merge_bci(t1 / f"{name}_coastline.bci", t1 / "_drain.bci", t1 / f"{name}_coastline.bci")
    _merge_bdy(t1 / f"{name}.bdy", t1 / "_drain.bdy", t1 / f"{name}.bdy")
    # enable the SGC width raster in the tier-1 .par (base already has `sgc_enable`)
    par_txt = (t1 / f"{name}.par").read_text()
    if "SGCwidth" not in par_txt:
        par_txt = par_txt.replace("sgc_enable", f"sgc_enable\nSGCwidth        SGCwidth_{name}.asc")
        (t1 / f"{name}.par").write_text(par_txt)

    cmds = [f"cd {t0} && {lisflood_bin} {name}.par", f"cd {t1} && {lisflood_bin} {name}.par"]
    print("\nRun both (HPC / wherever the LISFLOOD binary lives):")
    for c in cmds:
        print("  " + c)
    print(f"then: coral-drainage-compare {t0}/results_{name}/res_{name}.max "
          f"{t1}/results_{name}/res_{name}.max")
    return {"tier0": str(t0), "tier1": str(t1), "commands": cmds}


# ---------------------------------------------------------------- compare + verdict
def compare(max_tier0, max_tier1, *, wet_thresh=0.05, decision_pct=10.0, out_fig=None):
    """Compare two flood-max-depth rasters. Returns a metrics/verdict dict.

    Metrics: wet area (cells >= wet_thresh m), flood volume (sum depth), mean/max depth,
    and the Tier1-Tier0 difference field. Verdict: if the larger of |Δarea%| and |Δvolume%|
    exceeds `decision_pct`, drainage matters enough to justify building SWMM (Tier 2).
    """
    d0, h = _read_asc(max_tier0)
    d1, _ = _read_asc(max_tier1)
    if d0.shape != d1.shape:
        raise SystemExit(f"raster shapes differ: {d0.shape} vs {d1.shape}")
    a0, a1 = np.nan_to_num(d0), np.nan_to_num(d1)
    wet0, wet1 = a0 >= wet_thresh, a1 >= wet_thresh
    cell_area = h["cellsize"] ** 2                       # deg^2 if latlong; still comparable
    area0, area1 = wet0.sum() * cell_area, wet1.sum() * cell_area
    vol0, vol1 = a0[wet0].sum(), a1[wet1].sum()
    d_area = 100.0 * (area1 - area0) / area0 if area0 else 0.0
    d_vol = 100.0 * (vol1 - vol0) / vol0 if vol0 else 0.0
    swmm_justified = max(abs(d_area), abs(d_vol)) >= decision_pct

    res = {
        "wet_cells": (int(wet0.sum()), int(wet1.sum())),
        "area_pct_change": round(d_area, 2),
        "volume_pct_change": round(d_vol, 2),
        "mean_depth": (round(float(a0[wet0].mean() if wet0.any() else 0), 3),
                       round(float(a1[wet1].mean() if wet1.any() else 0), 3)),
        "max_depth": (round(float(a0.max()), 3), round(float(a1.max()), 3)),
        "decision_pct": decision_pct,
        "swmm_justified": bool(swmm_justified),
    }
    print(f"Tier0->Tier1: wet cells {res['wet_cells'][0]}->{res['wet_cells'][1]}, "
          f"area {d_area:+.1f}%, volume {d_vol:+.1f}%")
    print(f"VERDICT: SWMM (Tier 2) {'JUSTIFIED' if swmm_justified else 'NOT justified'} "
          f"(threshold |Δ| >= {decision_pct}%)")

    if out_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        diff = a1 - a0
        vmax = float(np.nanmax(np.abs(diff))) or 1.0
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for a, dd, t in ((ax[0], d0, "Tier 0 (no drainage)"), (ax[1], d1, "Tier 1 (drainage)")):
            im = a.imshow(dd, cmap="Blues", vmin=0); a.set_title(t); fig.colorbar(im, ax=a, shrink=.7)
        im = ax[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[2].set_title(f"Tier1 - Tier0 (m)\narea {d_area:+.1f}%  vol {d_vol:+.1f}%")
        fig.colorbar(im, ax=ax[2], shrink=.7)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_fig, dpi=150); plt.close(fig)
        print(f"  wrote {out_fig}")
        res["figure"] = out_fig
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compare Tier-0 vs Tier-1 flood-max rasters")
    ap.add_argument("max_tier0"); ap.add_argument("max_tier1")
    ap.add_argument("--wet-thresh", type=float, default=0.05)
    ap.add_argument("--decision-pct", type=float, default=10.0)
    ap.add_argument("--out-fig", default="reports/sagis/drainage_sensitivity.png")
    a = ap.parse_args()
    compare(a.max_tier0, a.max_tier1, wet_thresh=a.wet_thresh,
            decision_pct=a.decision_pct, out_fig=a.out_fig)
