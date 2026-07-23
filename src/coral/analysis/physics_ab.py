"""Physics A/B: test whether the NWI marsh-roughness Manning upgrade improves the HWM fit.

The comparison (reports/physics/) showed NLCD undersells marsh roughness (median n
0.061->0.110 over 54.7% of Pin Point cells), concentrated in the coastal fringe where the
model runs. This harness tests whether swapping in the NWI-refined Manning field actually
improves the modelled fit to observed high-water marks, as a controlled A/B on the existing
30 m compound run rather than a blind rerun.

  assemble(...)    -> stage baseline/ (current Manning) and nwi/ (NWI-refined Manning) run
                      dirs from a prepared compound base run; prints the HPC run commands.
  compare_hwm(...) -> after both run, sample each .mxe at USGS Matthew HWMs and report
                      bias + RMSE for each, and which wins. Runnable standalone.

Manning-only by design: per the full-domain finding, do not also globally swap POLARIS to
SSURGO infiltration (Ksat sign reverses off Pin Point; storage agreement doesn't generalize).
Infiltration is a separate, region-specific experiment.
"""
from __future__ import annotations
from pathlib import Path
import json, shutil, subprocess, urllib.request
import numpy as np

FT2M = 0.3048
USGS_EVENT = 135  # 2016 Hurricane Matthew (USGS STN)


def _read_grid(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.where((a == h.get("nodata_value", -9999)) | (a <= -9990), np.nan, a)
    return a, h


def _extent(h):
    nx, ny, cs = int(h["ncols"]), int(h["nrows"]), h["cellsize"]
    return [h["xllcorner"], h["xllcorner"] + nx * cs, h["yllcorner"], h["yllcorner"] + ny * cs]


# ---------------------------------------------------------------- assemble
def assemble(base_run, out_root, *, name, nlcd=None, wetlands=None,
             dem=None, lisflood_bin="lisflood"):
    """Stage baseline/ and nwi/ run dirs from a prepared compound `base_run`.

    baseline/ = verbatim copy. nwi/ = copy with Manning_{name}.asc regenerated with the
    NWI overlay. If `nlcd`+`wetlands`+`dem` are given, Manning is regenerated here via
    `coral.preprocess.make_manning --nwi`; otherwise the command to run is printed.
    """
    base, out = Path(base_run), Path(out_root); out.mkdir(parents=True, exist_ok=True)
    b, v = out / "baseline", out / "nwi"
    for d in (b, v):
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(base, d)

    man = v / f"Manning_{name}.asc"
    cmd = ["python", "-m", "coral.preprocess.make_manning", "--dem", str(dem or base / f'SUB_DEM_{name}.asc'),
           "--nlcd", str(nlcd), "--nwi", str(wetlands), "--out", str(man)]
    if nlcd and wetlands and dem:
        print("regenerating NWI-refined Manning:\n  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("stage-only (no --nlcd/--wetlands/--dem given). Regenerate the nwi Manning with:\n  "
              + " ".join(cmd))

    print("\nRun both (HPC):")
    print(f"  cd {b} && {lisflood_bin} {name}.par")
    print(f"  cd {v} && {lisflood_bin} {name}.par")
    print(f"then: python -m coral.analysis.physics_ab compare "
          f"{b}/results_{name}/res_{name}.mxe {v}/results_{name}/res_{name}.mxe "
          f"--dem {dem or base / f'SUB_DEM_{name}.asc'}")
    return {"baseline": str(b), "nwi": str(v)}


# ---------------------------------------------------------------- HWM compare
def _fetch_hwms(bbox):
    url = f"https://stn.wim.usgs.gov/STNServices/Events/{USGS_EVENT}/HWMs.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.loads(r.read().decode())
    w, e, s, n = bbox
    out = []
    for h in d:
        lon, lat, ev = h.get("longitude_dd"), h.get("latitude_dd"), h.get("elev_ft")
        if lon and lat and ev and w < lon < e and s < lat < n:
            out.append((lon, lat, ev * FT2M))
    return out


def _sample(mxe, h, lon, lat):
    cs, ny = h["cellsize"], int(h["nrows"])
    col = int((lon - h["xllcorner"]) / cs)
    row = int(((h["yllcorner"] + ny * cs) - lat) / cs)
    if 0 <= row < mxe.shape[0] and 0 <= col < mxe.shape[1]:
        return mxe[row, col]
    return np.nan


def _stats(mxe, h, hwms):
    obs, mod = [], []
    for lon, lat, o in hwms:
        m = _sample(mxe, h, lon, lat)
        if np.isfinite(m):
            obs.append(o); mod.append(m)
    obs, mod = np.array(obs), np.array(mod)
    if len(obs) == 0:
        return None
    err = mod - obs
    return {"n": len(obs), "bias_m": round(float(err.mean()), 3),
            "rmse_m": round(float(np.sqrt((err ** 2).mean())), 3)}


def compare_hwm(mxe_baseline, mxe_nwi, dem, *, out_fig=None):
    """Sample both .mxe at USGS Matthew HWMs; report bias + RMSE and the winner."""
    _, hd = _read_grid(dem)
    bbox = _extent(hd)
    hwms = _fetch_hwms(bbox)
    a, ha = _read_grid(mxe_baseline)
    v, hv = _read_grid(mxe_nwi)
    sb, sv = _stats(a, ha, hwms), _stats(v, hv, hwms)
    if not sb or not sv:
        raise SystemExit(f"no HWMs sampled in domain ({len(hwms)} HWMs in bbox {bbox})")
    print(f"HWMs in domain: {len(hwms)}")
    print(f"  baseline (NLCD Manning): bias {sb['bias_m']:+.3f} m, RMSE {sb['rmse_m']:.3f} m (n={sb['n']})")
    print(f"  nwi      (NWI Manning) : bias {sv['bias_m']:+.3f} m, RMSE {sv['rmse_m']:.3f} m (n={sv['n']})")
    better = "NWI" if sv["rmse_m"] < sb["rmse_m"] else "baseline"
    print(f"VERDICT: {better} fits HWMs better "
          f"(RMSE {min(sb['rmse_m'], sv['rmse_m']):.3f} vs {max(sb['rmse_m'], sv['rmse_m']):.3f} m)")
    res = {"baseline": sb, "nwi": sv, "better": better}

    if out_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        for mxe, hh, lab, c in ((a, ha, f"baseline RMSE {sb['rmse_m']}", "#888"),
                                (v, hv, f"NWI RMSE {sv['rmse_m']}", "#2a7")):
            o = [oo for lon, lat, oo in hwms if np.isfinite(_sample(mxe, hh, lon, lat))]
            m = [_sample(mxe, hh, lon, lat) for lon, lat, oo in hwms
                 if np.isfinite(_sample(mxe, hh, lon, lat))]
            ax.scatter(o, m, s=20, c=c, label=lab, alpha=.7)
        lim = [0, max(max(o for _, _, o in hwms), 4)]
        ax.plot(lim, lim, "k--", lw=.8); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("observed HWM WSE (m)"); ax.set_ylabel("modelled max WSE (m)")
        ax.legend(); ax.set_title("HWM fit: baseline vs NWI Manning")
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_fig, dpi=150); plt.close(fig)
        print(f"  wrote {out_fig}"); res["figure"] = out_fig
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Physics A/B: NWI Manning vs baseline, HWM fit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("mxe_baseline"); c.add_argument("mxe_nwi")
    c.add_argument("--dem", required=True)
    c.add_argument("--out-fig", default="reports/physics/physics_ab_hwm.png")
    a = ap.parse_args()
    if a.cmd == "compare":
        compare_hwm(a.mxe_baseline, a.mxe_nwi, a.dem, out_fig=a.out_fig)
