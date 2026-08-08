"""Compare running LISFLOOD jobs against a reference run.

Reads the .mass files of several runs in one directory and reports progress, mass
conservation, and the water balance at matched model times. Use it to check a long run
before it finishes, without waiting for output rasters.

The reference is normally the run that was validated against high-water marks. Its Area
and Vol are what a new configuration should be judged against, at the same model time.

    python -m coral.analysis.monitor_runs --base $BASE \\
        --ref results_matthew_sav --runs results_ext results_tide

Mass column order (LISFLOOD 8.0.3):
    Time Tstep MinTstep NumTsteps Area Vol Qin Hds Qout Qerror Verror Rain-(Inf+Evap)
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np

COLS = ("time", "tstep", "mintstep", "nsteps", "area", "vol",
        "qin", "hds", "qout", "qerror", "verror", "rain")

# Verror/Vol above this means the run is not conserving mass. The validated 30 m run sits
# at 1e-16; a failing configuration reached 1e-1.
BAD_RATIO = 1e-6


def read_mass(path):
    """Return a dict of column arrays from a .mass file.

    Rows before a 'Checkpoint restart' banner are kept: the banner only marks a run
    starting at a nonzero time (lisflood.cpp:565), not a restart from saved state.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.split()
        if len(s) < 11 or not s[0][0].isdigit():
            continue
        rows.append([float(v) for v in s[:12]])
    if not rows:
        return None
    a = np.array(rows)
    return {c: a[:, i] for i, c in enumerate(COLS[:a.shape[1]])}


def summarise(base, name, sim_time=None):
    """One line of run state, or None if the run has produced nothing."""
    d = Path(base) / name
    m = read_mass(d / f"res_matthew_sav.mass") if (d / "res_matthew_sav.mass").exists() \
        else next((read_mass(p) for p in sorted(d.glob("*.mass"))), None)
    if m is None:
        return None
    t = m["time"][-1]
    ratio = np.abs(m["verror"] / np.where(m["vol"] != 0, m["vol"], np.nan))
    tail = ratio[-10:]
    nwd = len(glob.glob(str(d / "*.wd")))
    out = {"name": name, "t": t, "area": m["area"][-1], "vol": m["vol"][-1],
           "depth": m["vol"][-1] / m["area"][-1] if m["area"][-1] else np.nan,
           "ratio_max": np.nanmax(ratio), "ratio_now": np.nanmax(tail),
           "tstep": m["tstep"][-1], "snapshots": nwd, "mass": m}
    if sim_time:
        out["pct"] = 100.0 * (t - m["time"][0]) / (sim_time - m["time"][0])
    return out


def at_time(m, t):
    """Area, Vol and mean depth interpolated to model time t. NaN if out of range."""
    if t < m["time"][0] or t > m["time"][-1]:
        return np.nan, np.nan, np.nan
    a = float(np.interp(t, m["time"], m["area"]))
    v = float(np.interp(t, m["time"], m["vol"]))
    return a, v, (v / a if a else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="directory holding the results dirs")
    ap.add_argument("--ref", default="results_matthew_sav",
                    help="reference run, normally the one validated against high-water marks")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sim-time", type=float, default=691200.0)
    a = ap.parse_args()

    ref = summarise(a.base, a.ref)
    rows = [(n, summarise(a.base, n, a.sim_time)) for n in a.runs]

    print(f"{'run':22s} {'t (s)':>9s} {'%':>5s} {'Tstep':>6s} {'Area':>10s} "
          f"{'Vol':>10s} {'depth':>6s} {'|Ver/Vol| now':>13s} {'max':>9s} {'snaps':>6s}")
    if ref:
        print(f"{a.ref + ' (ref)':22s} {ref['t']:9.0f} {'':>5s} {ref['tstep']:6.2f} "
              f"{ref['area']:10.3e} {ref['vol']:10.3e} {ref['depth']:6.2f} "
              f"{ref['ratio_now']:13.1e} {ref['ratio_max']:9.1e} {ref['snapshots']:6d}")
    for n, r in rows:
        if r is None:
            print(f"{n:22s}  no output yet")
            continue
        flag = "  <-- NOT CONSERVING" if r["ratio_now"] > BAD_RATIO else ""
        print(f"{n:22s} {r['t']:9.0f} {r.get('pct', float('nan')):5.1f} {r['tstep']:6.2f} "
              f"{r['area']:10.3e} {r['vol']:10.3e} {r['depth']:6.2f} "
              f"{r['ratio_now']:13.1e} {r['ratio_max']:9.1e} {r['snapshots']:6d}{flag}")

    # Water balance against the reference, at model times both runs have reached.
    if ref:
        print(f"\nversus {a.ref}, at matched model times:")
        for n, r in rows:
            if r is None:
                continue
            tmax = min(r["t"], ref["t"])
            ts = [t for t in (100000., 150000., 200000., 259200.) if t <= tmax]
            if not ts:
                print(f"  {n:22s} no overlap yet (ref ends at {ref['t']:.0f} s)")
                continue
            parts = []
            for t in ts:
                ra, rv, rd = at_time(ref["mass"], t)
                ca, cv, cd = at_time(r["mass"], t)
                parts.append(f"t={t/1000:5.0f}k area x{ca/ra:4.2f} vol x{cv/rv:4.2f} "
                             f"depth {cd:4.2f} vs {rd:4.2f}")
            print(f"  {n:22s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
