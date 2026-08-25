"""Stage a matched, snapshot-enabled pair from two existing LISFLOOD runs.

The source runs are never modified.  Regular input files are copied, symlinked inputs retain
their resolved target, results directories are omitted, and the par is rewritten only to enable
depth snapshots.  Use this for a native-grid intervention diagnostic where both arms already
exist in an ensemble (for example an Int2050 baseline and living-shoreline member).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .roughness_ablation import par_inputs


def stage_one(source: Path, dest: Path, snapshot_s: float):
    par, _ = par_inputs(source)
    dest.mkdir(parents=True)
    for p in source.iterdir():
        if p.name.startswith("results_") or p.name == par.name:
            continue
        q = dest / p.name
        if p.is_symlink():
            q.symlink_to(p.resolve())
        elif p.is_file():
            # Inputs edited by the intervention must be frozen into the diagnostic.  Copying the
            # small native-grid files also protects the pair from later ensemble housekeeping.
            shutil.copy2(p, q)
    lines = [ln for ln in par.read_text().splitlines()
             if ln.split()[:1] not in (["saveint"], ["qoutput"])]
    lines += [f"saveint        {snapshot_s:g}", "qoutput"]
    (dest / par.name).write_text("\n".join(lines) + "\n")
    return par.name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True)
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--selection-rule", required=True)
    ap.add_argument("--snapshot-s", type=float, default=1800.0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"output already exists: {out}; move it aside explicitly")
    control, intervention = Path(a.control), Path(a.intervention)
    cp = stage_one(control, out / "control", a.snapshot_s)
    ip = stage_one(intervention, out / "intervention", a.snapshot_s)
    if cp != ip:
        raise SystemExit(f"par names differ: control {cp}, intervention {ip}")
    report = {"label": a.label, "selection_rule": a.selection_rule,
              "control_source": str(control.resolve()),
              "intervention_source": str(intervention.resolve()),
              "snapshot_s": a.snapshot_s, "par": cp}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"staged pair -> {out}")


if __name__ == "__main__":
    main()
