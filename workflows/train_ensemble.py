#!/usr/bin/env python
"""Train the flood emulator on a finished LISFLOOD ensemble.

This is the step between `coral.emulator.sweep` (which writes the run dirs and a
manifest.json) and `coral.emulator.inference` (which turns a checkpoint into depth
rasters). 

The `coral.emulator` pieces need an entry point that takes an ensemble directory, figures out which members have finished running, and choose an approprate train/val split.

Steps:

  1. Reads manifest.json from every `--ensemble` directory.
  2. Reports coverage: how many members were planned, how many have a `.max`, and why the
     rest do not.
  3. Splits. The default holds out whole sea-level levels,
    `--val-ensemble` instead validates on a separate
     ensemble, which is the stronger test when that ensemble uses targeted siting while
     training used random placement.
  4. Trains the U-Net (`--arch unet`) or the GraphSAGE GNN (`--arch gnn`), early stopping
     on validation RMSE.
  5. Writes a JSON report next to the checkpoint: the split, the learning curve, and
     per-member RMSE and CSI on the held-out set, sorted worst first.

Usage:

    # train on the 30 m ensemble, holding out the two highest sea levels
    python workflows/train_ensemble.py --ensemble $SCR/runs/train30m \
        --holdout-slr IntHigh2100 High2100 --epochs 200 --ckpt emulator_unet.pt

    # validate on the separate targeted ensemble instead (the generalisation test)
    python workflows/train_ensemble.py --ensemble $SCR/runs/train30m \
        --val-ensemble $SCR/runs/valid30m --epochs 200

    # learning curve: is 50 realisations per adaptation enough, or do we need 100?
    python workflows/train_ensemble.py --ensemble $SCR/runs/train30m \
        --holdout-slr Int2070 --subsample-per-kind 10 25 50

Deps: torch (pip install -e ".[emulator]"). The GNN uses scatter-add over an edge list (torch_geometric can be used later if necessary),
so both architectures run in the same environment and the benchmark is not
confounded by library differences.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coral.emulator.dataset import FloodDataset, build_manifest, missing_runs  # noqa: E402
from coral.emulator import train as train_mod                                  # noqa: E402


# ------------------------------------------------------------------ loading and coverage

def load_runs(ensembles):
    """Concatenate the manifest entries of every ensemble dir, tagging each with its source
    so a report can tell members of different ensembles apart."""
    runs = []
    for d in ensembles:
        mf = Path(d) / "manifest.json"
        if not mf.exists():
            raise SystemExit(f"no manifest.json in {d}. Run coral.emulator.sweep first.")
        entries = json.load(open(mf))
        for r in entries:
            r["ensemble"] = str(Path(d).name)
        runs += entries
        print(f"  {mf}: {len(entries)} planned members")
    return runs


def coverage(runs, label):
    """Split planned members into finished and not, and summarise the reasons."""
    samples = build_manifest(runs, skip_missing=True)
    bad = missing_runs(runs)
    print(f"{label}: {len(samples)}/{len(runs)} members have a .max")
    if bad:
        reasons = {}
        for _, why in bad:
            reasons[why] = reasons.get(why, 0) + 1
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:5d}  {why}")
        unfinished = sum(n for w, n in reasons.items() if "not finished" in w)
        if unfinished != len(bad):
            print("    NOTE: some members are missing input grids, not just results. That is a"
                  " staging problem, not an unfinished run; check the sweep output.")
    if not samples:
        raise SystemExit(f"{label}: nothing to train on. Wait for the array, or check paths.")
    return samples


# ------------------------------------------------------------------ splitting

def kind_of(name):
    """Intervention kind from a sweep member name like `slrInt2050_marsh17` -> `marsh`.
    Baselines have no intervention and come back as `base`."""
    m = re.match(r"^slr[^_]+_([A-Za-z_]+?)\d*$", name)
    return m.group(1) if m else "base"


def realisation_of(name):
    m = re.match(r"^slr[^_]+_[A-Za-z_]+?(\d+)$", name)
    return int(m.group(1)) if m else -1


def slr_key(sample):
    """The label if the level came from a NOAA scenario, else the numeric offset as text.
    This is what `--holdout-slr` matches, so `Int2100` and `0.0` are both valid values."""
    lab = sample.forcing.get("slr_label")
    return lab if lab else f"{float(sample.forcing.get('slr_m', 0.0)):g}"


def realisation_index(name):
    """Trailing realisation number in a member name, e.g. slr0.0_marsh17 -> 17. None for baselines."""
    import re
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def split(samples, holdout_slr=(), holdout_kind=(), holdout_frac=0.0, seed=0,
          holdout_realisation=None):
    """Return (train, val) plus a description of how the split was made."""
    import numpy as np
    if holdout_realisation is not None:
        # Withhold high realisation indices across every kind and sea level. This tests unseen
        # PLACEMENTS of known intervention types, which is what a user does in the tool: draw a
        # wall somewhere the model has not seen. Holding out a sea level tests forcing
        # interpolation instead, and holding out a kind tests an unseen intervention type; the
        # three answer different questions and are reported separately.
        te = [s_ for s_ in samples
              if (realisation_index(s_.name) or -1) >= holdout_realisation]
        tr = [s_ for s_ in samples if s_ not in te]
        how = (f"held out realisations >= {holdout_realisation} across all kinds and sea levels "
               f"({len(te)} members): unseen placements of known types")
        return tr, te, how
    if holdout_slr or holdout_kind:
        hs, hk = set(holdout_slr), set(holdout_kind)
        held = {id(s) for s in samples if slr_key(s) in hs or kind_of(s.name) in hk}
        val = [s for s in samples if id(s) in held]
        tr = [s for s in samples if id(s) not in held]
        how = f"held out sea levels {sorted(hs) or '-'} and kinds {sorted(hk) or '-'}"
    elif holdout_frac:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(samples))
        n = max(1, int(round(holdout_frac * len(samples))))
        val = [samples[i] for i in idx[:n]]
        tr = [samples[i] for i in idx[n:]]
        how = (f"random {holdout_frac:.0%} holdout, seed {seed}. This measures fit, not "
               "generalisation: members at the same sea level are near-duplicates, so a "
               "random split leaves close neighbours of every held-out case in training.")
    else:
        return samples, None, "no holdout, training on everything"
    if not tr or not val:
        raise SystemExit(f"split left {len(tr)} train / {len(val)} val members. "
                         "Check that --holdout-slr values match the manifest labels.")
    return tr, val, how


def subsample_per_kind(samples, n):
    """Keep only the first `n` realisations of each intervention kind at each sea level,
    plus every baseline. Used for the learning curve: it answers whether the accuracy is
    still improving with ensemble size, and so whether 100 realisations is worth the
    extra 124 GB and 12 hours."""
    keep = []
    for s in samples:
        k, r = kind_of(s.name), realisation_of(s.name)
        if k == "base" or r < 0 or r < n:
            keep.append(s)
    return keep


# ------------------------------------------------------------------ reporting

def write_report(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2, default=str)
    print(f"report -> {path}")


def print_worst(rows, n=10):
    ranked = sorted(rows, key=lambda r: -(r["rmse_m"] if r["rmse_m"] == r["rmse_m"] else 0))
    print(f"  worst {min(n, len(ranked))} held-out members by RMSE:")
    for r in ranked[:n]:
        print(f"    {r['name']:32s} RMSE {r['rmse_m']:6.3f} m   CSI {r['csi']:.3f}")


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", nargs="+", required=True,
                    help="ensemble dir(s) containing manifest.json")
    ap.add_argument("--val-ensemble", nargs="+", default=None,
                    help="separate ensemble to validate on, e.g. the targeted valid30m set")
    ap.add_argument("--holdout-slr", nargs="+", default=[],
                    help="sea-level labels to hold out, e.g. IntHigh2100 High2100")
    ap.add_argument("--holdout-kind", nargs="+", default=[],
                    help="intervention kinds to hold out, e.g. living_shoreline")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="random holdout fraction; measures fit rather than generalisation")
    ap.add_argument("--subsample-per-kind", type=int, nargs="+", default=None,
                    help="train once per value, keeping that many realisations per kind, to "
                         "trace the learning curve against ensemble size")
    ap.add_argument("--holdout-realisation", type=int, default=None,
                    help="withhold realisation indices >= N across all kinds and sea levels; "
                         "tests unseen placements of known intervention types")
    ap.add_argument("--arch", choices=["unet", "gnn"], default="unet")
    ap.add_argument("--bci", default=None,
                    help="gnn: .bci marking boundary nodes, where surge and tide enter")
    ap.add_argument("--hidden", type=int, default=64, help="gnn: hidden width")
    ap.add_argument("--layers", type=int, default=8,
                    help="gnn: message-passing layers, i.e. how many cells information travels")
    ap.add_argument("--sub-nodes", type=int, default=60000,
                    help="gnn: nodes per sampled subgraph; the memory knob")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="U-Net width")
    ap.add_argument("--tail-gamma", type=float, default=1.0,
                    help="deep-cell loss weighting; 0 is plain masked MSE")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8,
                    help="evaluations without val improvement before stopping; 0 disables")
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers; the dataset is lazy so this hides grid I/O")
    ap.add_argument("--cache-dir", default=None,
                    help="directory for decoded .npy grids. Parsing ASCII costs 0.28 s per "
                         "grid and a sample reads five, so an epoch over 1506 members spends "
                         "about 35 min in the parser. Cached reloads are 113x faster. Keyed on "
                         "the resolved path, so symlinked base grids are stored once.")
    ap.add_argument("--eager", action="store_true",
                    help="cache decoded grids in RAM. About 43 MB per member on the 30 m "
                         "grid, so only for small ensembles")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N members")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="emulator_unet.pt")
    ap.add_argument("--report", default=None, help="default: <ckpt>.report.json")
    a = ap.parse_args(argv)

    if a.cache_dir:
        from coral.emulator.dataset import set_grid_cache
        d = set_grid_cache(a.cache_dir)
        print(f"decoded-grid cache: {d} (first epoch builds it, later epochs reuse it)")

    print("=== loading manifests ===")
    runs = load_runs(a.ensemble)
    samples = coverage(runs, "training pool")

    if a.val_ensemble:
        vruns = load_runs(a.val_ensemble)
        val = coverage(vruns, "validation pool")
        tr, how = samples, f"validated on a separate ensemble: {', '.join(a.val_ensemble)}"
    else:
        tr, val, how = split(samples, a.holdout_slr, a.holdout_kind,
                             a.holdout_frac, a.seed, a.holdout_realisation)

    if a.limit:
        tr = tr[:a.limit]

    print("\n=== split ===")
    print(f"  {how}")
    print(f"  train {len(tr)} members, validate {len(val) if val else 0}")
    levels = sorted({slr_key(s) for s in tr})
    print(f"  sea levels in training: {levels}")
    print(f"  kinds in training: {sorted({kind_of(s.name) for s in tr})}")
    if val:
        print(f"  sea levels held out: {sorted({slr_key(s) for s in val})}")

    schedule = a.subsample_per_kind or [None]
    results = []
    for n in schedule:
        tr_n = subsample_per_kind(tr, n) if n else tr
        tag = f"_n{n}" if n else ""
        ckpt = str(Path(a.ckpt).with_suffix("")) + tag + ".pt"
        print(f"\n=== training {a.arch}{f' on {n} realisations per kind' if n else ''} ===")
        print(f"  {len(tr_n)} training members -> {ckpt}")
        lazy = not a.eager
        tr_ds = FloodDataset(tr_n, lazy=lazy, seed=a.seed)
        va_ds = FloodDataset(val, stats=tr_ds.stats, lazy=lazy) if val else None
        print(f"  {tr_ds[0][0].shape[0]} input channels, grid {tuple(tr_ds[0][1].shape[-2:])}"
              f", {'eager' if a.eager else 'lazy'} loading")
        history = []
        if a.arch == "gnn":
            # Same members, same split, same loss as the U-Net path, so the comparison is
            # architecture only. Subgraph sampling is what makes a 1.2 M node grid fit.
            from coral.emulator import train_gnn as gnn_mod
            _, best = gnn_mod.train(
                tr_n, val or [], bci_path=a.bci, epochs=a.epochs, lr=a.lr,
                hidden=a.hidden, layers=a.layers, sub_nodes=a.sub_nodes,
                patience=a.patience or None, eval_every=a.eval_every,
                ckpt=ckpt, tail_gamma=a.tail_gamma, seed=a.seed, history=history)
            results.append({"n_per_kind": n, "n_train": len(tr_n), "ckpt": ckpt,
                            "history": history, "best_val_rmse": best, "arch": "gnn"})
            continue
        model = train_mod.train(
            tr_ds, val=va_ds, epochs=a.epochs, lr=a.lr, base=a.base,
            tail_gamma=a.tail_gamma, ckpt=ckpt, workers=a.workers,
            eval_every=a.eval_every, patience=a.patience or None, history=history)
        rows = []
        if va_ds is not None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            rows = train_mod.evaluate(model, va_ds, device, workers=a.workers)
            import numpy as np
            print(f"  held-out mean RMSE {np.nanmean([r['rmse_m'] for r in rows]):.3f} m, "
                  f"mean CSI {np.nanmean([r['csi'] for r in rows]):.3f}")
            print_worst(rows)
        results.append({"n_per_kind": n, "n_train": len(tr_n), "ckpt": ckpt,
                        "history": history, "holdout_metrics": rows})

    report = a.report or str(Path(a.ckpt).with_suffix("")) + ".report.json"
    write_report(report, {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "ensembles": a.ensemble, "val_ensembles": a.val_ensemble,
        "split": how, "n_planned": len(runs), "n_finished": len(samples),
        "arch": a.arch, "runs": results,
    })

    if len(schedule) > 1:
        print("\n=== learning curve ===")
        for r in results:
            best = min((h.get("val_rmse_m", float("inf")) for h in r["history"]),
                       default=float("nan"))
            print(f"  {r['n_per_kind']:4d} per kind ({r['n_train']:5d} members): "
                  f"best val RMSE {best:.3f} m")
        print("  If RMSE is still falling at the largest value, raise n_per_kind to 100 in"
              " configs/scenarios/pinpoint_train30m.yaml and extend the ensemble.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
