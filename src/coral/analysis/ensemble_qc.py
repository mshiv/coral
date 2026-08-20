"""Check an ensemble before running it.

An intervention that silently did nothing costs a member and teaches the emulator that the
intervention has no effect. A footprint in the wrong place costs more, because it looks fine in
aggregate. Both are cheap to catch before submission and expensive afterwards.

Five checks:

  noop      which members left every grid untouched. sweep symlinks grids the intervention did
            not change and writes real files for the ones it did, so this is a filesystem test
            and runs over the whole ensemble in seconds.
  panels    baseline, edited and difference for a sample member of each kind
  coverage  every sampled member's footprint per kind, random against targeted
  section   a transect through an elevation edit, showing the crest profile
  knobs     sampled knob values against the registry ranges

    python -m coral.analysis.ensemble_qc noop --ens <dir>
    python -m coral.analysis.ensemble_qc panels --ens <dir> --base <baseline dir>
    python -m coral.analysis.ensemble_qc coverage --ens <dir> --base <baseline dir>
    python -m coral.analysis.ensemble_qc section --ens <dir> --base <baseline dir>
    python -m coral.analysis.ensemble_qc knobs --ens <dir>
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .physics_ab import _read_grid
from ..viz.pinpoint_style import PALETTE

# which grid each kind is expected to edit first
PRIMARY = {"floodwall": "SUB_DEM", "road_raise": "SUB_DEM", "retreat": "SUB_DEM",
           "living_shoreline": "SUB_DEM", "marsh_restoration": "Manning",
           "marsh_migration": "Manning", "depave": "Manning"}
ELEV_KINDS = ("floodwall", "road_raise", "living_shoreline", "retreat")


def load(ens):
    ens = Path(ens)
    m = json.load(open(ens / "manifest.json"))
    return ens, m


def kinds_of(entry):
    return sorted({i["kind"] for i in (entry.get("interventions") or [])})


def grid_path(run_dir, field, name):
    """The member's copy of a grid. sweep names them <field>_<scenario>.asc."""
    hits = sorted(Path(run_dir).glob(f"{field}_*.asc"))
    return hits[0] if hits else None


# ---------------------------------------------------------------- noop
def check_noop(ens, manifest):
    """Members whose grids are all symlinks: the intervention changed nothing.

    sweep writes a real file only when the edited grid differs from the base, so a symlink means
    the edit was a no-op. Catches the failures that complete successfully: a floodwall alignment
    with no tie-in after 12 retries, a marsh zone that came out empty, a crest below the member's
    own waterline.
    """
    rows, by_kind = [], defaultdict(lambda: [0, 0])
    for e in manifest:
        ks = kinds_of(e)
        if not ks:
            continue                                   # baselines edit nothing by design
        run = Path(e["run_dir"])
        if not run.exists():
            rows.append((e["name"], ks, "MISSING run dir"))
            continue
        # Check the grid each kind is SUPPOSED to edit, not merely whether any file is real.
        # "any real .asc" reported 100 percent for floodwall while every floodwall DEM was an
        # unedited symlink, because sweep had written the wall into a differently-named grid.
        for k in ks:
            by_kind[k][1] += 1
            want = PRIMARY.get(k, "Manning")
            hits = [q for q in run.glob(f"{want}_*.asc") if not q.is_symlink()]
            if hits:
                by_kind[k][0] += 1
            else:
                rows.append((e["name"], [k], f"{want} not edited"))

    n_iv = sum(1 for e in manifest if kinds_of(e))
    print(f"{n_iv:,} intervention members, {len(manifest) - n_iv} baselines\n")
    print(f"{'kind':20s} {'edited':>8} {'members':>8}  {'rate':>6}")
    for k in sorted(by_kind):
        ed, tot = by_kind[k]
        flag = "" if ed == tot else "   <-- some did nothing"
        print(f"{k:20s} {ed:8d} {tot:8d}  {100*ed/max(tot,1):5.1f}%{flag}")
    if rows:
        print(f"\n{len(rows)} member(s) with no edit:")
        for n, ks, why in rows[:25]:
            print(f"  {n:28s} {','.join(ks):30s} {why}")
        if len(rows) > 25:
            print(f"  ... and {len(rows)-25} more")
    else:
        print("\nevery intervention member edited at least one grid")
    return rows


# ---------------------------------------------------------------- shared plotting
def _diff(run, base, field, thresh=1e-9):
    a = grid_path(base, field, None)
    b = grid_path(run, field, None)
    if a is None or b is None:
        return None, None, None
    za, h = _read_grid(a); zb, _ = _read_grid(b)
    za = np.asarray(za, float); zb = np.asarray(zb, float)
    za = np.where(za < -9000, np.nan, za); zb = np.where(zb < -9000, np.nan, zb)
    d = zb - za
    d = np.where(np.abs(d) > thresh, d, np.nan)
    return za, zb, d


def _bbox_of(d, pad=40):
    m = np.isfinite(d)
    if not m.any():
        return None
    r, c = np.where(m)
    return (max(r.min()-pad, 0), min(r.max()+pad, d.shape[0]),
            max(c.min()-pad, 0), min(c.max()+pad, d.shape[1]))


def pick(manifest, kind, siting=None, seed=0):
    c = [e for e in manifest if kind in kinds_of(e)
         and (siting is None or e.get("siting") == siting)]
    return random.Random(seed).choice(c) if c else None


# ---------------------------------------------------------------- panels
def panels(ens, manifest, base, out, seed=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = sorted({k for e in manifest for k in kinds_of(e)})
    fig, ax = plt.subplots(len(kinds), 3, figsize=(13, 3.5*len(kinds)), squeeze=False)
    for i, k in enumerate(kinds):
        e = pick(manifest, k, seed=seed)
        field = PRIMARY.get(k, "Manning")
        za, zb, d = _diff(e["run_dir"], base, field)
        if d is None or not np.isfinite(d).any():
            for j in range(3):
                ax[i][j].text(.5, .5, f"{k}\nno {field} edit found", ha="center",
                              va="center", transform=ax[i][j].transAxes,
                              color=PALETTE["intervention"])
                ax[i][j].axis("off")
            continue
        bb = _bbox_of(d)
        sl = (slice(bb[0], bb[1]), slice(bb[2], bb[3]))
        for j, (arr, ttl, cm) in enumerate([
                (za[sl], f"{k}: baseline {field}", "terrain" if field == "SUB_DEM" else "YlGnBu"),
                (zb[sl], "edited", "terrain" if field == "SUB_DEM" else "YlGnBu"),
                (d[sl], "difference", "RdBu_r")]):
            v = np.nanpercentile(np.abs(arr), 98) if j == 2 else None
            im = ax[i][j].imshow(arr, cmap=cm, **({"vmin": -v, "vmax": v} if j == 2 else {}))
            ax[i][j].set_title(ttl, fontsize=9.5, color=PALETTE["text"])
            ax[i][j].set_xticks([]); ax[i][j].set_yticks([])
            fig.colorbar(im, ax=ax[i][j], fraction=0.045, pad=0.02)
        n = int(np.isfinite(d).sum())
        ax[i][2].set_xlabel(f"{e['name']} ({e.get('siting')}), {n:,} cells, "
                            f"max |delta| {np.nanmax(np.abs(d)):.2f}", fontsize=8,
                            color=PALETTE["muted"])
    fig.suptitle("What each intervention changes", fontsize=14, y=0.995,
                 color=PALETTE["text"])
    fig.subplots_adjust(hspace=0.3, top=0.97)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


# ---------------------------------------------------------------- coverage
def coverage(ens, manifest, base, out, per_kind=25, seed=0):
    """Every sampled member's footprint per kind, random against targeted.

    The check the mixed-siting design rests on: random placement should spread across the kind's
    zone, targeted should concentrate on the high-scoring cells. If they look the same, the
    hold-out that trains on random and tests on targeted is not testing anything.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = sorted({k for e in manifest for k in kinds_of(e)})
    fig, ax = plt.subplots(2, len(kinds), figsize=(3.1*len(kinds), 7), squeeze=False)
    rng = random.Random(seed)
    for j, k in enumerate(kinds):
        field = PRIMARY.get(k, "Manning")
        for i, mode in enumerate(["random", "targeted"]):
            cand = [e for e in manifest if k in kinds_of(e) and e.get("siting") == mode]
            rng.shuffle(cand)
            acc = None
            for e in cand[:per_kind]:
                _, _, d = _diff(e["run_dir"], base, field)
                if d is None:
                    continue
                m = np.isfinite(d).astype("float32")
                acc = m if acc is None else acc + m
            a = ax[i][j]
            if acc is None or acc.max() == 0:
                a.text(.5, .5, "none", ha="center", va="center", transform=a.transAxes,
                       color=PALETTE["muted"])
            else:
                a.imshow(np.where(acc > 0, acc, np.nan), cmap="magma_r")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(k, fontsize=9, color=PALETTE["text"])
            if j == 0:
                a.set_ylabel(mode, fontsize=10, color=PALETTE["text"])
            a.set_xlabel(f"n={min(len(cand), per_kind)}", fontsize=7.5,
                         color=PALETTE["muted"])
    fig.suptitle(f"Placement coverage, up to {per_kind} members per kind and mode",
                 fontsize=13, y=0.99, color=PALETTE["text"])
    fig.text(0.5, 0.005, "Darker means more members edited that cell. Random should spread "
             "across the zone; targeted should concentrate.", ha="center", fontsize=8.5,
             color=PALETTE["muted"])
    fig.subplots_adjust(hspace=0.15, wspace=0.08, top=0.93)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


# ---------------------------------------------------------------- cross section
def section(ens, manifest, base, out, seed=0):
    """A transect across an elevation edit: is the crest a line, and at what height."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k in ELEV_KINDS if any(k in kinds_of(e) for e in manifest)]
    fig, ax = plt.subplots(len(ks), 1, figsize=(11, 2.9*len(ks)), squeeze=False)
    for i, k in enumerate(ks):
        e = pick(manifest, k, seed=seed)
        za, zb, d = _diff(e["run_dir"], base, "SUB_DEM")
        a = ax[i][0]
        if d is None or not np.isfinite(d).any():
            a.text(.5, .5, f"{k}: no DEM edit", ha="center", va="center",
                   transform=a.transAxes, color=PALETTE["intervention"]); a.axis("off"); continue
        r, c = np.where(np.isfinite(d))
        row = int(np.median(r))                       # a row crossing the structure
        lo, hi = max(c.min()-60, 0), min(c.max()+60, d.shape[1])
        x = np.arange(lo, hi) * 4.0
        a.plot(x, za[row, lo:hi], lw=1.4, color=PALETTE["muted"], label="baseline bed")
        a.plot(x, zb[row, lo:hi], lw=1.6, color=PALETTE["flood"], label="edited")
        slr = float(e.get("forcing", {}).get("slr_m", 0.0))
        a.axhline(1.114 + slr, ls="--", lw=1.0, color=PALETTE["intervention"],
                  label=f"member waterline {1.114+slr:.2f} m")
        a.set_title(f"{k} -- {e['name']} ({e.get('siting')}), row {row}", fontsize=10,
                    color=PALETTE["text"])
        a.set_ylabel("elevation (m)", fontsize=9); a.grid(alpha=0.25)
        a.legend(fontsize=7.5, frameon=False, ncol=3)
    ax[-1][0].set_xlabel("distance along the transect (m)", fontsize=9)
    fig.suptitle("Cross sections through the elevation edits", fontsize=13, y=0.997,
                 color=PALETTE["text"])
    fig.text(0.5, -0.01, "Every crest is freeboard ABOVE the member's own waterline, so it "
             "should sit above the dashed line at every sea level.", ha="center", fontsize=8.5,
             color=PALETTE["muted"])
    fig.subplots_adjust(hspace=0.45, top=0.95)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


# ---------------------------------------------------------------- knobs
def knobs(ens, manifest, out):
    """Sampled knob values against the registry ranges."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..interventions.generate import INTERVENTIONS
    vals = defaultdict(lambda: defaultdict(list))
    for e in manifest:
        for iv in (e.get("interventions") or []):
            for kk, vv in iv.items():
                if isinstance(vv, (int, float)) and kk in INTERVENTIONS.get(iv["kind"], {}):
                    vals[iv["kind"]][kk].append(vv)
    pairs = [(k, p) for k in sorted(vals) for p in sorted(vals[k])]
    n = len(pairs); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4*ncol, 2.5*nrow), squeeze=False)
    for i, (k, p) in enumerate(pairs):
        a = ax[i//ncol][i % ncol]
        v = np.array(vals[k][p])
        a.hist(v, bins=20, color=PALETTE["flood"], alpha=0.85)
        lo, hi = INTERVENTIONS[k][p]
        a.axvline(lo, ls="--", lw=1.1, color=PALETTE["intervention"])
        a.axvline(hi, ls="--", lw=1.1, color=PALETTE["intervention"])
        a.set_title(f"{k}\n{p}  n={v.size}", fontsize=8, color=PALETTE["text"])
        a.tick_params(labelsize=7)
        out_of = int(((v < lo - 1e-9) | (v > hi + 1e-9)).sum())
        if out_of:
            a.set_xlabel(f"{out_of} OUTSIDE range", fontsize=7.5,
                         color=PALETTE["intervention"])
    for j in range(n, nrow*ncol):
        ax[j//ncol][j % ncol].axis("off")
    fig.suptitle("Sampled knobs against the registry ranges (dashed)", fontsize=13, y=0.999,
                 color=PALETTE["text"])
    fig.subplots_adjust(hspace=0.75, wspace=0.25, top=0.94)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["noop", "panels", "coverage", "section", "knobs"])
    ap.add_argument("--ens", required=True, help="ensemble directory with manifest.json")
    ap.add_argument("--base", default=None, help="baseline run dir, for the grid comparisons")
    ap.add_argument("--per-kind", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ens, m = load(a.ens)
    default = f"reports/figures/qc_{a.cmd}.png"
    if a.cmd == "noop":
        check_noop(ens, m)
    elif a.cmd == "knobs":
        knobs(ens, m, a.out or default)
    else:
        if not a.base:
            raise SystemExit(f"{a.cmd} needs --base, the baseline run directory")
        {"panels": panels, "coverage": lambda *x: coverage(*x, per_kind=a.per_kind, seed=a.seed),
         "section": section}[a.cmd](ens, m, a.base, a.out or default)


if __name__ == "__main__":
    main()
