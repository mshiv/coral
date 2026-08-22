"""Flood response to each intervention: baseline depth, intervention depth, difference.

The same seven-row grid as ensemble_qc panels, but on the OUTPUT rather than the input.
qc_panels answers "what did the intervention change about the model"; this answers "what
did it change about the flood", which is the question a planner asks and the only one that
carries a sign a community would recognise.

Blue is depth reduced, red is depth increased. Red matters as much as blue: an intervention
that lowers water in one place can raise it in another, and a wall that protects its own
footprint while pushing water onto a neighbour is a real modelled outcome, not an artefact.
The reported `max_increase_m` and `area_worsened_m2` are the numbers that expose it.

Members are matched to the no-intervention baseline AT THE SAME SEA LEVEL, so the sea-level
signal cancels and what remains is the intervention.

Usage:
  python -m coral.analysis.response_gallery --ens <ensemble dir> --dem <DEM .asc> \
      --waterline 1.114 --out reports/figures/response_gallery.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NODATA = -9999.0
PALETTE = {"cut": "#2c7fb8", "add": "#a63f22"}


def read_asc(path):
    hdr, n = {}, 0
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 2 and not p[0][:1].isdigit() and p[0][:1] != "-":
                hdr[p[0].lower()] = float(p[1]); n += 1
            else:
                break
    return np.where((a := np.loadtxt(path, skiprows=n)) > NODATA + 1.0, a, np.nan), hdr


def find_max(run_dir):
    hits = sorted(Path(run_dir).glob("results_*/*.max"))
    return hits[0] if hits else None


def kinds_of(entry):
    return sorted({i["kind"] for i in (entry.get("interventions") or [])})


def slr_of(name):
    return name.split("_")[0]


def pick(manifest, slr=None, exclude=()):
    """One representative member per kind, plus the baseline at its sea level.

    `slr` pins the sea level so every panel is comparable and so a level whose members
    failed verification cannot be selected. Without it the largest area_frac wins, which
    biases selection toward the highest offset -- and if that level did not conserve mass,
    the whole figure is drawn from unusable runs. `exclude` drops named members.

    This is a figure, not a statistic. It shows what an operator does to the flood; the
    ensemble-wide distribution is the job of adaptation_effectiveness.
    """
    ex = set(exclude)
    base = {slr_of(e["name"]): e for e in manifest if not kinds_of(e)}
    best = {}
    for e in manifest:
        ks = kinds_of(e)
        if len(ks) != 1 or e["name"] in ex:
            continue
        lvl = slr_of(e["name"])
        if lvl not in base or (slr and lvl != slr):
            continue
        af = (e["interventions"][0] or {}).get("area_frac", 0.0) or 0.0
        if k := ks[0]:
            if k not in best or af > best[k][0]:
                best[k] = (af, e)
    return {k: (v[1], base[slr_of(v[1]["name"])]) for k, v in best.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--waterline", type=float, required=True)
    ap.add_argument("--threshold", type=float, default=0.10, help="wet depth, m")
    ap.add_argument("--cell-m", type=float, default=4.0)
    ap.add_argument("--out", default="reports/figures/response_gallery.png")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--slr", default=None,
                    help="pin one sea level, e.g. slr0.0 or slrInt2050. Without it the\nlargest footprint wins, which biases selection toward the highest offset.")
    ap.add_argument("--exclude", nargs="*", default=(),
                    help="member names to skip, e.g. any that failed mass verification")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    manifest = json.load(open(Path(a.ens) / "manifest.json"))
    chosen = pick(manifest, a.slr, a.exclude)
    if not chosen:
        raise SystemExit(f"no member/baseline pairs found for slr={a.slr!r}. "
                         "Check the level name against the manifest.")

    dem, _ = read_asc(a.dem)
    land = np.isfinite(dem) & (dem > a.waterline)

    rows, stats = [], {}
    for k in sorted(chosen):
        mem, base = chosen[k]
        mmax, bmax = find_max(mem["run_dir"]), find_max(base["run_dir"])
        if mmax is None or bmax is None:
            continue
        dm = np.nan_to_num(read_asc(mmax)[0], nan=0.0)
        db = np.nan_to_num(read_asc(bmax)[0], nan=0.0)
        if dm.shape != land.shape:
            raise SystemExit(f"{k}: .max shape {dm.shape} does not match the DEM {land.shape}")
        d = np.where(land, dm - db, np.nan)
        wet = land & ((dm > a.threshold) | (db > a.threshold))
        cell = a.cell_m ** 2
        stats[k] = {
            "member": mem["name"], "baseline": base["name"],
            "mean_change_m": float(np.nanmean(d[wet])) if wet.any() else 0.0,
            "max_reduction_m": float(-np.nanmin(d[wet])) if wet.any() else 0.0,
            "max_increase_m": float(np.nanmax(d[wet])) if wet.any() else 0.0,
            "area_improved_m2": float((wet & (d < -0.01)).sum()) * cell,
            "area_worsened_m2": float((wet & (d > 0.01)).sum()) * cell,
            "wet_area_change_m2": float((land & (dm > a.threshold)).sum()
                                        - (land & (db > a.threshold)).sum()) * cell,
        }
        rows.append((k, np.where(land, db, np.nan), np.where(land, dm, np.nan), d, mem["name"]))

    cmap_d = LinearSegmentedColormap.from_list(
        "cut_add", [PALETTE["cut"], "#ffffff", PALETTE["add"]])
    fig, ax = plt.subplots(len(rows), 3, figsize=(13, 3.5 * len(rows)), squeeze=False)
    for i, (k, db, dm, d, nm) in enumerate(rows):
        vmax = np.nanpercentile(np.concatenate([db[np.isfinite(db)], dm[np.isfinite(dm)]]), 99)
        for j, (arr, ttl) in enumerate(((db, "no intervention"), (dm, "with intervention"))):
            im = ax[i][j].imshow(arr, cmap="Blues", vmin=0, vmax=max(vmax, 0.1),
                                 interpolation="none")
            ax[i][j].set_title(f"{k}: {ttl}" if j == 0 else ttl, fontsize=10)
            fig.colorbar(im, ax=ax[i][j], fraction=0.046, label="depth (m)")
        v = np.nanpercentile(np.abs(d), 99.5) or 0.05
        im = ax[i][2].imshow(d, cmap=cmap_d, vmin=-v, vmax=v, interpolation="none")
        ax[i][2].set_title("change", fontsize=10)
        fig.colorbar(im, ax=ax[i][2], fraction=0.046, label="m")
        s = stats[k]
        ax[i][2].set_xlabel(
            f"{nm}\n-{s['max_reduction_m']:.2f} m deepest cut, "
            f"+{s['max_increase_m']:.2f} m worst increase, "
            f"{s['area_worsened_m2']/1e4:.1f} ha worsened", fontsize=8, color="0.35")
        for j in range(3):
            ax[i][j].set_xticks([]); ax[i][j].set_yticks([])

    fig.suptitle("Flood response to each intervention, against the baseline at the same sea level",
                 fontsize=13)
    fig.text(0.5, 0.005, "Blue: depth reduced.  Red: depth increased -- water moved, not removed.",
             ha="center", fontsize=9, color="0.4")
    fig.tight_layout(rect=[0, 0.012, 1, 0.985])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")

    print(f"\n{'kind':20s} {'mean dm':>9} {'max cut':>9} {'max add':>9} {'ha worse':>9}")
    for k in sorted(stats):
        s = stats[k]
        print(f"{k:20s} {s['mean_change_m']:9.3f} {-s['max_reduction_m']:9.3f} "
              f"{s['max_increase_m']:9.3f} {s['area_worsened_m2']/1e4:9.1f}")
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(stats, indent=2))
        print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
