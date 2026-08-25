"""Publication figure showing how representative intervention members edit model fields.

The figure uses real ensemble members.  A shared overview locates every footprint; seven
cards then show a local zoom and either a terrain transect (DEM edits) or a parameter profile
(roughness/infiltration edits).  This is the compact, reader-facing counterpart to the full
baseline/edited/difference QC grid, which remains an audit artifact.

Example
-------
python -m coral.analysis.intervention_anatomy \
  --ensemble "$SCR/runs/pp4_e01" --base "$SCR/runs/pp4_base" \
  --effect-csv reports/adapt/effect_metrics_pp4_e01.csv \
  --slr slrInt2050 --siting targeted \
  --out reports/figures/intervention_anatomy.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .intervention_gallery import GRIDS, TOL, _hdr, _read, member_deltas

KINDS = ["floodwall", "road_raise", "living_shoreline", "depave",
         "marsh_restoration", "marsh_migration", "retreat"]
LABELS = {"floodwall": "Floodwall", "road_raise": "Raised road",
          "living_shoreline": "Living shoreline", "depave": "De-paving",
          "marsh_restoration": "Marsh restoration",
          "marsh_migration": "Marsh migration", "retreat": "Managed retreat"}
COLORS = {"floodwall": "#D55E00", "road_raise": "#E69F00",
          "living_shoreline": "#009E73", "depave": "#56B4E9",
          "marsh_restoration": "#0072B2", "marsh_migration": "#CC79A7",
          "retreat": "#6B6B6B"}
FIELD_LABEL = {"dem": r"elevation $\Delta z$ (m)",
               "manning": r"roughness $\Delta n$",
               "ksat": r"$\Delta K_{sat}$", "awc": r"$\Delta$ storage"}


def _base_files(base):
    out = {}
    for key, pat in GRIDS.items():
        p = next(iter(sorted(Path(base).glob(pat))), None)
        if p is not None:
            out[key] = p
    if "dem" not in out:
        raise SystemExit(f"no SUB_DEM*.asc in {base}")
    return out


def _choose_members(effect_csv, manifest, slr, siting):
    """Choose the targeted member nearest the median footprint for each kind."""
    by_name = {r["name"]: r for r in manifest}
    grouped = {k: [] for k in KINDS}
    with open(effect_csv, newline="") as f:
        for r in csv.DictReader(f):
            k = r.get("kind")
            if k not in grouped or r.get("slr") != slr or r.get("siting") != siting:
                continue
            if r.get("name") not in by_name:
                continue
            grouped[k].append((float(r.get("footprint_m2") or 0), r["name"]))
    selected = []
    for k in KINDS:
        rows = sorted(grouped[k])
        if not rows:
            print(f"warning: no {k} member for {slr}, {siting}")
            continue
        median = np.median([x[0] for x in rows])
        area, name = min(rows, key=lambda x: abs(x[0] - median))
        e = by_name[name]
        rd = Path(e.get("run_dir", ""))
        selected.append((k, name, rd, area))
    return selected


def _resolve_run(ensemble, rd, name):
    if rd.is_dir():
        return rd
    p = Path(ensemble) / name
    if p.is_dir():
        return p
    raise FileNotFoundError(f"run directory missing for {name}: {rd} or {p}")


def _profile(mask):
    """Return the row or column crossing the most edited cells."""
    nr, nc = mask.sum(1), mask.sum(0)
    return ("row", int(np.argmax(nr))) if nr.max() >= nc.max() else ("col", int(np.argmax(nc)))


def _slice_profile(a, orient, idx):
    return a[idx, :] if orient == "row" else a[:, idx]


def _crop(mask, pad=30):
    r, c = np.where(mask)
    return (slice(max(0, r.min()-pad), min(mask.shape[0], r.max()+pad+1)),
            slice(max(0, c.min()-pad), min(mask.shape[1], c.max()+pad+1)))


def build(ensemble, base, effect_csv, out, *, slr="slrInt2050", siting="targeted"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.lines import Line2D

    manifest = json.load(open(Path(ensemble) / "manifest.json"))
    picks = _choose_members(effect_csv, manifest, slr, siting)
    files = _base_files(base)
    arrays = {k: _read(p) for k, p in files.items()}
    dem, hdr = arrays["dem"], _hdr(files["dem"])
    cell_m = float(hdr["cellsize"]) * 111_000.0 if hdr["cellsize"] < 1 else float(hdr["cellsize"])

    records = []
    for kind, name, rd, area in picks:
        rd = _resolve_run(ensemble, rd, name)
        delta = member_deltas(rd, base)
        if not delta:
            print(f"warning: {name} has no changed fields")
            continue
        mask = np.zeros_like(dem, dtype=bool)
        for a in delta.values():
            mask |= np.abs(a) > TOL
        records.append((kind, name, area, rd, delta, mask))
    if not records:
        raise SystemExit("no representative members could be loaded")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.2, 12.8), constrained_layout=False)
    outer = fig.add_gridspec(3, 4, height_ratios=[1.32, 1, 1], hspace=.30, wspace=.20)
    ax0 = fig.add_subplot(outer[0, :])
    ls = LightSource(azdeg=315, altdeg=38)
    bg = ls.shade(np.nan_to_num(dem, nan=np.nanmedian(dem)), cmap=plt.cm.Greys,
                  vert_exag=.35, blend_mode="soft")
    ax0.imshow(bg, interpolation="nearest")
    for kind, name, area, rd, delta, mask in records:
        ax0.contour(mask.astype(float), levels=[.5], colors=[COLORS[kind]], linewidths=1.25)
    handles = [Line2D([0], [0], color=COLORS[k], lw=2, label=LABELS[k])
               for k, *_ in records]
    ax0.legend(handles=handles, ncol=4, loc="lower center", frameon=True,
               bbox_to_anchor=(.5, -.01), fontsize=8)
    ax0.set_title("A  Representative intervention footprints in the common 4 m domain",
                  loc="left", fontweight="bold", fontsize=11)
    ax0.set_xticks([]); ax0.set_yticks([])

    for i, (kind, name, area, rd, delta, mask) in enumerate(records):
        cell = outer[1 + i//4, i % 4].subgridspec(2, 1, height_ratios=[1.38, 1], hspace=.12)
        amap = fig.add_subplot(cell[0])
        apro = fig.add_subplot(cell[1])
        sl = _crop(mask)
        amap.imshow(bg[sl], interpolation="nearest")
        # Draw every changed field; precedence keeps thin DEM structures visible.
        for field in ("awc", "ksat", "manning", "dem"):
            if field not in delta:
                continue
            m = np.abs(delta[field][sl]) > TOL
            amap.contourf(m.astype(float), levels=[.5, 1.5], colors=[COLORS[kind]], alpha=.62)
        amap.contour(mask[sl].astype(float), levels=[.5], colors=["#111111"], linewidths=.55)
        amap.set_xticks([]); amap.set_yticks([])
        fields = ", ".join(FIELD_LABEL[k].replace("$", "") for k in delta)
        amap.set_title(f"{'BCDEFGH'[i]}  {LABELS[kind]}\n{area/1e4:.1f} ha; edits {fields}",
                       loc="left", fontsize=8.4, fontweight="bold", color="#222222")

        orient, idx = _profile(mask)
        edited_line = _slice_profile(mask, orient, idx)
        q = np.flatnonzero(edited_line)
        lo, hi = max(0, q.min()-50), min(edited_line.size, q.max()+51)
        x = np.arange(lo, hi) * cell_m
        if "dem" in delta:
            z0 = _slice_profile(dem, orient, idx)[lo:hi]
            dz = _slice_profile(delta["dem"], orient, idx)[lo:hi]
            apro.plot(x, z0, color="#777777", lw=1.4, label="before")
            apro.plot(x, z0 + dz, color=COLORS[kind], lw=1.2, label="after")
            apro.set_ylabel("elevation (m)", fontsize=7)
        else:
            order = [k for k in ("manning", "ksat", "awc") if k in delta]
            for j, field in enumerate(order):
                y = _slice_profile(delta[field], orient, idx)[lo:hi]
                apro.plot(x, y, color=COLORS[kind], lw=1.2,
                          ls=("-", "--", ":")[j], label=FIELD_LABEL[field])
            apro.axhline(0, color="#888888", lw=.6)
            apro.set_ylabel("parameter change", fontsize=7)
        apro.set_xlabel("distance along section (m)", fontsize=7)
        apro.tick_params(labelsize=6.5)
        apro.legend(fontsize=6, frameon=False, loc="best")
        apro.grid(alpha=.16)

    # Blank eighth card becomes a compact reading key.
    if len(records) < 8:
        key = fig.add_subplot(outer[2, 3]); key.axis("off")
        key.text(0, 1, "How to read this figure", va="top", fontweight="bold", fontsize=9)
        key.text(0, .82, "Outline: union of all edited cells\n"
                         "Colour: intervention footprint\n"
                         "Profile: DEM before/after, or\nparameter change when terrain is fixed\n\n"
                         "Representative case: targeted Int2050\nnearest the median footprint",
                 va="top", fontsize=7.5, linespacing=1.35)
    fig.suptitle("Interventions are physical edits, not categorical labels",
                 fontsize=14, fontweight="bold", y=.985)
    fig.text(.5, .012, "Real ensemble members. The overview establishes placement; local cards expose "
             "the field and magnitude supplied to the hydraulic model.", ha="center", fontsize=8.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out} and {Path(out).with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--effect-csv", required=True)
    ap.add_argument("--slr", default="slrInt2050")
    ap.add_argument("--siting", default="targeted")
    ap.add_argument("--out", default="reports/figures/intervention_anatomy.png")
    a = ap.parse_args()
    build(a.ensemble, a.base, a.effect_csv, a.out, slr=a.slr, siting=a.siting)


if __name__ == "__main__":
    main()
