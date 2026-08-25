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
PRIMARY_FIELD = {"floodwall": "dem", "road_raise": "dem",
                 "living_shoreline": "dem", "retreat": "dem",
                 "depave": "manning", "marsh_restoration": "manning",
                 "marsh_migration": "manning"}
SHORT_FIELD = {"dem": "elevation", "manning": "Manning n",
               "ksat": "infiltration rate", "awc": "storage"}


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


def _largest_component(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == counts.argmax()


def _normal_section(component, arrays, half_width=75):
    """Sample arrays normal to the principal axis of one edited component."""
    from scipy.ndimage import map_coordinates
    rc = np.column_stack(np.where(component))
    centre = rc.mean(0)
    if len(rc) > 2:
        _, _, vh = np.linalg.svd(rc - centre, full_matrices=False)
        normal = vh[1]                 # second PC is normal to a linear feature
    else:
        normal = np.array([1.0, 0.0])
    s = np.linspace(-half_width, half_width, 2 * half_width + 1)
    coords = centre[:, None] + normal[:, None] * s
    values = {k: map_coordinates(a, coords, order=1, mode="nearest")
              for k, a in arrays.items()}
    return s, coords, values


def _build_hub(records, dem, out, siting, cell_m):
    """Top-left domain locator with mechanism cards along its right and lower edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.patches import Rectangle

    ls = LightSource(azdeg=315, altdeg=38)
    bg = ls.shade(np.nan_to_num(dem, nan=np.nanmedian(dem)), cmap=plt.cm.Greys,
                  vert_exag=.35, blend_mode="soft")
    fig = plt.figure(figsize=(18, 12))
    centre = fig.add_axes([.035, .415, .59, .49])
    # Three compact cards beside the domain and four deeper cards below it.
    # ``side`` cards place map and diagnostic side by side; ``bottom`` cards stack them.
    positions = [
        (.65, .748, .325, .157, "side"),
        (.65, .576, .325, .157, "side"),
        (.65, .404, .325, .157, "side"),
        (.035, .055, .218, .300, "bottom"),
        (.272, .055, .218, .300, "bottom"),
        (.509, .055, .218, .300, "bottom"),
        (.746, .055, .218, .300, "bottom"),
    ]
    centre.imshow(bg, interpolation="nearest")
    centre.set_xticks([]); centre.set_yticks([])
    centre.set_title("Full 4 m domain — representative edits within numbered zoom boxes",
                     fontsize=10, fontweight="bold")

    for i, ((kind, name, area, rd, delta, mask), pos) in enumerate(zip(records, positions)):
        primary = PRIMARY_FIELD[kind]
        pmask = np.abs(delta[primary]) > TOL
        component = _largest_component(pmask)
        sl = _crop(component, pad=35)
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        # Do not paint distributed masks over the whole domain.  Show the actual member
        # edits only inside the matching locator box; the zoom then resolves one feature.
        clipped = np.zeros_like(pmask, dtype=bool)
        clipped[sl] = pmask[sl]
        if primary == "dem" and kind in ("floodwall", "road_raise"):
            centre.contour(clipped.astype(float), levels=[.5], colors=[COLORS[kind]],
                           linewidths=1.15, alpha=.95)
        else:
            centre.contourf(clipped.astype(float), levels=[.5, 1.5],
                            colors=[COLORS[kind]], alpha=.46)
        centre.add_patch(Rectangle((c0, r0), c1-c0, r1-r0, fill=False,
                                   ec=COLORS[kind], lw=2.8, zorder=8))
        centre.text(c0+5, r0+5, str(i+1), ha="left", va="top", fontsize=7.5,
                    fontweight="bold", color="white", zorder=9,
                    bbox=dict(boxstyle="round,pad=.20", fc=COLORS[kind], ec="white", lw=.6))

        x, y, w, h, card_layout = pos
        if card_layout == "side":
            ax = fig.add_axes([x, y, .49*w, h])
            metric = fig.add_axes([x + .57*w, y + .12*h, .41*w, .69*h])
        else:
            ax = fig.add_axes([x, y + .43*h, w, .53*h])
            metric = fig.add_axes([x + .04*w, y + .04*h, .92*w, .29*h])
        ax.imshow(bg[sl], interpolation="nearest")
        local = component[sl]
        if primary == "dem" and kind in ("floodwall", "road_raise"):
            ax.contour(local.astype(float), levels=[.5], colors=["#111111"], linewidths=2.2)
            ax.contour(local.astype(float), levels=[.5], colors=[COLORS[kind]], linewidths=1.05)
        else:
            ax.contourf(local.astype(float), levels=[.5, 1.5], colors=[COLORS[kind]], alpha=.72)
            ax.contour(local.astype(float), levels=[.5], colors=["#111111"], linewidths=.65)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{i+1}  {LABELS[kind]}\n{area/1e4:.1f} ha; {SHORT_FIELD[primary]}",
                     fontsize=7.5, fontweight="bold", loc="left", pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5); spine.set_edgecolor(COLORS[kind])

        if "dem" in delta:
            s, coords, vals = _normal_section(component, {"z": dem, "dz": delta["dem"]})
            xx = s * cell_m
            metric.plot(xx, vals["z"], color="#777777", lw=1.1, label="before")
            metric.plot(xx, vals["z"] + vals["dz"], color=COLORS[kind], lw=1.05,
                        label="after")
            metric.set_xlabel("normal distance (m)", fontsize=6.1, labelpad=1)
            metric.set_ylabel("elevation (m)", fontsize=6.1, labelpad=1)
            metric.legend(fontsize=5.5, frameon=False, ncol=2, loc="best")
            ax.plot(coords[1]-c0, coords[0]-r0, color="white", lw=1.0, ls="--")
        else:
            vals = delta[primary][pmask]
            metric.hist(vals, bins=20, color=COLORS[kind], alpha=.82,
                        edgecolor="white", lw=.25)
            med, lo, hi = np.median(vals), np.percentile(vals, 5), np.percentile(vals, 95)
            metric.axvline(med, color="#111111", lw=.9, ls="--")
            metric.text(.98, .94, f"median {med:+.3f}\n5–95% {lo:+.3f} to {hi:+.3f}",
                        transform=metric.transAxes, ha="right", va="top", fontsize=5.4)
            metric.set_xlabel(f"change in {SHORT_FIELD[primary]}", fontsize=6.1, labelpad=1)
            metric.set_ylabel("cells", fontsize=6.1, labelpad=1)
        metric.tick_params(labelsize=5.6, pad=1)
        metric.grid(alpha=.14)

    fig.suptitle(f"Representative {siting} intervention placement and field edits",
                 fontsize=15, fontweight="bold", y=.985)
    fig.text(.5, .958, "Numbered boxes match the zooms. Field edits are drawn only inside their "
             "boxes to preserve domain context.", ha="center", fontsize=8.5)
    fig.text(.5, .012, "One Int2050 member nearest each family's median footprint. Zooms show the largest "
             "contiguous primary-field feature; diagnostics show a DEM section or all edited-cell changes.",
             ha="center", fontsize=8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out} and {Path(out).with_suffix('.pdf')}")


def build(ensemble, base, effect_csv, out, *, slr="slrInt2050", siting="targeted",
          layout="cards"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

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
    if layout == "hub":
        return _build_hub(records, dem, out, siting, cell_m)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.2, 8.8), constrained_layout=False)
    outer = fig.add_gridspec(2, 4, hspace=.34, wspace=.23)
    ls = LightSource(azdeg=315, altdeg=38)
    bg = ls.shade(np.nan_to_num(dem, nan=np.nanmedian(dem)), cmap=plt.cm.Greys,
                  vert_exag=.35, blend_mode="soft")

    for i, (kind, name, area, rd, delta, mask) in enumerate(records):
        cell = outer[i//4, i % 4].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=.15)
        amap = fig.add_subplot(cell[0])
        apro = fig.add_subplot(cell[1])
        primary = PRIMARY_FIELD[kind]
        pmask = np.abs(delta[primary]) > TOL
        component = _largest_component(pmask)
        sl = _crop(component, pad=35)
        amap.imshow(bg[sl], interpolation="nearest")
        local = component[sl]
        if primary == "dem" and kind in ("floodwall", "road_raise"):
            amap.contour(local.astype(float), levels=[.5], colors=["#111111"], linewidths=2.2)
            amap.contour(local.astype(float), levels=[.5], colors=[COLORS[kind]], linewidths=1.05)
        else:
            amap.contourf(local.astype(float), levels=[.5, 1.5],
                          colors=[COLORS[kind]], alpha=.68)
            amap.contour(local.astype(float), levels=[.5], colors=["#111111"], linewidths=.65)
        amap.set_xticks([]); amap.set_yticks([])
        fields = ", ".join(SHORT_FIELD[k] for k in delta)
        amap.set_title(f"{'ABCDEFG'[i]}  {LABELS[kind]}\n{area/1e4:.1f} ha member; "
                       f"local {SHORT_FIELD[primary]} feature",
                       loc="left", fontsize=8.2, fontweight="bold", color="#222222")

        if "dem" in delta:
            s, coords, vals = _normal_section(component, {"z": dem, "dz": delta["dem"]})
            x = s * cell_m
            z0, dz = vals["z"], vals["dz"]
            apro.plot(x, z0, color="#777777", lw=1.4, label="before")
            apro.plot(x, z0 + dz, color=COLORS[kind], lw=1.2, label="after")
            apro.set_ylabel("elevation (m)", fontsize=7)
            # Show the exact section on the zoom.
            r0, c0 = sl[0].start, sl[1].start
            apro.set_xlabel("distance normal to feature (m)", fontsize=7)
            amap.plot(coords[1]-c0, coords[0]-r0, color="white", lw=1.1, ls="--")
        else:
            # A distribution is more honest than a long section through disconnected masks.
            vals = delta[primary][pmask]
            apro.hist(vals, bins=24, color=COLORS[kind], alpha=.82, edgecolor="white", lw=.3)
            med, lo, hi = np.median(vals), np.percentile(vals, 5), np.percentile(vals, 95)
            apro.axvline(med, color="#111111", lw=1.0, ls="--")
            apro.text(.98, .92, f"median {med:+.3f}\n5–95% {lo:+.3f} to {hi:+.3f}",
                      transform=apro.transAxes, ha="right", va="top", fontsize=6.4)
            apro.set_xlabel(f"change in {SHORT_FIELD[primary]}", fontsize=7)
            apro.set_ylabel("edited cells", fontsize=7)
        apro.tick_params(labelsize=6.5)
        if "dem" in delta:
            apro.legend(fontsize=6, frameon=False, loc="best")
        apro.grid(alpha=.16)
        amap.text(.02, .02, "edits: " + fields, transform=amap.transAxes, fontsize=6.2,
                  bbox=dict(fc="white", ec="none", alpha=.82, pad=1.6))

    # Blank eighth card becomes the reading key and prevents another dense overview map.
    if len(records) < 8:
        key = fig.add_subplot(outer[1, 3]); key.axis("off")
        key.text(0, 1, "How to read this figure", va="top", fontweight="bold", fontsize=9)
        key.text(0, .82, "Map: largest contiguous feature in the\n"
                         "member's primary edited field\n"
                         "Dashed line: DEM section normal to feature\n"
                         "Histogram: actual cellwise parameter changes\n\n"
                         "Representative case: targeted Int2050\nnearest the median footprint",
                 va="top", fontsize=7.5, linespacing=1.35)
    fig.suptitle("How each intervention enters the hydraulic model",
                 fontsize=14, fontweight="bold", y=.985)
    fig.text(.5, .012, "Real targeted Int2050 ensemble members nearest each family's median footprint. "
             "Zooms show one representative contiguous feature; reported hectares refer to the full member.",
             ha="center", fontsize=8.3)
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
    ap.add_argument("--layout", choices=["cards", "hub"], default="cards")
    ap.add_argument("--out", default="reports/figures/intervention_anatomy.png")
    a = ap.parse_args()
    build(a.ensemble, a.base, a.effect_csv, a.out, slr=a.slr, siting=a.siting,
          layout=a.layout)


if __name__ == "__main__":
    main()
