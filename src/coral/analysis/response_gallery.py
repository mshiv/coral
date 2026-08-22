"""Flood response to each intervention: baseline depth, intervention depth, difference.

The same seven-row grid as ensemble_qc panels, but on the OUTPUT rather than the input.
qc_panels answers "what did the intervention change about the model"; this answers "what
did it change about the flood", which is the question a planner asks and the only one that
carries a sign a community would recognise.

Blue is depth reduced, red is depth increased. Red matters as much as blue: an intervention
that lowers water in one place can raise it in another, and a wall that protects its own
footprint while pushing water onto a neighbour is a real modelled outcome, not an artefact.
The reported `max_increase_m` and `area_worsened_m2` are the numbers that expose it.

The footprint is drawn on the CHANGE panel, not on the depth panels. Where the edit sits is
only interpretable against where the water moved: a wall outline over a depth map says little,
while the same outline against blue upstream and red downstream shows the mechanism. Linear
measures are drawn as an outline so the change beneath stays visible; area measures are drawn
as a hatched boundary for the same reason. The footprint is recovered by differencing the
member grids against the baseline, so it is what the model actually received rather than what
the manifest requested.

Terrain sits underneath as a hillshade with open water flooded in, so a reader can tell a
channel from a road without a separate reference figure.

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
PALETTE = {"cut": "#2c7fb8", "add": "#a63f22",
           "water": "#9CC3D5", "edit": "#C0392B"}


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


# Which grid carries each kind's footprint, and how to draw it. Linear structures read as
# outlines; area treatments read as a boundary with light fill.
FOOTPRINT = {
    "floodwall":         ("SUB_DEM", "line"),
    "road_raise":        ("SUB_DEM", "line"),
    "retreat":           ("SUB_DEM", "area"),
    "living_shoreline":  ("SUB_DEM", "area"),
    "marsh_restoration": ("Manning", "area"),
    "marsh_migration":   ("Manning", "area"),
    "depave":            ("Manning", "area"),
}


def footprint(run_dir, base_dir, field, tol=1e-6):
    """Cells the member actually changed, from the grids rather than the manifest."""
    a = sorted(Path(run_dir).glob(f"{field}_*.asc"))
    b = sorted(Path(base_dir).glob(f"{field}_*.asc"))
    if not a or not b:
        return None
    m = read_asc(a[0])[0]
    n = read_asc(b[0])[0]
    if m.shape != n.shape:
        return None
    d = np.abs(np.nan_to_num(m, nan=0.0) - np.nan_to_num(n, nan=0.0))
    return d > tol


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
    ap.add_argument("--base", default=None,
                    help="baseline input dir (pp4_base). Needed to recover footprints.")
    ap.add_argument("--no-basemap", action="store_true",
                    help="drop the terrain hillshade")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from coral.viz.pinpoint_style import make_flood_cmap
    FLOOD = make_flood_cmap()

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
        fp = None
        if a.base and k in FOOTPRINT:
            fp = footprint(mem["run_dir"], a.base, FOOTPRINT[k][0])
        stats[k] = {
            "member": mem["name"], "baseline": base["name"],
            "footprint_cells": int(fp.sum()) if fp is not None else None,
            "mean_change_m": float(np.nanmean(d[wet])) if wet.any() else 0.0,
            "max_reduction_m": float(-np.nanmin(d[wet])) if wet.any() else 0.0,
            "max_increase_m": float(np.nanmax(d[wet])) if wet.any() else 0.0,
            "area_improved_m2": float((wet & (d < -0.01)).sum()) * cell,
            "area_worsened_m2": float((wet & (d > 0.01)).sum()) * cell,
            "wet_area_change_m2": float((land & (dm > a.threshold)).sum()
                                        - (land & (db > a.threshold)).sum()) * cell,
        }
        rows.append((k, np.where(land, db, np.nan), np.where(land, dm, np.nan), d,
                     mem["name"], fp))

    cmap_d = LinearSegmentedColormap.from_list(
        "cut_add", [PALETTE["cut"], "#ffffff", PALETTE["add"]])
    shade = None
    if not a.no_basemap:
        from coral.viz.pinpoint_style import hillshade
        shade = hillshade(np.nan_to_num(dem, nan=float(np.nanmin(dem[np.isfinite(dem)]))))
    sea = np.isfinite(dem) & (dem <= a.waterline)

    def ground(ax):
        """Terrain under every panel, so a channel is distinguishable from a road."""
        if shade is not None:
            ax.imshow(shade, cmap="Greys_r", vmin=0, vmax=1.4, interpolation="none", zorder=0)
        ax.imshow(np.where(sea, 1.0, np.nan), cmap=ListedColormap([PALETTE["water"]]),
                  vmin=0, vmax=1, interpolation="none", zorder=1, alpha=0.9)

    def draw_footprint(ax, fp, style):
        """Outline, never a solid fill: the change beneath has to stay readable."""
        if fp is None or not fp.any():
            return
        ax.contour(fp.astype(float), levels=[0.5], colors=[PALETTE["edit"]],
                   linewidths=1.1 if style == "line" else 0.8, zorder=5)
        if style == "area":
            ax.imshow(np.where(fp, 1.0, np.nan),
                      cmap=ListedColormap([PALETTE["edit"]]), vmin=0, vmax=1,
                      alpha=0.13, interpolation="none", zorder=4)

    fig, ax = plt.subplots(len(rows), 3, figsize=(13.5, 3.6 * len(rows)), squeeze=False)
    for i, (k, db, dm, d, nm, fp) in enumerate(rows):
        finite = np.concatenate([db[np.isfinite(db)], dm[np.isfinite(dm)]])
        vmax = np.nanpercentile(finite, 99) if finite.size else 0.1
        for j, (arr, ttl) in enumerate(((db, "no intervention"), (dm, "with intervention"))):
            ground(ax[i][j])
            im = ax[i][j].imshow(np.where(arr > a.threshold, arr, np.nan), cmap=FLOOD,
                                 vmin=0, vmax=max(vmax, 0.1), interpolation="none", zorder=2)
            ax[i][j].set_title(f"{k}: {ttl}" if j == 0 else ttl, fontsize=10)
            fig.colorbar(im, ax=ax[i][j], fraction=0.046, label="depth (m)")
        ground(ax[i][2])
        v = np.nanpercentile(np.abs(d), 99.5) or 0.05
        im = ax[i][2].imshow(np.where(np.abs(d) > 0.005, d, np.nan), cmap=cmap_d,
                             vmin=-v, vmax=v, interpolation="none", zorder=2)
        draw_footprint(ax[i][2], fp, FOOTPRINT.get(k, (None, "area"))[1])
        ax[i][2].set_title("change, with the edit outlined", fontsize=10)
        fig.colorbar(im, ax=ax[i][2], fraction=0.046, label="m")
        s = stats[k]
        fpn = f", {s['footprint_cells']:,} cells edited" if s.get("footprint_cells") else ""
        ax[i][2].set_xlabel(
            f"{nm}\n-{s['max_reduction_m']:.2f} m deepest cut, "
            f"+{s['max_increase_m']:.2f} m worst increase, "
            f"{s['area_worsened_m2']/1e4:.1f} ha worsened{fpn}", fontsize=8, color="0.35")
        for j in range(3):
            ax[i][j].set_xticks([]); ax[i][j].set_yticks([])

    fig.suptitle("Flood response to each intervention, against the baseline at the same sea level",
                 fontsize=13)
    fig.text(0.5, 0.005, "Blue: depth reduced.  Red: depth increased -- water moved, not removed.  Dark red outline: the cells the member actually edited.",
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
