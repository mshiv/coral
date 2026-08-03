"""Adaptation effectiveness: each run against its matched no-adaptation baseline.

Every member is compared against the baseline member at the *same* sea level, and the reported quantity is a difference.

Three subcommands:

  pair       one adaptation .max against one baseline .max
  ensemble   walk an ensemble's manifest.json, match each member to its baseline, write a
             tidy CSV with one row per member
  figures    the headline plots from that CSV

Metrics:

  buildings   change in the number of buildings flooded at each depth threshold, plus how many
              are newly dry and how many are newly wet. This leads, per the sampling plan.
  depth       change in mean depth over flooded buildings
  domain      wet area and flood volume.

"Newly wet" matters as much as "newly dry" and is the reason the pair is tracked per building
rather than as a count difference. A net improvement of ten houses could be forty saved and
thirty newly flooded, which is a different intervention, and a different conversation with the
community, than forty saved and none lost.

Deps: numpy, matplotlib, geopandas + rasterio (only when footprints are supplied).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .building_exposure import THRESHOLDS, exposure, read_depth


def _cell_area_m2(dem_asc):
    """Cell area in m^2 from an EPSG:4326 grid, using the local metre-per-degree scaling at"""
    h = {}
    with open(dem_asc) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    lat = h["yllcorner"] + 0.5 * h["nrows"] * h["cellsize"]
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat))
    return h["cellsize"] ** 2 * m_per_deg_lat * m_per_deg_lon


def compare_pair(base_max, adapt_max, dem_asc, *, footprints=None, first_floor_m=0.0,
                 wet_thresh=0.05):
    """Metrics for one adaptation run against its matched baseline. Negative deltas are
    improvements: fewer buildings flooded, less water."""
    b, a = read_depth(base_max), read_depth(adapt_max)
    if b.shape != a.shape:
        raise SystemExit(f"grid mismatch: baseline {b.shape} vs adaptation {a.shape}")
    cell = _cell_area_m2(dem_asc)
    bw, aw = b > wet_thresh, a > wet_thresh
    out = {
        "wet_area_km2_base": float(bw.sum() * cell / 1e6),
        "wet_area_km2_adapt": float(aw.sum() * cell / 1e6),
        "d_wet_area_km2": float((aw.sum() - bw.sum()) * cell / 1e6),
        "volume_m3_base": float(b.sum() * cell),
        "volume_m3_adapt": float(a.sum() * cell),
        "d_volume_m3": float((a.sum() - b.sum()) * cell),
        "max_depth_increase_m": float(np.nanmax(a - b)),
        "max_depth_decrease_m": float(np.nanmin(a - b)),
    }
    if not footprints:
        return out

    # pass the grids already read above; without this each .max is parsed twice per pair
    br, bs = exposure(base_max, footprints, dem_asc, first_floor_m=first_floor_m, depth=b)
    ar, asum = exposure(adapt_max, footprints, dem_asc, first_floor_m=first_floor_m, depth=a)
    key = "depth_interior_m" if first_floor_m else "depth_ground_m"
    bd = np.array([r[key] for r in br]); ad = np.array([r[key] for r in ar])
    t0 = THRESHOLDS[0][1]
    out.update({
        "n_buildings": bs["n_buildings"],
        "newly_dry": int(((bd > t0) & (ad <= t0)).sum()),
        "newly_wet": int(((bd <= t0) & (ad > t0)).sum()),
        "mean_depth_flooded_m_base": bs["mean_depth_flooded_m"],
        "mean_depth_flooded_m_adapt": asum["mean_depth_flooded_m"],
        "d_mean_depth_flooded_m": asum["mean_depth_flooded_m"] - bs["mean_depth_flooded_m"],
    })
    for label, _ in THRESHOLDS:
        k = f"n_{label}"
        out[f"{k}_base"] = bs[k]
        out[f"{k}_adapt"] = asum[k]
        out[f"d_{k}"] = asum[k] - bs[k]
    return out


# ------------------------------------------------------------------ ensemble driver

def _slr_key(entry):
    """Group members by sea level. Prefer the scenario label, fall back to the numeric offset.
    Read from the manifest's forcing block rather than parsed from the member name, so this
    works for older ensembles whose names predate the slr<label>_<kind><n> convention."""
    f = entry.get("forcing", {})
    lab = f.get("slr_label")
    return lab if lab else f"{float(f.get('slr_m', 0.0)):g}"


def _kind(entry):
    """Intervention kind from the manifest, or `base` when there is none. Taken from the
    interventions list, for the same reason as _slr_key."""
    iv = entry.get("interventions") or []
    return iv[0]["kind"] if len(iv) == 1 else ("combo" if iv else "base")


def _max_path(entry, ensemble_dir):
    d = Path(entry["run_dir"])
    if not d.is_absolute():
        d = Path(ensemble_dir).parent.parent / d if not (Path(ensemble_dir) / d.name).exists() \
            else Path(ensemble_dir) / d.name
    root = entry.get("root", "res_matthew_sav")
    hits = sorted(d.glob(f"results_*/{root}.max")) or sorted(d.glob("results_*/*.max"))
    return hits[0] if hits else None


def run_ensemble(ensemble_dir, dem_asc, *, footprints=None, first_floor_m=0.0,
                 out_csv=None, quiet=False):
    """Compare every finished member against the baseline at its own sea level.

    Members whose `.max` is missing are skipped and counted, so this can run while the array
    is still going. A sea level with no finished baseline is skipped as a group, because
    without its baseline every delta at that level would be meaningless.
    """
    # Validate inputs before doing any work. An unset environment variable leaves an empty
    # path that expands to something like "/SUB_DEM_SAV.asc", and without this the failure
    # surfaces deep inside the first pair comparison rather than at the argument that caused it.
    for label, path in (("--dem", dem_asc), ("--footprints", footprints)):
        if path and not Path(path).exists():
            raise SystemExit(f"{label} does not exist: {path}\n"
                             "  (an unset $BASE or $SCR expands to nothing; export them first)")
    ens = Path(ensemble_dir)
    if not (ens / "manifest.json").exists():
        raise SystemExit(f"no manifest.json in {ens}")
    entries = json.load(open(ens / "manifest.json"))
    by_level = {}
    for e in entries:
        by_level.setdefault(_slr_key(e), []).append(e)

    rows, skipped, no_base = [], 0, []
    for level, members in sorted(by_level.items()):
        bases = [m for m in members if _kind(m) == "base"]
        bmax = _max_path(bases[0], ens) if bases else None
        if not bmax:
            no_base.append(level)
            skipped += len(members)
            continue
        for m in members:
            if _kind(m) == "base":
                continue
            amax = _max_path(m, ens)
            if not amax:
                skipped += 1; continue
            met = compare_pair(bmax, amax, dem_asc, footprints=footprints,
                               first_floor_m=first_floor_m)
            iv = (m.get("interventions") or [{}])[0]
            rows.append({"name": m["name"], "slr_level": level,
                         "slr_m": float(m.get("forcing", {}).get("slr_m", 0.0)),
                         "kind": _kind(m),
                         "area_frac": iv.get("area_frac"), "seed": iv.get("seed"),
                         **met})
            if not quiet:
                dd = met.get("d_n_damage_0.30m")
                print(f"  {m['name']:34s} {level:12s} "
                      + (f"buildings at 0.30 m {dd:+d}" if dd is not None
                         else f"volume {met['d_volume_m3']:+.3g} m3"))

    if no_base:
        print(f"NOTE: no finished baseline for sea level(s) {no_base}; those members were "
              "skipped because a delta without its own baseline is not interpretable.")
    print(f"{len(rows)} members compared, {skipped} skipped (unfinished or no baseline)")
    if out_csv and rows:
        import csv as _csv
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0])
        for r in rows:                                  # keep the header a superset
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"tidy results -> {out_csv}")
    return rows


# ------------------------------------------------------------------ figures

def _load_rows(csv_path):
    import csv as _csv
    out = []
    for r in _csv.DictReader(open(csv_path)):
        d = {}
        for k, v in r.items():
            if v is None or v == "":
                d[k] = None; continue
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def fig_reduction_bar(csv_path, out_png, metric="d_n_damage_0.30m",
                      title="Buildings flooded above 0.30 m, change from no adaptation"):
    """Effectiveness by adaptation kind. Bars are the mean across realisations, with the
    spread shown as a strip of individual realisations, because with 50 random placements per
    kind the spread is important."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in _load_rows(csv_path) if r.get(metric) is not None]
    if not rows:
        raise SystemExit(f"no rows with {metric} in {csv_path}. Was --footprints supplied?")
    kinds = sorted({r["kind"] for r in rows})
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(kinds)), 4.5))
    for i, k in enumerate(kinds):
        v = np.array([r[metric] for r in rows if r["kind"] == k])
        ax.bar(i, v.mean(), color="#3b7dd8" if v.mean() <= 0 else "#c0392b", alpha=.85)
        if len(v) > 1:
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).normal(0, .06, len(v)),
                       v, s=12, color="0.25", zorder=3, alpha=.6)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels(kinds, rotation=20, ha="right")
    ax.set_ylabel("change in buildings flooded")
    ax.set_title(title + "\nnegative is fewer buildings flooded", fontsize=10)
    fig.tight_layout(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160); plt.close(fig)
    print(f"figure -> {out_png}")


def fig_effectiveness_vs_slr(csv_path, out_png, metric="n_damage_0.30m_adapt"):
    """Buildings flooded against sea level, one line per adaptation plus the baseline. This is
    the decision-relevant view: it shows not just that an adaptation helps, but how many
    decades of sea-level rise it buys before exposure returns to today's level."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in _load_rows(csv_path) if r.get(metric) is not None]
    if not rows:
        raise SystemExit(f"no rows with {metric} in {csv_path}. Was --footprints supplied?")
    base_key = metric.replace("_adapt", "_base")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    levels = sorted({r["slr_m"] for r in rows})
    bl = [np.mean([r[base_key] for r in rows if r["slr_m"] == s]) for s in levels]
    ax.plot(levels, bl, "k--o", lw=2, label="no adaptation", zorder=5)
    for k in sorted({r["kind"] for r in rows}):
        ys = [np.mean([r[metric] for r in rows if r["kind"] == k and r["slr_m"] == s])
              for s in levels]
        ax.plot(levels, ys, "-o", ms=4, label=k)
    ax.set_xlabel("sea-level rise (m, relative to 2016)")
    ax.set_ylabel("buildings flooded above 0.30 m")
    ax.set_title("Pin Point building exposure against sea level", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160); plt.close(fig)
    print(f"figure -> {out_png}")


def fig_change_maps(ensemble_dir, dem_asc, out_png, *, level=None, wet_thresh=0.05,
                    footprints=None):
    """Depth-change map per adaptation: adaptation minus baseline, at one sea level. Red is
    deeper, blue is shallower. This is what shows displacement, where an intervention moves
    water somewhere else rather than removing it, which the summary counts cannot reveal."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Validate inputs before doing any work. An unset environment variable leaves an empty
    # path that expands to something like "/SUB_DEM_SAV.asc", and without this the failure
    # surfaces deep inside the first pair comparison rather than at the argument that caused it.
    for label, path in (("--dem", dem_asc), ("--footprints", footprints)):
        if path and not Path(path).exists():
            raise SystemExit(f"{label} does not exist: {path}\n"
                             "  (an unset $BASE or $SCR expands to nothing; export them first)")
    ens = Path(ensemble_dir)
    if not (ens / "manifest.json").exists():
        raise SystemExit(f"no manifest.json in {ens}")
    entries = json.load(open(ens / "manifest.json"))
    if level is not None:
        entries = [e for e in entries if _slr_key(e) == str(level)]
    else:
        level = _slr_key(entries[0])
        entries = [e for e in entries if _slr_key(e) == level]
    bases = [e for e in entries if _kind(e) == "base"]
    if not bases:
        raise SystemExit(f"no baseline member at sea level {level}")
    bmax = _max_path(bases[0], ens)
    b = read_depth(bmax)

    # one panel per kind, using the first finished realisation of each
    seen, panels = set(), []
    for e in entries:
        k = _kind(e)
        if k == "base" or k in seen:
            continue
        p = _max_path(e, ens)
        if p:
            panels.append((k, read_depth(p) - b)); seen.add(k)
    if not panels:
        raise SystemExit("no finished adaptation members to map")

    # Crop to where anything actually changed. Interventions are clustered at Pin Point, a few
    # hundred cells across, inside a 1356x882 domain, so the full-domain view is 99% blank and
    # the signal is invisible. The window is the union across panels so every panel shares it
    # and the maps stay comparable.
    changed = np.zeros(b.shape, bool)
    for _, d in panels:
        changed |= np.abs(d) > 1e-6
    if changed.any():
        rs, cs = np.where(changed)
        span = max(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1)
        pad = max(20, int(0.15 * span))
        r0, r1 = max(0, rs.min() - pad), min(b.shape[0], rs.max() + pad + 1)
        c0, c1 = max(0, cs.min() - pad), min(b.shape[1], cs.max() + pad + 1)
    else:
        r0, r1, c0, c1 = 0, b.shape[0], 0, b.shape[1]

    n = len(panels); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow), squeeze=False)
    vmax = max((np.abs(np.percentile(d[np.abs(d) > 1e-6], [1, 99])).max()
                if np.any(np.abs(d) > 1e-6) else 0.1) for _, d in panels) or 0.1
    # baseline extent as context, so a reader can see where the water was to begin with
    ctx = (b[r0:r1, c0:c1] > wet_thresh)
    im = None
    for ax, (k, d) in zip(axes.ravel(), panels):
        w = d[r0:r1, c0:c1]
        ax.imshow(np.where(ctx, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3, interpolation="none")
        im = ax.imshow(np.where(np.abs(w) > 1e-6, w, np.nan), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="none")
        nz = int((np.abs(w) > 1e-6).sum())
        ax.set_title(f"{k}  ({nz} cells changed)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Change in peak depth against no adaptation, sea level {level}\n"
                 "red deeper, blue shallower; grey is the baseline flood extent", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    cax = fig.add_axes([0.25, 0.03, 0.5, 0.022])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="depth change (m)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"figure -> {out_png} (cropped to rows {r0}:{r1}, cols {c0}:{c1})")


# ------------------------------------------------------------------ CLI

def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pair", help="one adaptation against one baseline")
    p.add_argument("base_max"); p.add_argument("adapt_max")
    p.add_argument("--dem", required=True)
    p.add_argument("--footprints", default=None)
    p.add_argument("--first-floor-m", type=float, default=0.0)

    e = sub.add_parser("ensemble", help="every member against its matched baseline")
    e.add_argument("ensemble_dir")
    e.add_argument("--dem", required=True)
    e.add_argument("--footprints", default=None)
    e.add_argument("--first-floor-m", type=float, default=0.0)
    e.add_argument("--out-csv", default="reports/adapt/effectiveness.csv")
    e.add_argument("--quiet", action="store_true")

    f = sub.add_parser("figures", help="headline plots from the ensemble CSV")
    f.add_argument("--csv", default="reports/adapt/effectiveness.csv")
    f.add_argument("--ensemble-dir", default=None, help="also draw the change maps")
    f.add_argument("--dem", default=None)
    f.add_argument("--level", default=None, help="sea level for the change maps")
    f.add_argument("--out-dir", default="reports/adapt")

    a = ap.parse_args(argv)
    if a.cmd == "pair":
        for k, v in compare_pair(a.base_max, a.adapt_max, a.dem, footprints=a.footprints,
                                 first_floor_m=a.first_floor_m).items():
            print(f"  {k:28s} {v:+.6g}" if isinstance(v, (int, float)) else f"  {k:28s} {v}")
    elif a.cmd == "ensemble":
        run_ensemble(a.ensemble_dir, a.dem, footprints=a.footprints,
                     first_floor_m=a.first_floor_m, out_csv=a.out_csv, quiet=a.quiet)
    else:
        out = Path(a.out_dir)
        fig_reduction_bar(a.csv, out / "adapt_flood_reduction_bar.png")
        try:
            fig_effectiveness_vs_slr(a.csv, out / "adapt_exposure_vs_slr.png")
        except SystemExit as err:
            print(f"skipped exposure-vs-SLR: {err}")
        if a.ensemble_dir and a.dem:
            fig_change_maps(a.ensemble_dir, a.dem, out / "adapt_flood_change_maps.png",
                            level=a.level)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
