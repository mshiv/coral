"""Visualise what the random intervention placements actually look like.

The ensemble samples each adaptation as a Gaussian random field thresholded inside an eligible
zone, so every member is a different footprint. That is easy to describe and hard to picture,
and a placement scheme that looks reasonable in code can still put marsh on rooftops. This
draws a grid of real members straight from the ensemble so the sampled footprints can be
inspected rather than assumed.

Framing. The training ensemble uses `focus_radius_km: 10.0`, so placements are spread over a
20 km circle rather than clustered on the community. Panels are therefore cropped to the union
of changed cells across every sample shown, which keeps them directly comparable, and each
carries two reference marks: Pin Point itself, and the rectangle of the clip whose 909
buildings are the exposure metric. Without those a viewer cannot tell whether an intervention
sits anywhere near the buildings it is supposed to protect.

The background is the baseline flood extent in grey, so a footprint can be read against the
water it is meant to act on.

Deps: numpy, matplotlib.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np

from .building_exposure import read_depth

# which grid carries the intervention's primary signal
PRIMARY = {"seawall": "dem", "retreat": "dem"}          # elevation edits; everything else is n
GRIDS = {"dem": "SUB_DEM*.asc", "manning": "Manning*.asc",
         "ksat": "infil_*.asc", "awc": "infilcap_*.asc"}

# Grids are written with four decimals, so every cell of an edited grid differs from the
# original by up to 5e-5 purely from rounding. At a 1e-6 tolerance that reads as "97% of the
# domain changed" and buries the actual footprint in noise. 1e-3 is far above the rounding
# floor and far below any real edit: elevations change by metres, Manning's n by ~0.07.
TOL = 1e-3


def _read(p):
    return np.loadtxt(p, skiprows=6)


def _hdr(p):
    h = {}
    with open(p) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
    return h


def _rowcol(h, lon, lat):
    """Grid indices for a lon/lat, for placing reference marks."""
    c = int((lon - h["xllcorner"]) / h["cellsize"])
    r = int(((h["yllcorner"] + h["nrows"] * h["cellsize"]) - lat) / h["cellsize"])
    return r, c


def member_deltas(run_dir, base_dir):
    """Per-grid difference against the base run, skipping grids the member symlinks.

    A member that did not edit a grid symlinks it to the base file, so comparing resolved paths
    identifies the untouched grids without reading them. That is most of the work avoided: a
    single-intervention member edits one or two of the four.
    """
    out = {}
    for name, pat in GRIDS.items():
        m = next(iter(sorted(Path(run_dir).glob(pat))), None)
        b = next(iter(sorted(Path(base_dir).glob(pat))), None)
        if m is None or b is None:
            continue
        if os.path.realpath(m) == os.path.realpath(b):
            continue                              # symlinked, therefore unchanged
        d = _read(m) - _read(b)
        d[~np.isfinite(d)] = 0.0
        if np.any(np.abs(d) > TOL):
            out[name] = d
    return out


def pick_samples(manifest, kinds=None, per_kind=3, seed=0):
    """A spread of members per kind: smallest, median and largest treated area, so a panel row
    shows the range the sampler actually produces rather than three draws that happen to look
    alike."""
    by = {}
    for e in manifest:
        iv = (e.get("interventions") or [{}])[0]
        k = iv.get("kind")
        if not k or (kinds and k not in kinds):
            continue
        size = iv.get("area_frac", iv.get("crest_m", 0.0))
        by.setdefault(k, []).append((float(size), e))
    out = []
    for k in sorted(by):
        rows = sorted(by[k], key=lambda x: x[0])
        if per_kind >= len(rows):
            pick = rows
        else:
            idx = np.linspace(0, len(rows) - 1, per_kind).round().astype(int)
            pick = [rows[i] for i in idx]
        out += [(k, s, e) for s, e in pick]
    return out


def _window(changed, h, ref_point, clip_bbox, pad=25):
    """Crop over the changed cells plus the reference marks, so an intervention far from the
    community still shows Pin Point and the building-clip rectangle."""
    rs, cs = np.where(changed)
    pr, pc = _rowcol(h, *ref_point)
    w, e_, s, n = clip_bbox
    br, bc = _rowcol(h, w, n)
    tr_, tc = _rowcol(h, e_, s)
    rmin = min(rs.min(), pr, br, tr_); rmax = max(rs.max(), pr, br, tr_)
    cmin = min(cs.min(), pc, bc, tc); cmax = max(cs.max(), pc, bc, tc)
    r0, r1 = max(0, rmin - pad), min(changed.shape[0], rmax + pad + 1)
    c0, c1 = max(0, cmin - pad), min(changed.shape[1], cmax + pad + 1)
    return r0, r1, c0, c1


def gallery(ensemble_dir, base_dir, out_png, *, kinds=None, per_kind=3,
            ref_point=(-81.0903, 31.9522),
            clip_bbox=(-81.1103, -81.0727, 31.9367, 31.9690), seed=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ens = Path(ensemble_dir)
    manifest = json.load(open(ens / "manifest.json"))
    picks = pick_samples(manifest, kinds=kinds, per_kind=per_kind, seed=seed)
    if not picks:
        raise SystemExit("no members matched; check --kinds against the manifest")

    dem_p = next(iter(sorted(Path(base_dir).glob("SUB_DEM*.asc"))))
    h = _hdr(dem_p)

    panels = []
    for kind, size, e in picks:
        rd = Path(e["run_dir"])
        if not rd.is_absolute():
            rd = ens / rd.name
        if not rd.is_dir():
            continue
        d = member_deltas(rd, base_dir)
        if not d:
            continue
        field = PRIMARY.get(kind, "manning")
        if field not in d:
            field = sorted(d, key=lambda k: -np.abs(d[k]).sum())[0]
        panels.append((kind, size, e["name"], field, d[field], sorted(d)))
    if not panels:
        raise SystemExit("no member had an edited grid; are they all symlinked to base?")

    changed = np.zeros_like(panels[0][4], bool)
    for *_, arr, _ in panels:
        changed |= np.abs(arr) > TOL
    r0, r1, c0, c1 = _window(changed, h, ref_point, clip_bbox)
    pr, pc = _rowcol(h, *ref_point)
    w, e_, s, n = clip_bbox
    br, bc = _rowcol(h, w, n)
    tr_, tc = _rowcol(h, e_, s)

    bmax = next(iter(sorted(Path(base_dir).glob("results_*/*.max"))), None)
    wet = (read_depth(bmax) > 0.05)[r0:r1, c0:c1] if bmax else None

    # one row per kind, per_kind columns: a row is then the range the sampler produces for
    # that adaptation, which is the comparison the reader wants to make.
    kinds_seen = sorted({p[0] for p in panels})
    nrow, ncol = len(kinds_seen), max(1, per_kind)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), squeeze=False)
    slots = {k: 0 for k in kinds_seen}
    used = set()
    for kind, size, name, field, arr, all_f in panels:
        r = kinds_seen.index(kind); c = slots[kind]; slots[kind] += 1
        if c >= ncol:
            continue
        ax = axes[r][c]; used.add((r, c))
        wnd = arr[r0:r1, c0:c1]
        if wet is not None:
            ax.imshow(np.where(wet, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3,
                      interpolation="none")
        v = np.abs(wnd[np.abs(wnd) > TOL])
        vmax = np.percentile(v, 99) if v.size else 1.0
        im = ax.imshow(np.where(np.abs(wnd) > TOL, wnd, np.nan), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="none")
        ax.plot(pc - c0, pr - r0, marker="*", ms=13, color="#111", mew=0.6, mec="w", zorder=6)
        ax.add_patch(Rectangle((bc - c0, br - r0), tc - bc, tr_ - br, fill=False,
                               ec="#111", lw=1.1, ls="--", zorder=5))
        frac = 100.0 * (np.abs(arr) > TOL).sum() / arr.size
        ax.set_title(f"{kind}  size {size:.2f}   {frac:.2f}% of domain\n"
                     f"{field} delta   edits: {', '.join(all_f)}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, (c1 - c0) - 0.5); ax.set_ylim((r1 - r0) - 0.5, -0.5)
        fig.colorbar(im, ax=ax, shrink=.75)
    for r in range(nrow):
        for c in range(ncol):
            if (r, c) not in used:
                axes[r][c].axis("off")
    fig.suptitle("Sampled intervention placements, real ensemble members\n"
                 "star = Pin Point, dashed box = the clip whose 909 buildings are counted, "
                 "grey = baseline flood extent", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.055 / max(nrow, 1) * 3))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"gallery -> {out_png}  ({len(panels)} members, crop rows {r0}:{r1} cols {c0}:{c1})")
    return out_png


def member_fields(ensemble_dir, base_dir, out_dir, *, kinds=None, per_kind=3,
                  ref_point=(-81.0903, 31.9522),
                  clip_bbox=(-81.1103, -81.0727, 31.9367, 31.9690), seed=0):
    """One figure per sampled member, with a panel for every edited grid plus the footprint mask.

    The gallery() grid draws only the primary signal (DEM for seawall and retreat, Manning
    otherwise) and leaves the other edits as a string in the title. That is enough to judge
    placement but not to see what a seawall actually changes: it raises the DEM and sets
    n=0.015 on the wall cells, and the gallery never draws that second edit. This writes one
    PNG per member, each with a panel per edited grid (dem, manning, ksat, awc) and a final
    mask panel showing the union of all changed cells, so the figure shows everything a member
    touched rather than the one field the gallery chose to highlight.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ens = Path(ensemble_dir)
    manifest = json.load(open(ens / "manifest.json"))
    picks = pick_samples(manifest, kinds=kinds, per_kind=per_kind, seed=seed)
    if not picks:
        raise SystemExit("no members matched; check --kinds against the manifest")

    dem_p = next(iter(sorted(Path(base_dir).glob("SUB_DEM*.asc"))))
    h = _hdr(dem_p)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bmax = next(iter(sorted(Path(base_dir).glob("results_*/*.max"))), None)
    wet_full = (read_depth(bmax) > 0.05) if bmax else None
    pr, pc = _rowcol(h, *ref_point)
    w, e_, s, n = clip_bbox
    br, bc = _rowcol(h, w, n)
    tr_, tc = _rowcol(h, e_, s)

    written = 0
    for kind, size, e in picks:
        rd = Path(e["run_dir"])
        if not rd.is_absolute():
            rd = ens / rd.name
        if not rd.is_dir():
            continue
        d = member_deltas(rd, base_dir)
        if not d:
            continue
        names = sorted(d)
        mask = np.zeros_like(d[names[0]], bool)
        for arr in d.values():
            mask |= np.abs(arr) > TOL
        r0, r1, c0, c1 = _window(mask, h, ref_point, clip_bbox)
        wet = wet_full[r0:r1, c0:c1] if wet_full is not None else None

        ncol = len(names) + 1
        fig, axes = plt.subplots(1, ncol, figsize=(3.9 * ncol, 4.0), squeeze=False)
        # first panel: the mask, i.e. the union of every grid a member edited
        ax = axes[0][0]
        wnd = mask[r0:r1, c0:c1]
        if wet is not None:
            ax.imshow(np.where(wet, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3,
                      interpolation="none")
        ax.imshow(np.where(wnd, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1,
                  interpolation="none", alpha=0.85)
        ax.plot(pc - c0, pr - r0, marker="*", ms=13, color="#111", mew=0.6, mec="w", zorder=6)
        ax.add_patch(Rectangle((bc - c0, br - r0), tc - bc, tr_ - br, fill=False,
                               ec="#111", lw=1.1, ls="--", zorder=5))
        frac = 100.0 * wnd.sum() / wnd.size
        ax.set_title(f"mask  {frac:.2f}% of domain", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, (c1 - c0) - 0.5); ax.set_ylim((r1 - r0) - 0.5, -0.5)
        # one panel per edited grid
        for j, name in enumerate(names):
            ax = axes[0][j + 1]
            wnd = d[name][r0:r1, c0:c1]
            if wet is not None:
                ax.imshow(np.where(wet, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3,
                          interpolation="none")
            v = np.abs(wnd[np.abs(wnd) > TOL])
            vmax = np.percentile(v, 99) if v.size else 1.0
            im = ax.imshow(np.where(np.abs(wnd) > TOL, wnd, np.nan), cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, interpolation="none")
            ax.plot(pc - c0, pr - r0, marker="*", ms=13, color="#111", mew=0.6, mec="w", zorder=6)
            ax.add_patch(Rectangle((bc - c0, br - r0), tc - bc, tr_ - br, fill=False,
                                   ec="#111", lw=1.1, ls="--", zorder=5))
            f2 = 100.0 * (np.abs(wnd) > TOL).sum() / wnd.size
            ax.set_title(f"{name}  {f2:.2f}%", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-0.5, (c1 - c0) - 0.5); ax.set_ylim((r1 - r0) - 0.5, -0.5)
            fig.colorbar(im, ax=ax, shrink=.75)
        fig.suptitle(f"{kind}  size {size:.2f}  (member {e['name']})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        png = out / f"{kind}_{e['name']}.png"
        fig.savefig(png, dpi=150); plt.close(fig)
        written += 1
        print(f"member_fields -> {png}  (edits: {', '.join(names)})")
    if not written:
        raise SystemExit("no members had an edited grid; are they all symlinked to base?")
    return out


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Grid of sampled intervention placements")
    ap.add_argument("ensemble_dir")
    ap.add_argument("--base", required=True, help="the base run the ensemble was built from")
    ap.add_argument("--kinds", nargs="+", default=None)
    ap.add_argument("--per-kind", type=int, default=3)
    ap.add_argument("--out", default="reports/adapt/intervention_gallery.png")
    ap.add_argument("--fields-out", default=None,
                    help="also write one figure per member with a panel for every edited grid "
                         "plus the footprint mask")
    a = ap.parse_args(argv)
    gallery(a.ensemble_dir, a.base, a.out, kinds=a.kinds, per_kind=a.per_kind)
    if a.fields_out:
        member_fields(a.ensemble_dir, a.base, a.fields_out, kinds=a.kinds, per_kind=a.per_kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
