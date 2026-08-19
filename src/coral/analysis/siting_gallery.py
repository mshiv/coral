"""Gallery of intervention placements, generated fresh from the siting code.

Runs `sample_intervention` and `apply_interventions` on a base run's grids, so the figure
shows what the sweep would actually place rather than re-reading members. Useful when the
ensemble is not to hand.

Fixes three things that make a whole-domain gallery unreadable:

- **Zoom.** Interventions occupy a fraction of a percent of the domain, so a full-extent panel
  shows a blank square. Panels are cropped to the intervention's own bounding box plus a margin,
  or to a fixed window around the site.
- **Shared colour scale per kind.** A per-panel colorbar makes realisations look different when
  only the scale changed. One scale per kind, symmetric about zero.
- **Terrain context.** The edit is drawn over hillshaded terrain with building footprints, so a
  reader can see what is being protected.

    python -m coral.analysis.siting_gallery --base runs/pinpoint_highres_4m \\
        --flood-depth runs/pinpoint_highres_4m/results_full/res_pinpoint_highres_4m.max \\
        --buildings data/raw/fema_structures_pinpoint.geojson \\
        --out reports/adapt/siting_gallery.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc
from ..interventions.generate import sample_intervention, apply_intervention

KINDS = ["floodwall", "living_shoreline", "marsh_restoration", "marsh_migration",
         "depave", "retreat", "road_raise"]
# Which grid each kind edits most visibly, and a label for the colour bar.
FIELD = {"floodwall": ("dem", "DEM raised (m)"),
         "seawall": ("dem", "DEM raised (m)"),        # legacy manifests
         "road_raise": ("dem", "DEM raised (m)"),
         "retreat": ("dem", "DEM change (m)"),
         "living_shoreline": ("manning", "Manning n change"),
         "marsh": ("manning", "Manning n change"),
         "depave": ("manning", "Manning n change")}


def _extent(h):
    ny, nx, cs = h["nrows"], h["ncols"], h["cellsize"]
    return (h["xllcorner"], h["xllcorner"] + nx * cs,
            h["yllcorner"], h["yllcorner"] + ny * cs)


def _hillshade(z, az=315.0, alt=45.0):
    dy, dx = np.gradient(np.nan_to_num(z, nan=float(np.nanmin(z))))
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, z0 = np.radians(az), np.radians(alt)
    return (np.sin(z0) * np.sin(slope)
            + np.cos(z0) * np.cos(slope) * np.cos(a - np.pi / 2 - aspect))


def _crop(mask, shape, pad_cells):
    """Bounding box of the edited cells, padded, clipped to the grid."""
    rr, cc = np.where(mask)
    if rr.size == 0:
        return None
    r0, r1 = max(rr.min() - pad_cells, 0), min(rr.max() + pad_cells + 1, shape[0])
    c0, c1 = max(cc.min() - pad_cells, 0), min(cc.max() + pad_cells + 1, shape[1])
    # keep the panel from being a sliver
    if r1 - r0 < 40:
        m = (r0 + r1) // 2; r0, r1 = max(m - 20, 0), min(m + 20, shape[0])
    if c1 - c0 < 40:
        m = (c0 + c1) // 2; c0, c1 = max(m - 20, 0), min(m + 20, shape[1])
    return r0, r1, c0, c1


def build(base, out, *, kinds=KINDS, per_kind=4, flood_depth=None, buildings=None,
          wetlands=None, seed=7, pad_m=250.0, sea_level=0.81):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    base = Path(base)
    dem_p = next(base.glob("SUB_DEM_corr*.asc"), None) or next(base.glob("SUB_DEM*.asc"))
    man_p = next(base.glob("Manning_veg*.asc"), None) or next(base.glob("Manning*.asc"))
    dem0, h = read_asc(dem_p)
    man0, _ = read_asc(man_p)
    res_m = h["cellsize"] * 111000.0
    pad = max(int(pad_m / res_m), 5)
    print(f"base: {dem_p.name}, {man_p.name}  ({res_m:.1f} m cells)")

    fdep = None
    if flood_depth:
        fdep, _ = read_asc(flood_depth)
        fdep = np.where(np.isfinite(fdep) & (fdep > 0), fdep, 0.0)

    bpoly = None
    if buildings:
        try:
            import geopandas as gpd
            bpoly = gpd.read_file(buildings).to_crs("EPSG:4326")
        except Exception as e:
            print(f"  buildings not drawn: {e}")

    ksat0 = np.full_like(dem0, 50.0)
    awc0 = np.full_like(dem0, 100.0)
    focus = None
    bmask = None
    if bpoly is not None:
        try:
            from rasterio.features import rasterize
            from rasterio.transform import from_origin
            tr = from_origin(h["xllcorner"], h["yllcorner"] + h["nrows"] * h["cellsize"],
                             h["cellsize"], h["cellsize"])
            bmask = rasterize(((g, 1) for g in bpoly.geometry), out_shape=dem0.shape,
                              transform=tr, fill=0, dtype="uint8").astype(bool)
            print(f"  buildings rasterised: {int(bmask.sum())} cells")
        except Exception as e:
            print(f"  building mask not built: {e}")

    wmask = None
    if wetlands:
        try:
            from ..interventions.context_rasters import wetlands_mask
            wmask = wetlands_mask(wetlands, str(dem_p))
            print(f"  wetlands rasterised: {int(wmask.sum())} cells")
        except Exception as e:
            print(f"  wetlands mask failed: {e}")

    hs = _hillshade(dem0)
    ext_full = _extent(h)
    rng = np.random.default_rng(seed)

    # Generate first so each kind's colour scale can span its realisations.
    panels = {k: [] for k in kinds}
    for k in kinds:
        tries = 0
        while len(panels[k]) < per_kind and tries < per_kind * 6:
            tries += 1
            knobs = sample_intervention(k, rng)
            knobs["alignment_index"] = tries          # distinct alignments for linear kinds
            dem1, man1, _, _, _ = apply_intervention(
                knobs, dem0.copy(), man0.copy(), ksat0.copy(), awc0.copy(),
                sea_level=sea_level, res_m=res_m, focus=focus, buildings=bmask, wetlands=wmask,
                place=("targeted" if fdep is not None else "random"), flood_depth=fdep)
            # Use whichever grid this realisation actually edited. living_shoreline raises the
            # DEM by a sill AND takes max(manning, n_target); a low n_target draw leaves Manning
            # untouched, so keying on one field alone silently drops the realisation.
            dd, dm = dem1 - dem0, man1 - man0
            nd, nm = int((np.abs(dd) > 1e-9).sum()), int((np.abs(dm) > 1e-9).sum())
            if nd == 0 and nm == 0:
                continue
            if nd >= nm:
                delta, label = dd, "DEM change (m)"
            else:
                delta, label = dm, "Manning n change"
            panels[k].append((knobs, delta, label))
        print(f"  {k:18s} {len(panels[k])} placements from {tries} draws")

    nrow = len(kinds)
    ncol = max(1, max(len(v) for v in panels.values()))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.7 * nrow), squeeze=False)

    for i, k in enumerate(kinds):
        vals = [np.abs(d[np.abs(d) > 1e-9]) for _, d, _ in panels[k]]
        vmax = float(np.percentile(np.concatenate(vals), 99)) if vals else 1.0
        vmax = max(vmax, 1e-6)
        for j in range(ncol):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(panels[k]):
                ax.axis("off"); continue
            knobs, delta, dlabel = panels[k][j]
            box = _crop(np.abs(delta) > 1e-9, dem0.shape, pad)
            r0, r1, c0, c1 = box
            cs = h["cellsize"]
            ext = (h["xllcorner"] + c0 * cs, h["xllcorner"] + c1 * cs,
                   h["yllcorner"] + (dem0.shape[0] - r1) * cs,
                   h["yllcorner"] + (dem0.shape[0] - r0) * cs)
            ax.imshow(hs[r0:r1, c0:c1], cmap="Greys_r", extent=ext, origin="upper",
                      alpha=0.55, zorder=0)
            ax.imshow(np.where(dem0[r0:r1, c0:c1] < sea_level, 1, np.nan), extent=ext,
                      origin="upper", cmap="Blues", vmin=0, vmax=1.6, zorder=1)
            if bpoly is not None:
                bpoly.plot(ax=ax, facecolor="0.15", edgecolor="none", lw=0, zorder=2)
            sub = delta[r0:r1, c0:c1]
            # Sparse edits vanish at panel scale. Dilate for display only, never for statistics.
            frac = float((np.abs(sub) > 1e-9).mean())
            if 0 < frac < 0.02:
                from scipy import ndimage as _nd
                grow = _nd.grey_dilation(np.abs(sub), size=3) * np.sign(
                    _nd.grey_dilation(sub, size=3) + _nd.grey_erosion(sub, size=3) + 1e-12)
                sub = np.where(np.abs(sub) > 1e-9, sub, grow)
            d = np.where(np.abs(sub) > 1e-9, sub, np.nan)
            im = ax.imshow(d, extent=ext, origin="upper", zorder=3,
                           cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -vmax, vmax))
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            n_edit = int((np.abs(delta) > 1e-9).sum())
            area = n_edit * res_m ** 2 / 1e4
            bits = [f"{v:.2f}" if isinstance(v, float) else str(v)
                    for kk, v in knobs.items() if kk in ("crest_m", "length_m", "size")]
            ax.set_title(f"{k} {'/'.join(bits)}\n{n_edit} cells, {area:.1f} ha",
                         fontsize=7)
            if j == len(panels[k]) - 1:
                cb = fig.colorbar(im, ax=axes[i][:], fraction=0.02, pad=0.01)
                cb.set_label(dlabel, fontsize=7)
                cb.ax.tick_params(labelsize=6)

    fig.suptitle("Intervention placements generated by the siting code, 4 m Pin Point\n"
                 "each panel cropped to its own edit; one colour scale per kind; "
                 "blue = water, dark = FEMA structures", fontsize=10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="reports/adapt/siting_gallery.png")
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--per-kind", type=int, default=4)
    ap.add_argument("--flood-depth", default=None)
    ap.add_argument("--buildings", default=None)
    ap.add_argument("--wetlands", default=None,
                    help="NWI geojson. REQUIRED for targeted living_shoreline: siting.py returns "
                         "an empty mask without it (siting.py:60)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(a.base, a.out, kinds=a.kinds, per_kind=a.per_kind, flood_depth=a.flood_depth,
          buildings=a.buildings, wetlands=a.wetlands, seed=a.seed)


if __name__ == "__main__":
    main()
