"""Figure 6 — every grid the model is actually forced with.

The physics is only as good as the fields underneath it. This puts each input grid on the same
page at the same extent, so a reader can see what the terrain, roughness, infiltration and
wetland masks look like, and where they disagree with each other.

Panels are built from whatever files are given, so the same module serves the 4 m Pin Point clip
and the 30 m Savannah domain. Anything missing is skipped rather than faked.

    python -m coral.viz.fig_model_inputs --dem SUB_DEM_SAV.asc --manning Manning_SAV.asc \\
        --infil infil_matthew_compound.asc --infilcap infilcap_matthew_compound.asc \\
        --nwi data/raw/sagis_savannah/sagis_wetlands_nwi.geojson --out reports/figs/fig6.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc
from .pinpoint_style import PALETTE, extent_of, panel_title


def nlcd_on_dem(tif, dem_path):
    """NLCD class grid resampled onto the DEM grid by nearest neighbour, or None."""
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
    except ImportError:
        return None
    _, h = read_asc(dem_path)
    dst = np.zeros((int(h["nrows"]), int(h["ncols"])), "float32")
    with rasterio.open(tif) as src:
        from rasterio.transform import from_origin
        tr = from_origin(h["xllcorner"], h["yllcorner"] + h["nrows"] * h["cellsize"],
                         h["cellsize"], h["cellsize"])
        reproject(rasterio.band(src, 1), dst, dst_transform=tr, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest)
    return np.where(dst > 0, dst, np.nan)


def _mask_from_geojson(path, dem_path, prefixes=("E2EM", "E2SS", "E2FO")):
    """Vegetated estuarine wetland only. Unfiltered NWI also carries subtidal bottom and
    marine deepwater, which covers the open ocean at domain scale."""
    from ..interventions.context_rasters import wetlands_mask
    try:
        return wetlands_mask(path, dem_path, cowardin_prefixes=prefixes).astype("float32")
    except Exception:
        return None


def build(dem_path, out, *, manning=None, infil=None, infilcap=None, chm=None,
          nwi=None, nlcd=None, sea_level=0.81, title="Model inputs", publication=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dem, h = read_asc(dem_path)
    ext = extent_of(h)
    land = np.isfinite(dem) & (dem > sea_level)

    panels = [(dem, "Terrain", "elevation (m NAVD88)", "terrain", None)]

    def add(path, label, units, cmap, loader=read_asc):
        if not path:
            return
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        if loader is read_asc:
            g, gh = loader(path)
            for key in ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize'):
                if not np.isclose(gh[key], h[key], rtol=0, atol=1e-10):
                    raise ValueError(f'{path}: {key} does not match DEM')
        else:
            g = loader(path)
        if g is None:
            return
        g = np.where(np.isfinite(g) & (g > -9990), g, np.nan)
        panels.append((g, label, units, cmap, None))

    add(manning, "Roughness", "Manning n (s m$^{-1/3}$)", "YlGn")
    add(infil, "Infiltration rate", "mm h$^{-1}$", "BrBG")
    add(infilcap, "Infiltration capacity", "mm", "PuBu")
    add(chm, "Canopy height", "m", "Greens")
    if nlcd:
        g = nlcd_on_dem(nlcd, dem_path)
        if g is not None:
            panels.append((g, "Land cover", "NLCD class", "tab20", None))
    if nwi:
        g = _mask_from_geojson(nwi, dem_path)
        if g is not None:
            panels.append((np.where(g > 0, 1.0, np.nan), "Tidal wetland",
                           "vegetated marsh (NWI E2EM/E2SS/E2FO)", "summer", None))

    n = len(panels)
    ncol = (2 if n <= 4 else 3) if publication else min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), squeeze=False)

    for k, (g, label, units, cmap, _) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        if label == "Terrain":
            lo, hi = np.nanpercentile(dem[np.isfinite(dem)], [2, 98])
            im = ax.imshow(dem, extent=ext, origin="upper", cmap="terrain", vmin=lo, vmax=hi)
        elif label == 'Land cover':
            from matplotlib.colors import ListedColormap, BoundaryNorm
            classes = [11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95]
            colors = ['#466b9f','#d1def8','#dec5c5','#d99282','#eb0000','#ab0000',
                      '#b3ac9f','#68ab5f','#1c5f2c','#b5c58f','#af963c','#ccb879',
                      '#dfdfc2','#d1d182','#a3cc51','#82ba9e','#dcd939','#ab6c28',
                      '#b8d9eb','#6c9fb8']
            mapped = np.full(g.shape, np.nan)
            for idx, code in enumerate(classes):
                mapped[g == code] = idx
            cmap = ListedColormap(colors)
            im = ax.imshow(mapped, extent=ext, origin='upper', cmap=cmap,
                           norm=BoundaryNorm(np.arange(21)-.5,20), interpolation='nearest')
        else:
            v = g[np.isfinite(g)]
            lo, hi = (np.nanpercentile(v, [2, 98]) if v.size else (0, 1))
            if hi <= lo:
                hi = lo + 1e-6
            im = ax.imshow(g, extent=ext, origin="upper", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(PALETTE["muted"]); s.set_linewidth(0.5)
        if publication:
            ax.text(.02,.98,f'({chr(97+k)})', transform=ax.transAxes,va='top',
                    weight='bold',fontsize=12,bbox=dict(fc='white',ec='none',alpha=.8))
        else:
            panel_title(ax, "ABCDEFGH"[k], label, units)
        # A binary mask has nothing to scale, so a colourbar on it is noise.
        if label != "Tidal wetland":
            cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
            cb.set_label(units if label != 'Land cover' else 'NLCD category code')
            if label == 'Land cover':
                present = [i for i,c in enumerate(classes) if np.any(g == c)]
                cb.set_ticks(present, labels=[str(classes[i]) for i in present])
            cb.ax.tick_params(labelsize=9 if publication else 6.5)
        # Coverage matters as much as the values: a grid that is mostly nodata over land is a
        # silent gap in the physics.
        cov = float(np.isfinite(g).sum()) / max(int(np.isfinite(dem).sum()), 1)
        onland = float((np.isfinite(g) & land).sum()) / max(int(land.sum()), 1)
        if publication:
            continue
        wording = 'mapped habitat in' if label == 'Tidal wetland' else 'valid data over'
        ax.text(0.02, 0.02, f"{wording} {cov:.0%} of grid, {onland:.0%} of land",
                transform=ax.transAxes, fontsize=7, color=PALETTE["text"],
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.subplots_adjust(wspace=0.10, hspace=0.22, top=0.92)
    if not publication:
        fig.suptitle(title, fontsize=15, y=0.98, color=PALETTE["text"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    if publication:
        fig.savefig(Path(out).with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    Path(out).with_suffix('.json').write_text(json.dumps(dict(
        grid=h, sources={k: str(Path(v).resolve()) for k,v in dict(
            dem=dem_path,manning=manning,infil=infil,infilcap=infilcap,
            chm=chm,nwi=nwi,nlcd=nlcd).items() if v},
        display='continuous fields clipped to 2nd–98th percentiles; NLCD categorical',
        panels=[v[1] for v in panels]), indent=2))
    plt.close(fig)
    print(f"wrote {out}  ({n} panels)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--manning", default=None)
    ap.add_argument("--infil", default=None)
    ap.add_argument("--infilcap", default=None)
    ap.add_argument("--chm", default=None)
    ap.add_argument("--nwi", default=None)
    ap.add_argument("--nlcd", default=None, help="NLCD GeoTIFF; resampled onto the DEM grid")
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--title", default="Model inputs")
    ap.add_argument("--out", default="reports/figs/fig6_model_inputs.png")
    ap.add_argument('--publication', action='store_true')
    a = ap.parse_args()
    build(a.dem, a.out, manning=a.manning, infil=a.infil, infilcap=a.infilcap,
          chm=a.chm, nwi=a.nwi, nlcd=a.nlcd, sea_level=a.sea_level, title=a.title,
          publication=a.publication)


if __name__ == "__main__":
    main()
