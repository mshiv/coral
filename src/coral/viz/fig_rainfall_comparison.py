"""Compare rainfall products, and decide whether uniform rain is good enough.

AORC v1.1 and MRMS are both about 1 km here -- AORC returns 66x48 cells over this domain and
MRMS 55x40, so AORC is slightly finer. The comparison is of method, gauge-conditioned reanalysis
against radar with gauge correction, not of resolution.

Resolution only matters if the model uses spatial rain. With `rain_mode: uniform` the field is
collapsed to a domain-average series, and the question becomes narrower: does the domain average
differ from what actually fell on the site?

That is the comparison this figure is for. The top row is total accumulation from each product,
the difference between them, and the bottom row is the series that the model would actually be
forced with -- domain mean against the value at the site. If those two lines sit close, uniform
rain is defensible for a site-scale ensemble whatever the product resolution is.

Both fetchers write the same layout, APCP_surface(time, latitude, longitude) in mm, so one
reader serves both.

    python -m coral.viz.fig_rainfall_comparison --aorc data/interim/rain_aorc_<name>.nc \\
        --mrms data/interim/rain_mrms_<name>.nc --dem <dem.asc> \\
        --site -81.0903 31.9522 --out reports/figures/rain_comparison.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .pinpoint_style import PALETTE


def _open(path):
    """(time, lat, lon, mm) from an AORC or MRMS file written by the fetchers."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        if 'APCP_surface' not in ds:
            raise ValueError('Expected archived hourly APCP_surface precipitation')
        da = ds['APCP_surface']
        lat = da["latitude"].values if "latitude" in da.coords else da["lat"].values
        lon = da["longitude"].values if "longitude" in da.coords else da["lon"].values
        lon = np.where(lon > 180, lon - 360, lon)
        return da["time"].values.copy(), lat.copy(), lon.copy(), da.values.copy()


def _site_series(lat, lon, arr, site):
    """Series at the grid cell containing the site."""
    i = int(np.abs(lat - site[1]).argmin())
    j = int(np.abs(lon - site[0]).argmin())
    return arr[:, i, j]


def build(out, *, aorc=None, mrms=None, site=(-81.0903, 31.9522), bbox=None,
          site_label="Pin Point", publication=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = {}
    for name, p in (("AORC", aorc), ("MRMS", mrms)):
        if p and Path(p).exists():
            sets[name] = _open(p)
        else:
            print(f"  {name}: not found at {p}, skipping")
    if not sets:
        raise SystemExit("neither product was readable")
    common=next(iter(sets.values()))[0]
    for t,_,_,_ in sets.values():
        common=np.intersect1d(common,t)
    if len(common)<2 or np.any(np.diff(common)!=np.timedelta64(1,'h')):
        raise ValueError('Rainfall comparison needs common, contiguous hourly records')
    for name,(t,lat,lon,arr) in list(sets.items()):
        sets[name]=(common,lat,lon,arr[np.searchsorted(t,common)])
    shared_max=max(float(np.nanmax(np.sum(arr,axis=0))) for _,_,_,arr in sets.values())

    ncol = len(sets) + (1 if len(sets) == 2 else 0)
    fig, axes = plt.subplots(2, max(ncol, 2), figsize=(5.0 * max(ncol, 2), 8.4),
                             squeeze=False)
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]:
        ax.remove()
    axts = fig.add_subplot(gs[1, :])

    totals, stats = {}, {}
    for k, (name, (t, lat, lon, a)) in enumerate(sets.items()):
        tot = np.sum(a, axis=0)  # incomplete time series remain missing, never zero rainfall
        totals[name] = (lat, lon, tot)
        ext = (lon.min(), lon.max(), lat.min(), lat.max())
        ax = axes[0, k]
        im = ax.imshow(tot, extent=ext, origin="upper" if lat[0] > lat[-1] else "lower",
                       cmap="YlGnBu", vmin=0, vmax=shared_max)
        ax.plot(*site, marker="*", ms=13, color=PALETTE["intervention"], zorder=5)
        if bbox:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2],
                                   fill=False, ec=PALETTE["text"], lw=1.2, ls="--", zorder=4))
        ax.set_xticks([]); ax.set_yticks([])
        cell_km = abs(float(np.diff(lat)[0])) * 111.0
        ax.set_title(f"{name} total, {a.shape[2]}x{a.shape[1]} cells (~{cell_km:.1f} km)",
                     fontsize=10.5, color=PALETTE["text"])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("mm", fontsize=8)

        dom = np.nanmean(a, axis=(1, 2))
        loc = _site_series(lat, lon, a, site)
        stats[name] = (t, dom, loc)

    # difference of totals, on the coarser grid
    if len(sets) == 2 and ncol == 3:
        (la_a, lo_a, ta), (la_m, lo_m, tm) = totals["AORC"], totals["MRMS"]
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((la_m[::-1] if la_m[0] > la_m[-1] else la_m, lo_m),
                                    tm[::-1] if la_m[0] > la_m[-1] else tm,
                                    bounds_error=False, fill_value=np.nan)
        LA, LO = np.meshgrid(la_a, lo_a, indexing="ij")
        tm_on_a = f(np.stack([LA.ravel(), LO.ravel()], -1)).reshape(ta.shape)
        d = tm_on_a - ta
        ax = axes[0, 2]
        v = np.nanpercentile(np.abs(d), 98)
        im = ax.imshow(d, extent=(lo_a.min(), lo_a.max(), la_a.min(), la_a.max()),
                       origin="upper" if la_a[0] > la_a[-1] else "lower",
                       cmap="RdBu_r", vmin=-v, vmax=v)
        ax.plot(*site, marker="*", ms=13, color=PALETTE["text"], zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("MRMS minus AORC, total", fontsize=10.5, color=PALETTE["text"])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("mm", fontsize=8)

    for name, (t, dom, loc) in stats.items():
        c = PALETTE["flood"] if name == "AORC" else PALETTE["intervention"]
        th = (t - t[0]) / np.timedelta64(1, "h")
        axts.plot(th, dom, lw=1.8, color=c, label=f"{name} domain mean")
        axts.plot(th, loc, lw=1.4, ls="--", color=c, label=f"{name} at {site_label}")
        print(f"  {name}: domain total {np.nansum(dom):.1f} mm, {site_label} total "
              f"{np.nansum(loc):.1f} mm, ratio {np.nansum(loc)/max(np.nansum(dom),1e-9):.2f}; "
              f"peak hourly domain {np.nanmax(dom):.1f}, site {np.nanmax(loc):.1f} mm/hr")
    axts.set_xlabel("hours from the start of the rainfall record", fontsize=9)
    axts.set_ylabel("rainfall (mm/hr)", fontsize=9)
    axts.legend(fontsize=8.5, frameon=False, ncol=2)
    axts.grid(alpha=0.3)
    axts.set_title("What the model is forced with: solid is the uniform series, dashed is the "
                   "site", fontsize=10.5, color=PALETTE["text"])
    for s in ("top", "right"):
        axts.spines[s].set_visible(False)

    fig.subplots_adjust(hspace=0.28, wspace=0.14, top=0.93)
    fig.suptitle("Rainfall product and whether uniform forcing is enough",
                 fontsize=14, y=0.985, color=PALETTE["text"])
    fig.text(0.5, 0.005,
             "The gap between solid and dashed of the same colour is the error uniform rain "
             "makes at the site. The gap between colours is the error the product choice "
             "makes. Whichever is larger is the one worth fixing.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    if publication:
        from coral.viz.publication_style import caption_first
        caption_first(fig,[*axes[0,:],axts])
        axts.set_xlabel('Hours since '+str(common[0])[:16].replace('T',' ')+' UTC')
        for ax,name in zip(axes[0,:],list(sets)+(['MRMS − AORC'] if len(sets)==2 else [])):
            ax.set_xlabel(name)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    if publication:
        fig.savefig(Path(out).with_suffix('.pdf'),bbox_inches='tight')
    from coral.analysis.chapter_figure_bundle import file_record
    Path(out).with_suffix('.json').write_text(json.dumps(dict(
        sources={name:file_record(p) for name,p in [('AORC',aorc),('MRMS',mrms)] if p},
        common_first_record=str(common[0]),common_last_record=str(common[-1]),hourly_records=len(common),
        totals={name:dict(domain_mean_mm=float(np.sum(dom)),site_mm=float(np.sum(loc)))
                for name,(_,dom,loc) in stats.items()},
        interpretation='Product comparison, not independent observational validation; means use each archived product footprint'),indent=2))
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aorc", default=None)
    ap.add_argument("--mrms", default=None)
    ap.add_argument("--site", nargs=2, type=float, default=[-81.0903, 31.9522],
                    metavar=("LON", "LAT"))
    ap.add_argument("--site-label", default="Pin Point")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("W", "E", "S", "N"), help="draw the model domain")
    ap.add_argument("--out", default="reports/figures/rain_comparison.png")
    ap.add_argument('--publication',action='store_true')
    a = ap.parse_args()
    build(a.out, aorc=a.aorc, mrms=a.mrms, site=tuple(a.site),
          site_label=a.site_label, bbox=a.bbox, publication=a.publication)


if __name__ == "__main__":
    main()
