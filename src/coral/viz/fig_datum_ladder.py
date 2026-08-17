"""Which elevation should count as the waterline, and why.

Three numbers get used as a waterline in this project and only two of them are datums. This
figure puts them on the observed record so the choice can be made on evidence:

  A  the Matthew record, showing where 0.81 m comes from -- the stage at the landfall hour
  B  how often the water is actually above each level, over a full year of predictions
  C  how much of the domain the choice moves

Panel B is the one that decides it. A threshold for "wet often enough to be treated as water"
has to be a statement about typical conditions, so the number that matters is the fraction of
time the water sits above it. An event value cannot answer that.

Datums come from the NOAA CO-OPS station metadata API rather than being typed in, so the figure
carries its own citation.

    python -m coral.viz.fig_datum_ladder --dem <30 m DEM> --year 2016 \\
        --out reports/figures/datum_ladder.png
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from .pinpoint_style import PALETTE

STATION = "8670870"          # Fort Pulaski
MDAPI = ("https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/"
         "{s}/datums.json?units=metric")
DATA = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product={p}"
        "&application=coral&begin_date={b}&end_date={e}&datum=NAVD&station={s}"
        "&time_zone=gmt&units=metric&interval=h&format=json")


def station_datums(station=STATION):
    """Published tidal datums in m NAVD88, plus the epoch, straight from the API."""
    d = json.load(urllib.request.urlopen(MDAPI.format(s=station), timeout=60))
    v = {x["name"]: x["value"] for x in d["datums"]}
    navd = v["NAVD88"]
    keep = ("MHHW", "MHW", "MSL", "MLW", "MLLW")
    return {k: v[k] - navd for k in keep if k in v}, d.get("epoch", "?")


def series(product, begin, end, station=STATION):
    """(datetime64, level m NAVD88) for a CO-OPS product."""
    d = json.load(urllib.request.urlopen(
        DATA.format(p=product, b=begin, e=end, s=station), timeout=120))
    rows = d.get("predictions") or d.get("data") or []
    t, y = [], []
    for r in rows:
        if r.get("v") in ("", None):
            continue
        t.append(np.datetime64(r["t"].replace(" ", "T")))
        y.append(float(r["v"]))
    return np.array(t), np.array(y)


def observed_datums(year, storm=None, station=STATION):
    """MHW, MSL and MLW as realised in one year, from the 6-minute record.

    NOAA publishes datums on the 1983-2001 epoch, which is centred on 1992 and so describes a
    sea level well below a 2016 event. Extrema are separated by at least 10 h because the tide
    is semidiurnal and 6-minute data has plenty of local wiggles that are not tidal highs.
    """
    import calendar
    from scipy.signal import find_peaks
    T, Y = [], []
    for m in range(1, 13):
        last = calendar.monthrange(year, m)[1]
        try:
            t, y = series("water_level", f"{year}{m:02d}01", f"{year}{m:02d}{last}", station)
            T.append(t); Y.append(y)
        except Exception:
            pass
    if not T:
        return None
    t = np.concatenate(T); y = np.concatenate(Y)
    step = float(np.median(np.diff(t).astype("timedelta64[m]").astype(int)))
    sep = int(round(10 * 60 / step))
    hi, _ = find_peaks(y, distance=sep)
    lo, _ = find_peaks(-y, distance=sep)
    def drop(idx):
        if storm is None:
            return np.ones(len(idx), bool)
        a, b = np.datetime64(storm[0]), np.datetime64(storm[1])
        return ~((t[idx] > a) & (t[idx] < b))
    return {"MHW": float(y[hi][drop(hi)].mean()), "MSL": float(y.mean()),
            "MLW": float(y[lo][drop(lo)].mean()), "n_high": int(drop(hi).sum())}


def build(out, *, dem=None, year=2016, event=("20161005", "20161010"),
          landfall="2016-10-08T13:00", candidates=(0.81,),
          storm=("2016-10-05", "2016-10-12"), context_years=(2014, 2015, 2017, 2018)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dat, epoch = station_datums()
    print(f"NOAA {STATION} datums, epoch {epoch}, m NAVD88: "
          + ", ".join(f"{k} {v:+.3f}" for k, v in dat.items()))

    obs = observed_datums(year, storm=storm)
    if obs:
        print(f"{year} observed, storm excluded: MHW {obs['MHW']:+.3f}  MSL {obs['MSL']:+.3f}  "
              f"MLW {obs['MLW']:+.3f}  ({obs['n_high']} high waters)")
        candidates = tuple(candidates) + (obs["MHW"],)
    ctx = {}
    for yy in context_years:
        d = observed_datums(yy)
        if d:
            ctx[yy] = d["MHW"]
            print(f"  {yy} MHW {d['MHW']:+.3f}")

    te, ye = series("water_level", *event)
    tp, yp = series("predictions", *event)
    ty, yy = series("predictions", f"{year}0101", f"{year}1231")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    lines = {**dat, **{f"{c:.2f} m (landfall stage)": c for c in candidates}}
    colour = {"MHHW": "#7FB2D3", "MHW": PALETTE["flood"], "MSL": PALETTE["muted"],
              "MLW": "#9CC3D5", "MLLW": "#C9C2B6"}

    # --- A. the event -------------------------------------------------------------------
    ax = axes[0]
    ax.plot(te, ye, lw=1.4, color=PALETTE["text"], label="observed")
    ax.plot(tp, yp, lw=1.2, ls="--", color=PALETTE["flood"], label="tide prediction")
    lf = np.datetime64(landfall)
    ax.axvline(lf, color=PALETTE["intervention"], lw=1.2)
    j = int(np.abs(te - lf).argmin())
    ax.plot(te[j], ye[j], "o", ms=7, color=PALETTE["intervention"], zorder=5)
    ax.annotate(f"landfall hour\n{ye[j]:.2f} m", (te[j], ye[j]), xytext=(10, -28),
                textcoords="offset points", fontsize=8, color=PALETTE["intervention"])
    for k, v in dat.items():
        ax.axhline(v, color=colour.get(k, PALETTE["muted"]), lw=0.8, ls=":")
        ax.text(0.005, v, f" {k}", transform=ax.get_yaxis_transform(), fontsize=7,
                va="bottom", color=colour.get(k, PALETTE["muted"]))
    for c in candidates:
        ax.axhline(c, color=PALETTE["intervention"], lw=1.1)
    if ctx:
        ax.axhspan(min(ctx.values()), max(ctx.values()), color=PALETTE["intervention"],
                   alpha=0.10, lw=0)
        ax.text(0.99, max(ctx.values()), f" MHW {min(ctx)}-{max(ctx)} ", ha="right",
                va="bottom", transform=ax.get_yaxis_transform(), fontsize=7,
                color=PALETTE["intervention"])
    ax.set_ylabel("water level (m NAVD88)", fontsize=9)
    ax.set_title("A  Matthew at Fort Pulaski", fontsize=11, color=PALETTE["text"])
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7, axis="x", rotation=30)
    ax.grid(alpha=0.25)

    # --- B. exceedance ------------------------------------------------------------------
    ax = axes[1]
    lv = np.linspace(np.nanmin(yy), np.nanmax(yy), 400)
    exc = [(yy > x).mean() * 100 for x in lv]
    ax.plot(exc, lv, lw=1.8, color=PALETTE["flood"])
    for k, v in dat.items():
        p = (yy > v).mean() * 100
        ax.plot([p], [v], "o", ms=6, color=colour.get(k, PALETTE["muted"]))
        ax.annotate(f"{k}  {p:.0f}%", (p, v), xytext=(8, 0), textcoords="offset points",
                    fontsize=8, va="center", color=colour.get(k, PALETTE["muted"]))
    for c in candidates:
        p = (yy > c).mean() * 100
        ax.plot([p], [c], "o", ms=7, color=PALETTE["intervention"])
        ax.annotate(f"{c:.2f} m  {p:.0f}%", (p, c), xytext=(8, -12),
                    textcoords="offset points", fontsize=8.5, fontweight="bold",
                    color=PALETTE["intervention"])
        print(f"  {c:.2f} m is exceeded {p:.1f}% of {year}")
    for k, v in dat.items():
        print(f"  {k:5s} {v:+.3f} m is exceeded {(yy > v).mean()*100:.1f}% of {year}")
    ax.set_xlabel(f"percent of {year} the water is above this level", fontsize=9)
    ax.set_ylabel("elevation (m NAVD88)", fontsize=9)
    ax.set_title("B  How often each level is wet", fontsize=11, color=PALETTE["text"])
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25)

    # --- C. how much ground the choice moves ---------------------------------------------
    ax = axes[2]
    if dem and Path(dem).exists():
        from ..emulator.dataset import read_asc
        z, _ = read_asc(dem)
        z = z[np.isfinite(z)]
        lv2 = np.linspace(-3, 3, 400)
        frac = [(z <= x).mean() * 100 for x in lv2]
        ax.plot(frac, lv2, lw=1.8, color=PALETTE["terrain"])
        for k, v in dat.items():
            ax.axhline(v, color=colour.get(k, PALETTE["muted"]), lw=0.8, ls=":")
        for c in candidates:
            ax.axhline(c, color=PALETTE["intervention"], lw=1.1)
        a = (z <= min(candidates)).mean() * 100
        b = (z <= dat["MHW"]).mean() * 100
        ax.fill_betweenx([min(candidates), dat["MHW"]], 0, 100,
                         color=PALETTE["intervention"], alpha=0.12)
        ax.annotate(f"the band the choice moves\n{b - a:.1f}% of the domain",
                    (55, (min(candidates) + dat["MHW"]) / 2), fontsize=8.5,
                    color=PALETTE["intervention"], ha="center")
        print(f"  domain below {min(candidates):.2f} m: {a:.1f}%; below MHW: {b:.1f}%; "
              f"band {b - a:.1f}%")
        ax.set_xlabel("percent of the domain below this elevation", fontsize=9)
        ax.set_ylim(-3, 3)
    else:
        ax.text(0.5, 0.5, "no DEM given", ha="center", va="center", transform=ax.transAxes,
                color=PALETTE["muted"])
    ax.set_ylabel("elevation (m NAVD88)", fontsize=9)
    ax.set_title("C  How much ground it moves", fontsize=11, color=PALETTE["text"])
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25)

    fig.subplots_adjust(wspace=0.26, top=0.86)
    fig.suptitle(f"Choosing the waterline: NOAA {STATION} datums, epoch {epoch}",
                 fontsize=14, y=0.97, color=PALETTE["text"])
    fig.text(0.5, -0.02,
             "0.81 m has no derivation on record. The published MHW is on the 1983-2001 epoch, "
             "centred on 1992, so it sits below the sea level the event actually had. MHW "
             "computed from the event year answers the question the threshold is asking.",
             ha="center", fontsize=8.8, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default=None)
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--candidates", nargs="*", type=float, default=[0.81])
    ap.add_argument("--out", default="reports/figures/datum_ladder.png")
    a = ap.parse_args()
    build(a.out, dem=a.dem, year=a.year, candidates=tuple(a.candidates))


if __name__ == "__main__":
    main()
