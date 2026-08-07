"""Forcing components at the model boundary, and what the local surge model misses.

Five terms drive the flood model: storm surge from GeoClaw, astronomical tide, rainfall, river
discharge, and the post-hurricane abnormal water level (PHAWL).

The PHAWL residual is computed as

    residual(t) = observed(t) - predicted_tide(t) - (surge(t) - sea_level)

so whatever GeoClaw already contains is subtracted out and cannot be double counted. GeoClaw is
barotropic and has no Gulf Stream, so it cannot produce PHAWL; the residual isolates it.

Data: NOAA CO-OPS 8670870 Fort Pulaski (tide predictions and observed water level), USGS 02202500
Ogeechee near Eden (discharge), the model .bdy (surge), and the model rain file.
"""
import json, urllib.request
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

BDY = "runs/ab/baseline/matthew_savannah.bdy"
RAIN = "runs/ab/baseline/rain_matthew_compound.txt"
SEA_LEVEL = 0.81
# Model t=172800 s is landfall (the par comments date sim_time 259200 as landfall + 1 day).
T0 = datetime(2016, 10, 6, 12, 0, tzinfo=timezone.utc)


def read_bdy(path):
    L = open(path).read().splitlines()
    i = 1
    while not L[i].strip():
        i += 1
    n = int(L[i + 1].split()[0])
    a = np.array([[float(x) for x in r.split()[:2]] for r in L[i + 2:i + 2 + n]])
    return a[:, 1], a[:, 0]                       # seconds, stage


def coops(product, begin, end, station="8670870"):
    u = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
         f"product={product}&application=CORAL&begin_date={begin}&end_date={end}"
         f"&datum=NAVD&station={station}&time_zone=gmt&units=metric&format=json")
    d = json.load(urllib.request.urlopen(u, timeout=90))
    key = "predictions" if product == "predictions" else "data"
    t = [datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc) for r in d[key]]
    v = [float(r["v"]) if r["v"] not in ("", None) else np.nan for r in d[key]]
    return np.array(t), np.array(v)


def usgs(site, start, end):
    u = (f"https://waterservices.usgs.gov/nwis/iv/?sites={site}&startDT={start}&endDT={end}"
         "&parameterCd=00060&format=json")
    d = json.load(urllib.request.urlopen(u, timeout=90))
    vals = d["value"]["timeSeries"][0]["values"][0]["value"]
    t = [datetime.fromisoformat(x["dateTime"]).astimezone(timezone.utc) for x in vals]
    v = [float(x["value"]) * 0.0283168 for x in vals]      # cfs -> m3/s
    return np.array(t), np.array(v)


secs, surge = read_bdy(BDY)
tb = np.array([T0 + timedelta(seconds=float(s)) for s in secs])
tp, pred = coops("predictions", "20161001", "20161022")
to, obs = coops("water_level", "20161001", "20161022")
tq, q = usgs("02202500", "2016-10-01", "2016-10-22")

# PHAWL residual on the observation clock
op = np.array([d.timestamp() for d in to])
pred_o = np.interp(op, [d.timestamp() for d in tp], pred)
surge_o = np.interp(op, [d.timestamp() for d in tb], surge, left=np.nan, right=np.nan)
# Non-tidal residual, Park's NTRA: what remains after the astronomical tide is removed. During the
# storm this is the surge; after it, it is PHAWL. Defined over the whole record, unlike a difference
# against the model boundary, which only exists inside the modelled window.
ntr = obs - pred_o
geoclaw_ntr = surge_o - SEA_LEVEL          # what the local model contributes, same quantity

# LISFLOOD rain file: comment line, then "<n> seconds", then rows of "rate_mm_hr  model_seconds"
_rl = open(RAIN).read().splitlines()
rain = np.array([[float(x) for x in r.split()[:2]] for r in _rl[2:] if len(r.split()) >= 2])
rt = np.array([T0 + timedelta(seconds=float(s)) for s in rain[:, 1]])

fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                       gridspec_kw={"height_ratios": [2.4, 1.4, 1, 1.4]})

ax[0].plot(to, obs, lw=1.1, color="#333", label="observed water level (Fort Pulaski)")
ax[0].plot(tp, pred, lw=1.0, color="#0072B2", label="predicted tide")
ax[0].plot(tb, surge, lw=1.4, color="#D40000", label="GeoClaw surge (model boundary)")
ax[0].axhline(SEA_LEVEL, color="#888", ls=":", lw=1, label=f"static datum {SEA_LEVEL} m")
ax[0].set_ylabel("m NAVD88"); ax[0].legend(fontsize=8, ncol=2, loc="upper left")
ax[0].set_title("Water level components", fontsize=10)

ax[1].plot(to, ntr, lw=1.2, color="#009E73", label="observed non-tidal residual")
ax[1].plot(to, geoclaw_ntr, lw=1.4, color="#D40000", label="GeoClaw contribution")
ax[1].axhline(0, color="k", lw=.8)
ax[1].axhspan(0.20, 0.58, color="#009E73", alpha=.12)
ax[1].axvspan(datetime(2016,10,10,tzinfo=timezone.utc), datetime(2016,10,14,tzinfo=timezone.utc),
              color="#999", alpha=.18)
ax[1].legend(fontsize=8, loc="upper right")
ax[1].set_ylabel("m"); ax[1].set_title(
    "Non-tidal residual (observed $-$ tide). Green band 20$-$58 cm PHAWL, Park et al. 2024; "
    "grey 10$-$14 Oct. GeoClaw ends before it.", fontsize=9)

if rt is not None:
    ax[2].plot(rt, rain[:, 0], lw=1.2, color="#56B4E9")
    ax[2].set_ylabel("mm/hr"); ax[2].set_title("Rainfall (model forcing)", fontsize=10)

ax[3].plot(tq, q, lw=1.2, color="#E69F00")
ax[3].set_ylabel("m$^3$/s"); ax[3].set_title("Ogeechee River discharge, USGS 02202500", fontsize=10)
ax[3].set_xlabel("2016")

for a in ax:
    a.grid(alpha=.3, lw=.5)
    a.axvspan(tb[0], tb[-1], color="#D40000", alpha=.06)
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.suptitle("Compound forcing at Pin Point, Hurricane Matthew 2016. Red shading marks the modelled "
             "window; the surge boundary returns to datum while observations stay elevated.",
             fontsize=11)
fig.tight_layout()
fig.savefig("reports/fig_forcing_components.png", dpi=150, bbox_inches="tight")
m = (to >= datetime(2016,10,10,tzinfo=timezone.utc)) & (to <= datetime(2016,10,14,tzinfo=timezone.utc))
print(f"observed minus tide, 10-14 Oct: mean {np.nanmean((obs-pred_o)[m]):.3f} m, "
      f"max {np.nanmax((obs-pred_o)[m]):.3f} m")
print(f"model window: {tb[0]:%d %b %H:%M} to {tb[-1]:%d %b %H:%M}")
print("wrote reports/fig_forcing_components.png")
