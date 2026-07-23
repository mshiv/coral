#!/usr/bin/env python3
"""
Validate modelled flood against USGS high-water marks (HWMs) for Hurricane
Matthew (USGS STN event 135). Samples modelled max water-surface elevation
(.mxe) at each HWM in the domain and compares to the observed HWM elevation.

Observed HWM elev_ft is a water-surface elevation (NAVD88, hdatum_id 4 typical);
converted to metres. Modelled .mxe is max water-surface elevation. Comparable
(mind the GeoClaw MSL vs NAVD88 ~0.07 m offset + the known surge undershoot).

Deps: numpy, matplotlib, requests
Usage:
  python 08_validate_hwm.py --mxe ../lisflood/results_matthew_sav/res_matthew_sav.mxe \
      --dem ../inputs/SUB_DEM_SAV.asc --out fig_hwm_validation.png
"""
import argparse
import json
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FT2M = 0.3048
EVENT = 135  # 2016 Matthew (USGS STN)


def read_grid(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.where((a == h.get("nodata_value", -9999)) | (a <= -9990), np.nan, a)
    nx, ny, cs = int(h["ncols"]), int(h["nrows"]), h["cellsize"]
    x0, y0 = h["xllcorner"], h["yllcorner"]
    x = x0 + (np.arange(nx) + 0.5) * cs
    y = (y0 + ny * cs) - (np.arange(ny) + 0.5) * cs   # rows N->S
    ext = [x0, x0 + nx * cs, y0, y0 + ny * cs]
    return a, x, y, ext


def fetch_hwms(bbox):
    url = f"https://stn.wim.usgs.gov/STNServices/Events/{EVENT}/HWMs.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.loads(r.read().decode())
    w, e, s, n = bbox
    out = []
    for h in d:
        lon, lat, ev = h.get("longitude_dd"), h.get("latitude_dd"), h.get("elev_ft")
        if lon and lat and ev and w < lon < e and s < lat < n:
            out.append((lon, lat, ev * FT2M, h.get("hwm_quality_id")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mxe", required=True, help="modelled max water-surface elev grid")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", default="fig_hwm_validation.png")
    ap.add_argument("--buffer", type=float, default=0.0,
                    help="extra degrees around the domain to include HWMs")
    args = ap.parse_args()

    mxe, gx, gy, ext = read_grid(args.mxe)
    bbox = (ext[0] - args.buffer, ext[1] + args.buffer,
            ext[2] - args.buffer, ext[3] + args.buffer)
    hwms = fetch_hwms(bbox)
    if not hwms:
        raise SystemExit("no HWMs in domain")

    obs, mod, lons, lats = [], [], [], []
    for lon, lat, ev_m, q in hwms:
        i = np.argmin(np.abs(gx - lon)); j = np.argmin(np.abs(gy - lat))
        m = mxe[j, i]
        if np.isfinite(m):                # only where the model has water
            obs.append(ev_m); mod.append(m); lons.append(lon); lats.append(lat)
    obs = np.array(obs); mod = np.array(mod)
    if obs.size == 0:
        raise SystemExit("no HWMs coincide with modelled wet cells")

    resid = mod - obs
    rmse = np.sqrt(np.mean(resid**2)); bias = resid.mean()
    print(f"{len(hwms)} HWMs in domain; {obs.size} on modelled wet cells")
    print(f"RMSE {rmse:.2f} m, bias {bias:+.2f} m (model - obs)")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # scatter
    lim = [min(obs.min(), mod.min()) - 0.3, max(obs.max(), mod.max()) + 0.3]
    ax[0].plot(lim, lim, "k--", lw=1, label="1:1")
    ax[0].scatter(obs, mod, c="steelblue", ec="k", s=45)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_aspect("equal")
    ax[0].set_xlabel("observed HWM elevation (m)")
    ax[0].set_ylabel("modelled max WSE (m)")
    ax[0].set_title(f"HWM validation — RMSE {rmse:.2f} m, bias {bias:+.2f} m")
    ax[0].grid(alpha=0.3); ax[0].legend()
    # map of residuals
    mp = ax[1].imshow(np.where(np.isfinite(mxe), mxe, np.nan), extent=ext,
                      origin="upper", cmap="Greys", alpha=0.4)
    sc = ax[1].scatter(lons, lats, c=resid, cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                       ec="k", s=55)
    plt.colorbar(sc, ax=ax[1], shrink=0.8, label="model - obs (m)")
    ax[1].set_title("HWM residuals (red = model high, blue = model low)")
    ax[1].set_xlabel("lon"); ax[1].set_ylabel("lat")
    fig.suptitle("Modelled flood vs USGS high-water marks — Matthew 2016",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
