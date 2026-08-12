"""Why an explicit tide raises modelled peak water level: a timing figure.

High-water marks record a maximum over time. A run forced on a static datum can only reach
the surge maximum. A run carrying an explicit tide reaches a higher combined maximum at a
different instant, whenever astronomical high water falls near the surge peak. That is
kinematics, not tide-surge dynamics, and it can be computed from the boundary alone:

    max_t(surge - z_base + tide) - max_t(surge)

For Matthew at Savannah this is +0.307 m over the 62 boundary blocks, against an observed
high-water-mark difference of +0.326 m between the tide-free and tide-inclusive runs. It is
worth stating explicitly that this coupling superposes the tide on the boundary rather than
simulating it in the surge model, so shelf-scale tide-surge interaction is absent by
construction and cannot be what the difference measures.

    python -m coral.validate.tide_timing --surge runs/ab/baseline/matthew_savannah.bdy \\
        --out reports/tide_timing.png
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..couple.build_bdy import MODEL_T0_UTC

LANDFALL_S = 172800.0


def noaa(product, begin, end, station="8670870"):
    """NOAA CO-OPS series on the model clock. product = predictions | water_level."""
    t0 = datetime.fromisoformat(MODEL_T0_UTC).replace(tzinfo=timezone.utc)
    u = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
         f"product={product}&application=coral&begin_date={begin}&end_date={end}"
         f"&datum=NAVD&station={station}&time_zone=gmt&units=metric&interval=h&format=json")
    d = json.load(urllib.request.urlopen(u, timeout=90))
    rows = d.get("predictions") or d.get("data")
    t, v = [], []
    for r in rows:
        if r.get("v") in ("", None):
            continue
        dt = datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        t.append((dt - t0).total_seconds()); v.append(float(r["v"]))
    return np.array(t), np.array(v)


def blocks(bdy):
    """All (values, model_seconds) blocks in a .bdy."""
    tok = Path(bdy).read_text().split()
    out, i = [], 0
    while i < len(tok):
        if tok[i].startswith("bc") or tok[i] in ("ocean",):
            n = int(tok[i + 1]); i += 3
            out.append((np.array(tok[i:i + 2 * n:2], dtype=float),
                        np.array(tok[i + 1:i + 2 * n:2], dtype=float)))
            i += 2 * n
        else:
            i += 1
    return out


def run(surge_bdy, out=None, *, z_base=0.42, begin="20161005", end="20161016", block=0):
    bl = blocks(surge_bdy)
    pt, pv = noaa("predictions", begin, end)

    deltas = []
    for v, t in bl:
        m = (t >= pt.min()) & (t <= pt.max())
        if m.sum() < 10:
            continue
        s = v[m]
        c = s - z_base + np.interp(t[m], pt, pv)
        deltas.append(c.max() - s.max())
    deltas = np.array(deltas)
    print(f"{len(bl)} blocks; max-sampling gain per block: mean {deltas.mean():+.3f} m, "
          f"median {np.median(deltas):+.3f}, range {deltas.min():+.3f} to {deltas.max():+.3f}")

    v, t = bl[block]
    m = (t >= pt.min()) & (t <= pt.max())
    tt, s = t[m], v[m]
    td = np.interp(tt, pt, pv)
    c = s - z_base + td
    ks, kc = int(s.argmax()), int(c.argmax())
    print(f"block {block}: surge-only max {s.max():.2f} at t={tt[ks]:.0f}; "
          f"combined max {c.max():.2f} at t={tt[kc]:.0f} (tide {td[kc]:+.2f}); "
          f"maxima {abs(tt[kc]-tt[ks])/3600:.1f} h apart")

    if out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hr = (tt - LANDFALL_S) / 3600.0
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1]))

        ax[0].plot(hr, s, color="0.35", lw=1.6, label="surge only (static datum)")
        ax[0].plot(hr, c, color="C3", lw=1.6, label="surge + explicit tide")
        ax[0].plot(hr[ks], s[ks], "o", color="0.35", ms=8, zorder=5)
        ax[0].plot(hr[kc], c[kc], "o", color="C3", ms=8, zorder=5)
        ax[0].annotate("", xy=(hr[kc], c[kc]), xytext=(hr[kc], s[ks]),
                       arrowprops=dict(arrowstyle="<->", color="C0", lw=1.6))
        ax[0].text(hr[kc] + 1.2, (c[kc] + s[ks]) / 2,
                   f"+{c[kc]-s[ks]:.2f} m\nhigher maximum",
                   fontsize=9, color="C0", va="center")
        ax[0].axvline(0, color="0.6", ls="--", lw=0.8)
        ax[0].text(0.3, ax[0].get_ylim()[0], " landfall", fontsize=8, color="0.4")
        ax[0].set_ylabel("water level (m NAVD88)")
        ax[0].set_title("The two runs reach their maxima at different times\n"
                        f"maxima {abs(hr[kc]-hr[ks]):.1f} h apart; the tide run is higher by "
                        f"{c[kc]-s[ks]:.2f} m", fontsize=11)
        ax[0].legend(fontsize=9, loc="upper left"); ax[0].grid(alpha=0.3)

        ax[1].plot(hr, td, color="C0", lw=1.4, label="NOAA astronomical tide")
        ax[1].axhline(0, color="0.7", lw=0.8)
        ax[1].axvline(0, color="0.6", ls="--", lw=0.8)
        ax[1].plot(hr[kc], td[kc], "o", color="C0", ms=7)
        ax[1].annotate(f"tide {td[kc]:+.2f} m at the combined maximum",
                       (hr[kc], td[kc]), xytext=(8, 8), textcoords="offset points", fontsize=8)
        ax[1].plot(hr[ks], td[ks], "o", color="0.35", ms=6)
        ax[1].annotate(f"tide {td[ks]:+.2f} m at the surge maximum",
                       (hr[ks], td[ks]), xytext=(8, -14), textcoords="offset points", fontsize=8)
        ax[1].set_ylabel("tide (m)"); ax[1].set_xlabel("hours from landfall")
        ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.3)

        fig.text(0.5, -0.02,
                 f"Across all {len(deltas)} boundary blocks the gain is "
                 f"{deltas.mean():+.3f} m (range {deltas.min():+.3f} to {deltas.max():+.3f}), "
                 f"against an observed high-water-mark difference of +0.326 m.\n"
                 "The tide is superposed on the boundary, not simulated in the surge model, so "
                 "this is a maximum-sampling effect and not tide-surge interaction.",
                 ha="center", fontsize=8.5)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"wrote {out}")
    return deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surge", required=True, help="tide-free .bdy (surge with static datum)")
    ap.add_argument("--z-base", type=float, default=0.42,
                    help="static datum removed before superposing the tide")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--out", default="reports/tide_timing.png")
    a = ap.parse_args()
    run(a.surge, a.out, z_base=a.z_base, block=a.block)


if __name__ == "__main__":
    main()
