"""Compare the coupling boundary between two GeoClaw refinement levels.

The 30 m LISFLOOD run reads GeoClaw only at the coupling gauges, so the resolution question is
not "does the surge field look better", it is "does the boundary series the child actually
consumes change". This compares eta gauge by gauge between two runs and reports the difference
in the terms that matter downstream: the peak each gauge sees, and the residual over the whole
series.

Gauge ids 1..N are the coupling front; 9001-9005 are the NOAA validation stations and are
reported separately rather than pooled, since they sit inside the estuary and answer a
different question.

The two runs use different timesteps, so every series is interpolated onto the coarser run's
gauge times before differencing. Comparing raw samples would report the sampling difference as
a physical one.

    python -m coral.viz.fig_amr_convergence --coarse <l6>/_output --fine <l7>/_output \\
        --labels "level 6 (145 m)" "level 7 (36 m)" --out reports/figures/amr_convergence.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from .gauge_io import read_gauge
from .pinpoint_style import PALETTE


def load_run(out_dir, max_coupling=8999):
    """{gauge_id: (lat, t, eta)} for the coupling gauges in one _output directory."""
    runs = {}
    for p in sorted(Path(out_dir).glob("gauge*.txt")):
        try:
            gid = int(p.stem.replace("gauge", ""))
        except ValueError:
            continue
        lon, lat, t, eta = read_gauge(p)
        if lat is None or t.size < 2:
            continue
        runs[gid] = (lat, t, eta)
    return runs


def compare(coarse, fine):
    """Per-gauge stats on the common time span, coarse gauge times as the reference grid."""
    rows = []
    for gid, (lat, tc, ec) in coarse.items():
        if gid not in fine:
            continue
        _, tf, ef = fine[gid]
        lo, hi = max(tc.min(), tf.min()), min(tc.max(), tf.max())
        m = (tc >= lo) & (tc <= hi)
        if m.sum() < 2:
            continue
        t = tc[m]
        a = ec[m]
        b = np.interp(t, tf, ef)
        d = b - a
        rows.append((gid, lat, a.max(), b.max(), b.max() - a.max(),
                     float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max())))
    dt = np.dtype([("gid", int), ("lat", float), ("peak_c", float), ("peak_f", float),
                   ("dpeak", float), ("rmse", float), ("dmax", float)])
    return np.array(rows, dtype=dt)


def build(coarse_dir, fine_dir, out, *, labels=("coarse", "fine"), n_show=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_all, f_all = load_run(coarse_dir), load_run(fine_dir)
    coup_c = {k: v for k, v in c_all.items() if k < 9000}
    coup_f = {k: v for k, v in f_all.items() if k < 9000}
    st = compare(coup_c, coup_f)
    if st.size == 0:
        raise SystemExit("no gauges in common; check the two _output paths")
    st = st[np.argsort(st["lat"])]

    print(f"{len(st)} coupling gauges compared, {labels[1]} against {labels[0]}")
    print(f"  peak eta   {labels[0]}: {st['peak_c'].max():+.3f} m max, "
          f"{np.median(st['peak_c']):+.3f} median")
    print(f"  peak eta   {labels[1]}: {st['peak_f'].max():+.3f} m max, "
          f"{np.median(st['peak_f']):+.3f} median")
    print(f"  peak diff  median {np.median(st['dpeak']):+.4f} m, "
          f"mean {st['dpeak'].mean():+.4f} m, max |.| {np.abs(st['dpeak']).max():.4f} m")
    print(f"  series     RMSE median {np.median(st['rmse']):.4f} m, "
          f"worst gauge {st['rmse'].max():.4f} m")
    for thr in (0.02, 0.05, 0.10):
        print(f"  gauges with |peak diff| > {thr:.2f} m: "
              f"{int((np.abs(st['dpeak']) > thr).sum())} of {len(st)}")

    noaa = compare({k: v for k, v in c_all.items() if k >= 9000},
                   {k: v for k, v in f_all.items() if k >= 9000})
    for r in noaa:
        print(f"  NOAA {r['gid']}: peak {r['peak_c']:+.3f} -> {r['peak_f']:+.3f} m "
              f"({r['dpeak']:+.4f}), series RMSE {r['rmse']:.4f} m")

    fig, ax = plt.subplots(1, 3, figsize=(16, 6.4),
                           gridspec_kw={"width_ratios": [1, 1, 1.25]})

    # --- A. peak along the front ----------------------------------------------------------
    ax[0].plot(st["peak_c"], st["lat"], lw=1.6, color=PALETTE["muted"], label=labels[0])
    ax[0].plot(st["peak_f"], st["lat"], lw=1.2, color=PALETTE["flood"], label=labels[1])
    ax[0].set_xlabel("peak eta (m)", fontsize=9)
    ax[0].set_ylabel("latitude", fontsize=9)
    ax[0].set_title("A  Peak along the coupling front", fontsize=11, color=PALETTE["text"])
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.25)

    # --- B. the difference ------------------------------------------------------------------
    ax[1].axvline(0, color=PALETTE["muted"], lw=0.8)
    ax[1].plot(st["dpeak"], st["lat"], lw=1.2, color=PALETTE["intervention"])
    ax[1].fill_betweenx(st["lat"], 0, st["dpeak"], color=PALETTE["intervention"], alpha=0.25)
    ax[1].set_xlabel(f"peak difference, {labels[1]} minus {labels[0]} (m)", fontsize=9)
    ax[1].set_title("B  What the child grid would see", fontsize=11, color=PALETTE["text"])
    ax[1].grid(alpha=0.25)

    # --- C. series at the gauges that differ most -------------------------------------------
    worst = st[np.argsort(-np.abs(st["dpeak"]))][:n_show]
    for k, r in enumerate(worst):
        gid = int(r["gid"])
        _, tc, ec = coup_c[gid]
        _, tf, ef = coup_f[gid]
        off = k * 0.6
        ax[2].plot(tc / 3600.0, ec + off, lw=1.5, color=PALETTE["muted"],
                   label=labels[0] if k == 0 else None)
        ax[2].plot(tf / 3600.0, ef + off, lw=1.1, color=PALETTE["flood"],
                   label=labels[1] if k == 0 else None)
        ax[2].text(tc.min() / 3600.0, off + 0.08,
                   f"gauge {gid}, lat {r['lat']:.3f}, dpeak {r['dpeak']:+.3f} m",
                   fontsize=7.5, color=PALETTE["text"])
    ax[2].axvline(0, color=PALETTE["intervention"], lw=1.0, ls="--")
    ax[2].set_xlabel("hours from landfall", fontsize=9)
    ax[2].set_ylabel("eta (m), series offset for clarity", fontsize=9)
    ax[2].set_title(f"C  The {n_show} gauges that disagree most", fontsize=11,
                    color=PALETTE["text"])
    ax[2].legend(fontsize=8, frameon=False, loc="upper left")
    ax[2].grid(alpha=0.25)

    for a in ax:
        a.tick_params(labelsize=7.5)
    fig.subplots_adjust(wspace=0.28, top=0.86)
    fig.suptitle(f"Boundary convergence: {labels[1]} against {labels[0]}",
                 fontsize=14, y=0.965, color=PALETTE["text"])
    fig.text(0.5, 0.005,
             "Panel B is the result. The 4 m nest never reads GeoClaw, so the only way "
             "refinement reaches Pin Point is through these gauge series. A difference small "
             "against the high-water-mark residual is a difference the chain cannot carry.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse", required=True, help="_output of the lower-level run")
    ap.add_argument("--fine", required=True, help="_output of the higher-level run")
    ap.add_argument("--labels", nargs=2, default=["level 6 (145 m)", "level 7 (36 m)"])
    ap.add_argument("--n-show", type=int, default=3)
    ap.add_argument("--csv", default=None, help="also write the per-gauge table")
    ap.add_argument("--out", default="reports/figures/amr_convergence.png")
    a = ap.parse_args()
    st = build(a.coarse, a.fine, a.out, labels=tuple(a.labels), n_show=a.n_show)
    if a.csv:
        np.savetxt(a.csv, np.column_stack([st[n] for n in st.dtype.names]),
                   delimiter=",", header=",".join(st.dtype.names), comments="")
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
