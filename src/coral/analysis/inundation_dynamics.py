"""How long cells stay wet, and how fast they drain, for one or two runs.

High-water marks record peak stage and nothing else, so they cannot distinguish a time-varying
tide from a static one at the same mean level: both arms reach a similar maximum. The tide shows
up in the parts of the record a mark never sees, which is where a static datum cannot compete:

  duration   hours a cell spends above a depth threshold
  recession  hours from a cell's own peak to the last time it is above the threshold
  cycling    whether a cell dries between tidal cycles, or stays wet for the whole event

Depth snapshots are read one at a time and reduced on the fly. Holding a 2.53M-cell 30 m domain
for 96 snapshots would be about a gigabyte per run, and nothing here needs the full stack.

    python -m coral.analysis.inundation_dynamics \\
        --runs "static tide:<dirA>" "time-varying tide:<dirB>" \\
        --dem <dem.asc> --saveint 1800 --out reports/figures/inundation_dynamics.png
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np

from .physics_ab import _read_grid
from ..viz.pinpoint_style import PALETTE

FRAME = re.compile(r"-(\d+)\.wd$")


def frames(results_dir):
    """[(index, path)] of the .wd snapshots, in time order."""
    out = []
    for p in Path(results_dir).glob("*-[0-9]*.wd"):
        m = FRAME.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def land_mask(dem_path, waterline):
    """Cells above the waterline: the ones whose wetting is a flood rather than the sea.

    Without this, three quarters of a 30 m estuary domain is ocean and permanently wet marsh,
    duration saturates at the window length in every run, and the metric cannot separate
    anything. The waterline is datums.mhw, not geoclaw.sea_level.
    """
    z, _ = _read_grid(dem_path)
    z = np.asarray(z, dtype=np.float32)
    return np.isfinite(z) & (z > waterline) & (z > -9000)


def reduce_run(results_dir, thresh=0.05, saveint=1800.0, mask=None):
    """Per-cell duration, peak depth, peak time, recession and dry-spell count.

    One pass, one frame in memory at a time. `recession` is the interval from a cell's own peak
    to the last snapshot it is above the threshold, so it measures how long water lingers after
    that cell crests rather than after the domain does.
    """
    fr = frames(results_dir)
    if not fr:
        raise SystemExit(f"no .wd snapshots in {results_dir}")
    h0, hdr = _read_grid(fr[0][1])
    shape = np.asarray(h0).shape

    n_wet = np.zeros(shape, np.int32)
    peak = np.full(shape, -np.inf, np.float32)
    peak_i = np.zeros(shape, np.int32)
    last_i = np.full(shape, -1, np.int32)
    transitions = np.zeros(shape, np.int32)      # dry -> wet crossings
    was_wet = np.zeros(shape, bool)

    for k, (idx, p) in enumerate(fr):
        h = np.asarray(_read_grid(p)[0], dtype=np.float32)
        h = np.where(np.isfinite(h), h, 0.0)
        wet = h > thresh
        n_wet += wet
        transitions += (wet & ~was_wet)
        was_wet = wet
        newpeak = h > peak
        peak_i = np.where(newpeak, k, peak_i)
        peak = np.where(newpeak, h, peak)
        last_i = np.where(wet, k, last_i)

    if mask is not None:
        if mask.shape != shape:
            raise SystemExit(f"mask {mask.shape} does not match the grid {shape}")
        n_wet = np.where(mask, n_wet, 0)
        last_i = np.where(mask, last_i, -1)
        transitions = np.where(mask, transitions, 0)
    duration_h = n_wet * saveint / 3600.0
    recession_h = np.where(last_i >= 0, (last_i - peak_i) * saveint / 3600.0, np.nan)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return dict(duration_h=duration_h, peak=peak, recession_h=recession_h,
                transitions=transitions, n_frames=len(fr), header=hdr)


def summarise(name, r, thresh):
    ever = r["duration_h"] > 0
    n = int(ever.sum())
    d = r["duration_h"][ever]
    rec = r["recession_h"][ever]
    rec = rec[np.isfinite(rec)]
    tr = r["transitions"][ever]
    print(f"{name}")
    print(f"  frames {r['n_frames']}, cells ever wet above {thresh} m: {n:,}")
    print(f"  duration  median {np.median(d):6.2f} h   mean {d.mean():6.2f} h   "
          f"p90 {np.percentile(d, 90):6.2f} h")
    print(f"  recession median {np.median(rec):6.2f} h   p90 {np.percentile(rec, 90):6.2f} h")
    print(f"  cells wetting more than once (tidal cycling): {int((tr > 1).sum()):,} "
          f"({100 * (tr > 1).mean():.1f}% of ever-wet)")
    full_h = r["n_frames"] * (d.max() / r["n_frames"]) if r["n_frames"] else 0.0
    print(f"  cells wet in every frame (never drain): "
          f"{int((d >= full_h - 1e-9).sum()):,}")
    return dict(n=n, dur_med=float(np.median(d)), rec_med=float(np.median(rec)),
                cycling=float((tr > 1).mean()))


def build(runs, out, *, dem=None, thresh=0.05, saveint=1800.0, waterline=None):
    """runs is [(label, results_dir), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = None
    if dem and waterline is not None:
        mask = land_mask(dem, waterline)
        print(f"land mask: {int(mask.sum()):,} of {mask.size:,} cells above {waterline} m "
              f"({100 * mask.mean():.1f}%)\n")

    red, stats = {}, {}
    for label, d in runs:
        red[label] = reduce_run(d, thresh=thresh, saveint=saveint, mask=mask)
        stats[label] = summarise(label, red[label], thresh)
        print()

    labels = list(red)
    ncol = 3 if len(labels) >= 2 else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.6 * ncol, 5.4))
    colours = [PALETTE["muted"], PALETTE["flood"], PALETTE["intervention"]]

    # --- A. duration distribution --------------------------------------------------------
    for k, lab in enumerate(labels):
        d = red[lab]["duration_h"]
        d = d[d > 0]
        ax[0].hist(d, bins=60, histtype="step", lw=1.6, color=colours[k % 3], label=lab)
    ax[0].set_xlabel(f"hours above {thresh} m", fontsize=9)
    ax[0].set_ylabel("cells", fontsize=9)
    ax[0].set_title("A  How long cells stay wet", fontsize=11, color=PALETTE["text"])
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.25)

    # --- B. recession ---------------------------------------------------------------------
    for k, lab in enumerate(labels):
        r = red[lab]["recession_h"]
        r = r[np.isfinite(r) & (red[lab]["duration_h"] > 0)]
        xs = np.sort(r)
        ax[1].plot(xs, np.linspace(0, 100, xs.size), lw=1.8, color=colours[k % 3], label=lab)
    ax[1].set_xlabel("hours from a cell's own peak to last wet", fontsize=9)
    ax[1].set_ylabel("percent of wet cells at or below", fontsize=9)
    ax[1].set_title("B  How fast the water leaves", fontsize=11, color=PALETTE["text"])
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.25)

    # --- C. where they differ --------------------------------------------------------------
    if len(labels) >= 2:
        a, b = red[labels[0]], red[labels[1]]
        diff = b["duration_h"] - a["duration_h"]
        m = (a["duration_h"] > 0) | (b["duration_h"] > 0)
        v = np.nanpercentile(np.abs(diff[m]), 98) if m.any() else 1.0
        im = ax[2].imshow(np.where(m, diff, np.nan), cmap="RdBu_r", vmin=-v, vmax=v)
        ax[2].set_xticks([]); ax[2].set_yticks([])
        ax[2].set_title(f"C  Duration, {labels[1]} minus {labels[0]}",
                        fontsize=11, color=PALETTE["text"])
        fig.colorbar(im, ax=ax[2], fraction=0.045, pad=0.02).set_label("hours", fontsize=8)
        print(f"duration difference, {labels[1]} minus {labels[0]}: "
              f"median {np.nanmedian(diff[m]):+.2f} h, "
              f"mean {np.nanmean(diff[m]):+.2f} h, "
              f"p5 {np.nanpercentile(diff[m], 5):+.2f} h, "
              f"p95 {np.nanpercentile(diff[m], 95):+.2f} h")

    for a_ in ax:
        a_.tick_params(labelsize=7.5)
    fig.subplots_adjust(wspace=0.26, top=0.86)
    fig.suptitle("Inundation duration and recession", fontsize=14, y=0.965,
                 color=PALETTE["text"])
    fig.text(0.5, -0.02,
             "High-water marks record peak stage only, so they cannot separate a time-varying "
             "tide from a static one at the same mean level. These are the quantities that can.",
             ha="center", fontsize=8.6, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, metavar="LABEL:RESULTS_DIR")
    ap.add_argument("--dem", default=None)
    ap.add_argument("--thresh", type=float, default=0.05,
                    help="depth counted as wet; matches LISFLOOD's DepthThresh")
    ap.add_argument("--saveint", type=float, default=1800.0)
    ap.add_argument("--waterline", type=float, default=None,
                    help="with --dem, restrict to cells above this elevation. Without it the "
                         "domain is mostly ocean, duration saturates at the window length in "
                         "every run, and nothing separates. Use datums.mhw (1.114).")
    ap.add_argument("--out", default="reports/figures/inundation_dynamics.png")
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        if ":" not in spec:
            raise SystemExit(f"--runs wants LABEL:RESULTS_DIR, got {spec!r}")
        lab, d = spec.split(":", 1)
        runs.append((lab, d))
    build(runs, a.out, dem=a.dem, thresh=a.thresh, saveint=a.saveint,
          waterline=a.waterline)


if __name__ == "__main__":
    main()
