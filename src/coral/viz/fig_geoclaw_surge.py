"""Figure 7 — the GeoClaw surge that drives the boundary, in the same visual language.

The LISFLOOD domain is forced at its edge by a water level that comes from a much larger
GeoClaw run. That step is usually invisible in the write-up: a boundary file appears with no
picture of where it came from. This draws the parent run, frame by frame, with the child domain
marked on it, so the handoff between the two models is something a reader can see.

GeoClaw writes ASCII `fort.q` files, one per output frame, each holding every AMR patch in the
grid hierarchy. Patches are drawn coarsest first so finer levels land on top.

    python -m coral.viz.fig_geoclaw_surge --output <geoclaw>/_output \\
        --child-bbox -81.22 -80.82 31.82 32.08 --out reports/figs/fig7_geoclaw.png
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np

from .pinpoint_style import PALETTE

FORT_PULASKI = (-80.9017, 32.0367)


def read_fort_t(path):
    """(time_s, meqn, ngrids) from a fort.t header file."""
    txt = Path(path).read_text().split()
    return float(txt[0]), int(txt[2]), int(txt[4])


def read_fort_q(path, meqn=4):
    """Every AMR patch in one frame.

    Returns a list of dicts with level, extent and a (my, mx, meqn) array. GeoClaw writes the
    cells row by row from the bottom of each patch, so the array is stored origin-lower.
    """
    lines = Path(path).read_text().splitlines()
    patches, i, n = [], 0, len(lines)
    while i < n:
        if "grid_number" not in lines[i]:
            i += 1
            continue
        level = int(lines[i + 1].split()[0])
        mx = int(lines[i + 2].split()[0]); my = int(lines[i + 3].split()[0])
        xlow = float(lines[i + 4].split()[0]); ylow = float(lines[i + 5].split()[0])
        dx = float(lines[i + 6].split()[0]); dy = float(lines[i + 7].split()[0])
        i += 8
        vals, got = [], 0
        while i < n and got < mx * my:
            f = lines[i].split()
            if len(f) >= meqn:
                vals.append([float(v) for v in f[:meqn]]); got += 1
            i += 1
        if got < mx * my:
            break
        patches.append(dict(level=level, mx=mx, my=my,
                            ext=(xlow, xlow + mx * dx, ylow, ylow + my * dy),
                            q=np.array(vals).reshape(my, mx, meqn)))
    return patches


def build(output, out, *, n_frames=6, child_bbox=None, dry=0.01, vmax=1.5,
          bbox=None, label="GeoClaw parent run", sea_level=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    O = Path(output)
    qs = sorted(O.glob("fort.q[0-9]*"))
    if not qs:
        raise SystemExit(f"no fort.q frames in {O}")
    pick = np.linspace(0, len(qs) - 1, min(n_frames, len(qs))).round().astype(int)
    frames = [qs[i] for i in sorted(set(pick))]

    ncol = len(frames)
    fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 4.6), squeeze=False)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for k, qp in enumerate(frames):
        ax = axes[0][k]
        tp = O / ("fort.t" + re.search(r"fort\.q(\d+)", qp.name).group(1))
        t_s, meqn, _ = read_fort_t(tp) if tp.exists() else (np.nan, 4, 0)
        patches = read_fort_q(qp, meqn=meqn)
        # Coarse first, fine on top: the AMR hierarchy refines where the surge is.
        for p in sorted(patches, key=lambda d: d["level"]):
            h = p["q"][:, :, 0]
            eta = p["q"][:, :, meqn - 1]
            # Anomaly against the run's still-water level. This run was configured with
            # sea_level = 0.81, so raw eta puts the whole ocean at +0.81 and the surge
            # disappears into the background.
            field = np.where(h > dry, eta - sea_level, np.nan)
            ax.imshow(field, extent=p["ext"], origin="lower", cmap="RdBu_r", norm=norm,
                      interpolation="nearest")
        if child_bbox:
            w, e, s, nn = child_bbox
            ax.add_patch(Rectangle((w, s), e - w, nn - s, fill=False,
                                   edgecolor="#111111", lw=1.8, zorder=6))
        ax.plot(*FORT_PULASKI, marker="^", ms=5, color=PALETTE["text"], zorder=7)
        if bbox:
            ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for s_ in ax.spines.values():
            s_.set_edgecolor(PALETTE["muted"]); s_.set_linewidth(0.5)
        ax.set_title(f"t = {t_s / 3600.0:+.0f} h", fontsize=10, color=PALETTE["text"], pad=6)
        ax.text(0.03, 0.03, f"{len(patches)} AMR patches", transform=ax.transAxes,
                fontsize=7, color=PALETTE["text"],
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))

    im = axes[0][-1].images[-1]
    cax = fig.add_axes([0.92, 0.25, 0.008, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("sea surface anomaly (m)\nrelative to the run still-water level",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(wspace=0.05, right=0.9, top=0.86)
    fig.suptitle(f"{label} — the surge that sets the boundary condition",
                 fontsize=14, y=0.97, color=PALETTE["text"])
    fig.text(0.5, -0.02,
             "Time is relative to landfall. The black box is the LISFLOOD domain, the triangle "
             "is Fort Pulaski. Refinement follows the surge, so the patch count is itself a "
             "record of where the parent model spent its effort.",
             ha="center", fontsize=8.4, color=PALETTE["muted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({ncol} of {len(qs)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="GeoClaw _output directory")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--child-bbox", nargs=4, type=float, default=None,
                    metavar=("W", "E", "S", "N"))
    ap.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("W", "E", "S", "N"))
    ap.add_argument("--vmax", type=float, default=1.5)
    ap.add_argument("--sea-level", type=float, default=0.0,
                    help="still-water level of the run; subtracted to give anomaly")
    ap.add_argument("--label", default="GeoClaw parent run")
    ap.add_argument("--out", default="reports/figs/fig7_geoclaw.png")
    a = ap.parse_args()
    build(a.output, a.out, n_frames=a.n_frames, child_bbox=a.child_bbox,
          bbox=a.bbox, vmax=a.vmax, label=a.label, sea_level=a.sea_level)


if __name__ == "__main__":
    main()
