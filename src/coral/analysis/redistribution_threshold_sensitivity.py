"""Plot sensitivity of paired peak-depth proxies to the material-change threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLORS = {"depave": "#56B4E9", "living_shoreline": "#009E73",
          "living_shoreline_fractional": "#006D5B", "marsh_migration": "#CC79A7",
          "marsh_restoration": "#7A9A32", "road_raise": "#E69F00",
          "floodwall": "#D55E00"}
LABELS = {"depave": "De-paving", "living_shoreline": "Living shoreline (maximum)",
          "living_shoreline_fractional": "Living shoreline (fractional)",
          "marsh_migration": "Marsh migration", "marsh_restoration": "Marsh restoration",
          "road_raise": "Raised road", "floodwall": "Floodwall"}


def load(root):
    groups = {"native4m": [], "regional30m": []}
    for p in sorted(Path(root).glob("*.json")):
        d = json.loads(p.read_text())
        prefix = next((x for x in groups if p.stem.startswith(x + "_")), None)
        if prefix and d.get("threshold_sensitivity"):
            kind = p.stem[len(prefix) + 1:]
            groups[prefix].append((kind, d["threshold_sensitivity"]))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="reports/adapt/paired")
    ap.add_argument("--out", default="reports/figures/redistribution_threshold_sensitivity.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = load(a.root)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    for row, (group, title) in enumerate((("native4m", "Native 4 m diagnostics"),
                                          ("regional30m", "Regional 30 m diagnostics"))):
        for kind, rows in groups[group]:
            x = [r["threshold_m"] * 100 for r in rows]
            for col, key in enumerate(("benefit_m3", "adverse_m3")):
                y = [r[key] / 1000 for r in rows]
                if max(y, default=0) == 0:
                    continue
                axes[row, col].plot(x, y, marker="o", ms=4, lw=1.8,
                                    color=COLORS.get(kind, "#666666"),
                                    label=LABELS.get(kind, kind.replace("_", " ")))
        axes[row, 0].set_ylabel(f"{title}\npeak-depth proxy (10³ m³)", fontsize=9)

    for col, heading in enumerate(("Benefit retained", "Adverse redistribution retained")):
        axes[0, col].set_title(heading, fontweight="bold")
    for ax in axes.ravel():
        ax.set_xticks([.5, 1, 2, 5, 10])
        ax.set_xlabel("material depth-change threshold (cm)")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2)
    axes[1, 0].legend(frameon=False, fontsize=7.5, ncol=2)
    fig.suptitle("Sensitivity of integrated peak-depth changes to reporting threshold",
                 fontsize=13, fontweight="bold")
    fig.text(.5, .015, "Only land cells exceeding 0.05 m in either run are evaluated. "
             "Zero-effect interventions are omitted from plotted lines.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.11, right=.98, bottom=.11, top=.89, hspace=.30, wspace=.20)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\nwrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
