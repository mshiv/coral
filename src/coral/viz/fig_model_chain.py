"""Draw the CORAL physics-to-emulator modelling chain."""
from pathlib import Path
import argparse


def build(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")
    colors = {"forcing": "#E8F2EE", "physics": "#E8EDF2",
              "scenario": "#FFF4DF", "ml": "#F7E8E4"}

    def box(x, y, w, h, title, body, color):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04,rounding_size=.10",
                           fc=color, ec="#65727B", lw=.9)
        ax.add_patch(p)
        ax.text(x+.12, y+h-.20, title, va="top", fontsize=9.0, fontweight="bold",
                color="#263238")
        ax.text(x+.12, y+h-.62, body, va="top", fontsize=7.3, color="#455A64",
                linespacing=1.3)
        return p

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=11, lw=1.2, color="#607D8B"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2+.14, label, ha="center", fontsize=7,
                    color="#455A64", bbox=dict(fc="white", ec="none", pad=.4))

    box(.15, 2.35, 1.65, 1.38, "Storm forcing",
        "Matthew track\nwind + pressure\nbathymetry", colors["forcing"])
    box(2.15, 2.35, 1.85, 1.38, "GeoClaw",
        "Atlantic surge model\nadaptive mesh\nproduction: AMR level 6", colors["physics"])
    box(4.40, 2.35, 2.00, 1.38, "Coastal boundary",
        "439 surge gauges\n+ Fort Pulaski tide\n+ datum/clock transform", colors["forcing"])
    box(6.80, 2.35, 2.00, 1.38, "30 m LISFLOOD",
        "regional estuary\nrainfall + infiltration\noverland routing", colors["physics"])
    box(9.20, 2.35, 1.65, 1.38, "4 m Pin Point",
        "nested boundary\ncommunity terrain\npeak-depth parent", colors["physics"])
    box(9.20, .35, 1.65, 1.28, "Scenario ensemble",
        "sea level × siting\nphysical field edits\n1,928 members", colors["scenario"])
    box(11.20, .35, 1.65, 1.28, "U-Net emulator",
        "fields → peak depth\nholdout tests\nparent-model fallback", colors["ml"])

    arrow(1.80, 3.04, 2.15, 3.04)
    arrow(4.00, 3.04, 4.40, 3.04, "surge residual")
    arrow(6.40, 3.04, 6.80, 3.04, "stage + flux")
    arrow(8.80, 3.04, 9.20, 3.04, "nested stage")
    arrow(10.02, 2.35, 10.02, 1.63, "fixed event physics")
    arrow(10.85, .99, 11.20, .99, "training pairs")
    ax.add_patch(FancyArrowPatch((12.02, 1.63), (10.85, 2.35), arrowstyle="-|>",
                                connectionstyle="arc3,rad=-.22", mutation_scale=11,
                                lw=1.0, ls="--", color="#D55E00"))
    ax.text(11.75, 2.08, "out-of-envelope\nreturn to physics", ha="center", fontsize=7,
            color="#D55E00")
    ax.text(.15, .62, "Fixed in this chapter", fontsize=8.5, fontweight="bold", color="#37474F")
    ax.text(.15, .30, "Hurricane Matthew, domains, terrain, rainfall product, tide treatment",
            fontsize=7.8, color="#546E7A")
    ax.set_title("CORAL modelling chain: information moves one way; evidence does not",
                 fontsize=14, fontweight="bold", pad=8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/figures/model_chain.png")
    a = p.parse_args(); build(a.out)


if __name__ == "__main__":
    main()
