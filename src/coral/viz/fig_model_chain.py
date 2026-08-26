"""Draw the CORAL physics-to-emulator modelling chain."""
from pathlib import Path
import argparse


def _legacy_build(out):
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


def build(out):
    """Current chapter scope; no multi-storm capability or drainage implied."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(xlim=(0, 12), ylim=(0, 7)); ax.axis('off')
    centers = [1.9, 6, 10.1]
    colors = ['#e8f1f3', '#e8efe6', '#faeedb']
    for x, title in zip(centers, ['(a) Fixed event and landscape',
                                '(b) Linked physics models', '(c) Scenario emulation']):
        ax.text(x, 6.7, title, ha='center', fontsize=12, weight='bold')
    def box(col, y, title, body):
        x = centers[col]
        ax.add_patch(FancyBboxPatch((x-1.7, y), 3.4, 1.35, boxstyle='round,pad=.04',
                     facecolor=colors[col], edgecolor='#53636d', linewidth=.9))
        ax.text(x, y+1.06, title, ha='center', va='center', fontsize=11, weight='bold')
        ax.text(x, y+.51, body, ha='center', va='center', fontsize=10, linespacing=1.35)
    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13,
                                    linewidth=1.4, color='#52636d'))
    box(0, 4.8, 'Hurricane Matthew (2016)', 'Track, wind and pressure\nRegional topobathymetry')
    box(0, 2.7, 'Inland forcing', 'Astronomical tide + datum\nHourly AORC rainfall')
    box(0, .6, 'Landscape fields', 'Terrain and Manning roughness\nInfiltration rate and storage')
    box(1, 4.8, 'GeoClaw', 'Adaptive refinement: level 6\nDistributed surge residuals')
    box(1, 2.7, '30 m LISFLOOD-FP', 'Stage boundary + rainfall\nRegional estuary routing')
    box(1, .6, '4 m Pin Point nest', 'Nested forcing + physical edits\nPhysics peak-depth labels')
    box(2, 4.8, 'Scenario design', 'Sea-level offsets and placements\nSingle edits and combinations')
    box(2, 2.7, 'U-Net training', 'Nine raster input channels\nOne peak-depth output field')
    box(2, .6, 'Conditional evaluation', 'Member, family, portfolio holdouts\nForcing and design stress tests')
    for p, q in [((3.65,5.47),(4.23,5.47)), ((6,4.75),(6,4.10)),
                 ((3.65,3.37),(4.23,3.37)), ((6,2.65),(6,2.0)),
                 ((3.65,1.27),(4.23,1.27)), ((7.73,1.3),(8.37,2.9)),
                 ((10.1,2.65),(10.1,2.0))]:
        arrow(p,q)
    # Route design to the fine-grid experiment without crossing the training arrow.
    ax.plot([11.85,12,12,6.9],[5.45,5.45,.25,.25],color='#a57026',lw=1.4,clip_on=False)
    ax.annotate('', xy=(6.9,.55), xytext=(6.9,.25),
                arrowprops=dict(arrowstyle='-|>',color='#a57026',lw=1.4))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for path in (Path(out), Path(out).with_suffix('.pdf')):
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/figures/model_chain.png")
    a = p.parse_args(); build(a.out)


if __name__ == "__main__":
    main()
