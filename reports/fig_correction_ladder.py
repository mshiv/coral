"""Attribution ladder: what each land-surface correction changes, at both resolutions.

base -> corr -> full, where corr adds the NWI tidal infiltration mask and the lidar canopy-bias
DEM correction, and full adds canopy-height-modulated marsh roughness. Numbers are from the
completed Matthew runs; see docs and the wiki status log for provenance.

Two results the figure is meant to carry:

  1. The corrections do not degrade agreement with observations. On the 23 quality-filtered
     high-water marks bias goes to zero and RMSE is flat; on all 28 marks RMSE falls 0.68 -> 0.65 m.
     With 0.35 m scatter over 23 points these data cannot discriminate between configurations, so
     the claim is "no cost", not "confirmed".

  2. Roughness modulation is null. full and corr are identical to three decimals in every metric
     at both resolutions, because the modulation moves marsh n only from 0.061 to 0.053-0.063 --
     far too small to matter against a 3-4 m surge.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CFG = ["base", "corr", "full"]

# 30 m, high-water marks (quality <= 3, n=23) and all marks (n=28)
HWM_Q_RMSE, HWM_Q_BIAS = [0.354, 0.355, 0.355], [-0.004, -0.000, -0.000]
HWM_A_RMSE, HWM_A_BIAS = [0.68, 0.65, 0.65], [+0.17, +0.16, +0.16]
# 4 m vs its 30 m parent, over the overlap
FINE_BIAS, FINE_AREA, FINE_R = [-0.001, +0.009, +0.009], [0.992, 1.033, 1.033], [0.9951, 0.9957, 0.9957]

x = np.arange(3)
C = {"base": "#8C8C8C", "corr": "#1F6F8B", "full": "#B4413C"}
col = [C[c] for c in CFG]

fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))

ax[0].bar(x, HWM_Q_RMSE, color=col, width=.6)
ax[0].set_ylim(0.30, 0.40); ax[0].set_ylabel("RMSE (m)")
ax[0].set_title("HWM, quality $\\leq$3 (n=23)", fontsize=10)

ax[1].bar(x, HWM_A_RMSE, color=col, width=.6)
ax[1].set_ylim(0.55, 0.75); ax[1].set_ylabel("RMSE (m)")
ax[1].set_title("HWM, all marks (n=28)", fontsize=10)

ax[2].bar(x, FINE_BIAS, color=col, width=.6)
ax[2].axhline(0, color="k", lw=.8)
ax[2].set_ylabel("WSE bias, 4 m $-$ 30 m (m)")
ax[2].set_title("4 m vs parent: bias", fontsize=10)

ax[3].bar(x, FINE_AREA, color=col, width=.6)
ax[3].axhline(1.0, color="k", lw=.8, ls=":")
ax[3].set_ylim(0.97, 1.05); ax[3].set_ylabel("flooded area ratio")
ax[3].set_title("4 m vs parent: extent", fontsize=10)

for a in ax:
    a.set_xticks(x); a.set_xticklabels(CFG)
    a.grid(axis="y", alpha=.3, lw=.5)
    a.set_axisbelow(True)

# The null result is the point of the last two panels, so state it rather than leaving the
# reader to notice two identical bars.
# Draw the bracket under the axis, where it cannot overlap a bar.
for a in (ax[2], ax[3]):
    a.annotate("", xy=(1, -0.11), xytext=(2, -0.11), xycoords=("data", "axes fraction"),
               arrowprops=dict(arrowstyle="|-|,widthA=.3,widthB=.3", lw=.9, color="#777"))
    a.annotate("identical", (1.5, -0.20), xycoords=("data", "axes fraction"),
               ha="center", fontsize=8, color="#555")

fig.suptitle("Land-surface corrections: attribution across configurations "
             "(corr = NWI infiltration mask + lidar canopy DEM; full = + canopy-modulated marsh n)",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("reports/fig_correction_ladder.png", dpi=160)
print("wrote reports/fig_correction_ladder.png")
