"""Four-panel summary for the 2026-08-06 advisor meeting.

Numbers from the completed runs; provenance in the CORAL status log. Each panel answers one
question a reviewer would ask.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

G, B, R = "#8C8C8C", "#1F6F8B", "#B4413C"
fig, ax = plt.subplots(1, 4, figsize=(15, 3.7))

# 1. emulator skill: does corrected physics change what the network can learn
ax[0].bar([0, 1], [0.047, 0.033], color=[G, B], width=.6)
ax[0].set_ylabel("held-out RMSE (m)"); ax[0].set_ylim(0, 0.055)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["baseline\nphysics", "corrected\nphysics"])
ax[0].set_title("Emulator skill, n=50", fontsize=10)
for x, v, c in ((0, 0.047, 0.966), (1, 0.033, 0.973)):
    ax[0].text(x, v + .002, f"{v:.3f}\nCSI {c:.3f}", ha="center", fontsize=8)

# 2. learning curve: is the ensemble big enough
N = [130, 255, 630, 1255]; R_ = [0.062, 0.057, 0.047, 0.048]
ax[1].plot(N, R_, "o-", color=R, lw=1.8, ms=6)
ax[1].axvspan(630, 1400, color="#999", alpha=.10, lw=0)
ax[1].set_xscale("log"); ax[1].set_xticks(N); ax[1].set_xticklabels(N)
ax[1].minorticks_off(); ax[1].set_xlabel("training members"); ax[1].set_ylabel("RMSE (m)")
ax[1].set_title("Ensemble size: saturated at 630", fontsize=10)

# 3. corrections against observations: do they cost anything
w = .35; x = np.arange(2)
ax[2].bar(x - w/2, [0.354, 0.68], w, color=G, label="baseline")
ax[2].bar(x + w/2, [0.355, 0.65], w, color=B, label="corrected")
ax[2].set_xticks(x); ax[2].set_xticklabels(["quality$\\leq$3\n(n=23)", "all marks\n(n=28)"])
ax[2].set_ylabel("HWM RMSE (m)"); ax[2].legend(fontsize=8, frameon=False)
ax[2].set_title("Corrections vs observations", fontsize=10)

# 4. 4 m against its parent, before and after the boundary fix
lab = ["broken\nboundary", "fixed", "fixed +\ncorrections"]
r = [-0.018, 0.9951, 0.9957]
ax[3].bar([0, 1, 2], r, color=[R, G, B], width=.6)
ax[3].axhline(0, color="k", lw=.8)
ax[3].set_ylabel("correlation with 30 m parent"); ax[3].set_ylim(-0.15, 1.08)
ax[3].set_xticks([0, 1, 2]); ax[3].set_xticklabels(lab, fontsize=8)
ax[3].set_title("4 m nest validation", fontsize=10)

for a in ax:
    a.grid(axis="y", alpha=.3, lw=.5); a.set_axisbelow(True)
fig.suptitle("CORAL status, 2026-08-06: corrected physics improves emulator skill 30%; "
             "4 m nest validated; ensemble size sufficient", fontsize=10.5)
fig.tight_layout()
fig.savefig("reports/fig_meeting_summary.png", dpi=160)
print("wrote reports/fig_meeting_summary.png")
