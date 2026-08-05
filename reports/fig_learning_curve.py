"""Emulator learning curve: held-out skill against training-ensemble size.

Numbers are the held-out evaluations printed by workflows/train_ensemble.py, sea level
Int2070 held out (251 members). They are taken from the job logs rather than the report
JSONs because the n=5/10/25 sweep was killed by its 12 h wall limit during n=50, and the
report is only written after the whole schedule finishes -- so those three points never
reached a JSON. n=50 was rerun separately (mode curve50) and did write one.

An earlier version of this curve was invalid: 130/255/630/1255 members gave 0.212, 1.059,
0.260 and 0.338 m, which is BatchNorm instability at batch size 1, not a data-size effect.
GroupNorm fixed it and these are the rerun values.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_PER_KIND = [5, 10, 25, 50]
N_TRAIN    = [130, 255, 630, 1255]
RMSE_M     = [0.062, 0.057, 0.047, 0.048]
CSI        = [0.939, 0.954, 0.966, 0.964]

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
for a, y, lab, col, better in [
    (ax[0], RMSE_M, "held-out RMSE (m)", "#B4413C", "lower"),
    (ax[1], CSI,    "held-out CSI",      "#1F6F8B", "higher"),
]:
    a.plot(N_TRAIN, y, "o-", color=col, lw=1.8, ms=7)
    a.set_xscale("log")
    # Explicit ticks at the four sampled sizes. Matplotlib's default log minor labels
    # ("3 x 10^2", "4 x 10^2") overlap at this width and none of them are sampled points.
    a.set_xticks(N_TRAIN); a.set_xticklabels([str(v) for v in N_TRAIN])
    a.minorticks_off()
    a.set_xlabel("training members")
    a.set_ylabel(lab)
    a.grid(alpha=.3, which="both", lw=.5)
    for xx, yy, n in zip(N_TRAIN, y, N_PER_KIND):
        a.annotate(f"n={n}", (xx, yy), textcoords="offset points",
                   xytext=(0, 9 if better == "lower" else -14),
                   ha="center", fontsize=8, color="#555")

# The curve is flat from 630 members on: 0.047 -> 0.048 m is within run-to-run noise, so the
# ensemble is large enough and n=100 (a further ~124 GB and ~12 h) would buy nothing.
for a in ax:
    a.axvspan(630, 1400, color="#999", alpha=.10, lw=0)
ax[0].annotate("saturated", (890, 0.0595), ha="center", fontsize=8.5, color="#555")

fig.suptitle("CORAL emulator skill vs training-ensemble size (30 m, Int2070 held out)",
             fontsize=11)
fig.tight_layout()
fig.savefig("reports/fig_learning_curve.png", dpi=160)
print("wrote reports/fig_learning_curve.png")
