"""Intervention effect, and whether the emulator reproduces it.

RMSE against truth measures how well the emulator reproduces a depth field. It does not measure
whether the emulator gets the *intervention* right. A model can score well on absolute depth and
still predict that a seawall makes flooding worse, because the intervention signal is small next to
the flood itself.

The quantity of interest is the change against the no-intervention baseline at the same sea level:

    delta_physics  = baseline_depth - intervention_depth      (positive = intervention helps)
    delta_emulator = baseline_pred  - intervention_pred

Skill is then agreement between those two instead of between the raw fields. Sign agreement is reported
separately, because getting the direction wrong is a different kind of failure.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..emulator.dataset import read_asc


def _depth(mx, dem, sea_level=0.81):
    d, _ = read_asc(mx)
    z, _ = read_asc(dem)
    land = np.isfinite(z) & (z > sea_level)
    return np.where(land & np.isfinite(d), np.clip(d, 0, None), 0.0), land


def delta_field(baseline_max, member_max, dem, sea_level=0.81):
    """Depth reduction relative to the baseline. Positive means the intervention reduced depth."""
    b, land = _depth(baseline_max, dem, sea_level)
    m, _ = _depth(member_max, dem, sea_level)
    return b - m, land


def score_delta(d_phys, d_emu, land, thresh=0.01):
    """Agreement on the intervention effect.

    `thresh` excludes cells where the physics effect is negligible: including them would let a
    model score well by predicting no change almost everywhere, which is true almost everywhere.
    """
    act = land & (np.abs(d_phys) > thresh)
    if not act.any():
        return {"n_active": 0}
    a, b = d_phys[act], d_emu[act]
    ss = float(((a - a.mean()) ** 2).sum())
    return {
        "n_active": int(act.sum()),
        "rmse_delta_m": float(np.sqrt(((a - b) ** 2).mean())),
        "bias_delta_m": float((b - a).mean()),
        "corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 else float("nan"),
        # Fraction of cells where the predicted effect points the same way as the physics.
        # This is the decision-relevant number: a wrong sign says an intervention helps when it
        # does not.
        "sign_agreement": float((np.sign(a) == np.sign(b)).mean()),
        # Skill against "predict no effect anywhere". Negative means the emulator is worse than
        # assuming the intervention did nothing.
        "skill_vs_null": float(1.0 - ((a - b) ** 2).sum() / ss) if ss > 0 else float("nan"),
        "mean_effect_phys_m": float(a.mean()),
        "mean_effect_emu_m": float(b.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-max", required=True, help="no-intervention .max at this sea level")
    ap.add_argument("--member-max", required=True, help="intervention member .max")
    ap.add_argument("--pred-baseline", help="emulator prediction for the baseline (.asc)")
    ap.add_argument("--pred-member", help="emulator prediction for the member (.asc)")
    ap.add_argument("--dem", required=True)
    ap.add_argument("--sea-level", type=float, default=0.81)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    dp, land = delta_field(a.baseline_max, a.member_max, a.dem, a.sea_level)
    print(f"physics effect: mean {dp[land].mean():+.4f} m, "
          f"max reduction {dp[land].max():.3f} m, cells helped "
          f"{int((dp[land] > 0.01).sum())}, cells worsened {int((dp[land] < -0.01).sum())}")

    if a.pred_baseline and a.pred_member:
        de, _ = delta_field(a.pred_baseline, a.pred_member, a.dem, a.sea_level)
        r = score_delta(dp, de, land)
        for k, v in r.items():
            print(f"  {k:22s} {v}")
        if a.out_json:
            Path(a.out_json).write_text(json.dumps(r, indent=2))
            print(f"  -> {a.out_json}")


if __name__ == "__main__":
    main()
