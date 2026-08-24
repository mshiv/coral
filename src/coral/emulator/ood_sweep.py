"""Build an out-of-distribution test ensemble: the same design, knobs pushed past their ranges.

The production ensemble reports a held-out error of about a centimetre, and that number is easy
to over-read. Every member shares one storm, one domain and one set of static fields, so the
network is interpolating among near-neighbours of a single event. Holding out the highest sea
level already showed what leaving that neighbourhood costs: sixty times the error, with no
improvement after the first epoch.

This probes the other edge. Members are generated exactly as the production ensemble generates
them, then each continuous knob is pushed beyond the range it was sampled from -- a wall taller
than any trained on, a roughness denser, an area fraction larger. The physics runs normally,
because the parent model has no notion of a sampled range. Only the emulator does.

Scoring the existing checkpoints against these members without retraining answers the question
the claims table currently cannot: how far outside its training envelope does the emulator remain
usable, and does its error grow gracefully or collapse.

Knobs are pushed multiplicatively above the registry maximum, except elevation freeboards, which
are pushed additively so the result stays a plausible structure rather than an implausible one.

Usage:
  python -m coral.emulator.ood_sweep --config configs/scenarios/pp4_e01.yaml \
      --base <pp4_base> --out <runs>/pp4_ood --factor 1.5 --n-per-kind 4
"""
from __future__ import annotations

import argparse
import copy

# Knobs pushed by ADDING to the registry maximum rather than scaling it. A freeboard of 3.7 m
# scaled by 1.5 is 5.6 m, which is a sea wall rather than a floodwall; adding 1.5 m gives a tall
# but recognisable structure, which is the regime worth testing.
ADDITIVE = {"crest_above_water_m", "sill_m"}
# Fractions cannot exceed 1. Pushing them scales toward that ceiling instead.
FRACTIONS = {"area_frac"}
SKIP = {"kind", "seed", "wall_slot", "alignment_index"}


def push(knobs, registry, factor):
    """Move every continuous knob past the top of its sampled range.

    Returns the edited knobs and a record of what moved, so the manifest carries how far outside
    the training envelope each member sits rather than only that it is outside.
    """
    out, moved = copy.deepcopy(knobs), {}
    rng = registry.get(knobs["kind"], {})
    for k, v in list(out.items()):
        if k in SKIP or not isinstance(v, (int, float)):
            continue
        bounds = rng.get(k)
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            continue
        hi = float(bounds[1])
        if k in ADDITIVE:
            new = hi + (factor - 1.0) * hi
        elif k in FRACTIONS:
            new = min(0.95, hi + (1.0 - hi) * (factor - 1.0))
        else:
            new = hi * factor
        out[k] = float(new)
        moved[k] = {"sampled_max": hi, "ood_value": float(new),
                    "times_max": float(new) / hi if hi else None}
    return out, moved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, default=1.5,
                    help="how far past the sampled maximum, as a multiple (1.5 = 50 percent above)")
    ap.add_argument("--n-per-kind", type=int, default=4)
    ap.add_argument("--nlcd", default=None)
    a = ap.parse_args()

    from ..config import load
    from ..interventions.generate import INTERVENTIONS
    from . import sweep as sw

    cfg = load(a.config)
    iv = cfg.interventions
    # Same design as production, fewer members: this is a probe, not a training set.
    lv = list(iv.slr_levels)
    if iv.slr_scenarios:
        from ..preprocess.fetch_slr import slr_levels as resolve
        lv += resolve([(s, y) for s, y in iv.slr_scenarios], station=cfg.forcing.tide_station)
    specs = sw.plan_sweep(lv, iv.kinds, a.n_per_kind, False, iv.seed + 991,
                          seawall_walls=iv.seawall_walls, siting=iv.siting,
                          targeted_frac=iv.targeted_frac)

    n_pushed = 0
    for s in specs:
        rec = []
        for kb in s["interventions"]:
            kb2, moved = push(kb, INTERVENTIONS, a.factor)
            kb.clear(); kb.update(kb2)
            rec.append(moved)
            n_pushed += len(moved)
        if rec:
            s["ood"] = {"factor": a.factor, "knobs": rec}
    print(f"{len(specs)} members, {n_pushed} knob values pushed past their sampled maximum "
          f"by a factor of {a.factor}")
    for s in specs[:3]:
        if s.get("ood"):
            print(f"  {s['name']}: {s['ood']['knobs'][0]}")

    return sw.from_config(cfg, a.base, a.out, nlcd=a.nlcd, specs=specs)


if __name__ == "__main__":
    main()
