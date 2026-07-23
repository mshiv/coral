"""Build the LISFLOOD .bdy from GeoClaw coupling gauges (config-driven).

De-hardcoded version of savannah_matthew_workflow/scripts/build_bdy.py:
- drops gauges dry at their eta peak (eta = topography there, unusable)
- sorts + de-duplicates timestamps (LISFLOOD needs strictly increasing)
- shifts the time axis so it starts at 0 (LISFLOOD rejects times <= -1)
- optional datum offset (MSL -> NAVD88)
- emits a matching filtered .bci

All knobs come from the scenario config (coupling.*), not hardcoded.
"""
from __future__ import annotations
import os
import re
import numpy as np


def _coupling_ids_from_bci(bci_in):
    """The .bci is the authoritative list of boundary points: return the gauge
    ids referenced by its P-lines (tag 'bcN'). This excludes GeoClaw validation
    gauges (NOAA tide stations etc.) that exist in _output/ but aren't boundary
    conditions, and adapts per scenario without a hardcoded count."""
    ids = []
    for line in open(bci_in):
        s = line.split()
        if s and s[0] == "P":
            m = re.search(r"bc0*(\d+)$", s[-1])
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def _read_tide(path):
    """Read a make_tide series (model_seconds  level_m) -> (t, v) arrays."""
    t, v = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        s = line.split()
        if len(s) >= 2:
            t.append(float(s[0])); v.append(float(s[1]))
    return np.array(t), np.array(v)


def _read_gauge(path):
    t, h, eta = [], [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        s = line.split()
        if len(s) >= 6:
            try:
                t.append(float(s[1])); h.append(float(s[2])); eta.append(float(s[5]))
            except ValueError:
                pass
    return np.array(t), np.array(h), np.array(eta)


def build_bdy(output_dir, bci_in, bdy_out, bci_out, *,
              n_coupling=None, dry_thresh=0.05, datum_offset=0.0,
              time_offset=None, landfall_s=None,
              tide=None, surge_baseline=0.0):
    """Write <bdy_out> and a filtered <bci_out>. Returns a summary dict.

    n_coupling=None (default) takes the coupling gauge ids from the .bci P-lines
    (the authoritative boundary-point list), so validation gauges are excluded
    and the count adapts per scenario. Pass an int only to force range(1, N+1).
    landfall_s, if given, sets the time shift (toff=landfall_s) AND asserts the
    GeoClaw window is consistent with it, so the .bdy and .par clocks can't drift.

    tide (path to a make_tide series, or (t_model_s, level_m) arrays): superpose a
    time-varying tide onto the surge — the linear-superposition compound water level
    TWL(t) = tide(t) + surge_residual(t). `surge_baseline` is the STATIC sea_level
    baked into the GeoClaw surge run (geoclaw.sea_level); it is removed before adding
    the tide so the tidal datum isn't double-counted. Both must be on the same datum
    (NAVD88) and the same model clock (landfall origin). Tide is interpolated onto each
    gauge's written timestamps and applied uniformly to all boundary points (single
    station; spatial tide is a future upgrade).
    """
    if isinstance(tide, str):
        tide = _read_tide(tide)
    ids = _coupling_ids_from_bci(bci_in) if n_coupling is None \
        else list(range(1, n_coupling + 1))
    kept, dropped, blocks = [], [], {}
    for i in ids:
        p = os.path.join(output_dir, f"gauge{i:05d}.txt")
        if not os.path.exists(p):
            dropped.append((i, "missing")); continue
        t, h, eta = _read_gauge(p)
        if len(t) == 0:
            dropped.append((i, "empty")); continue
        if h[eta.argmax()] < dry_thresh:
            dropped.append((i, "dry@peak")); continue
        order = np.argsort(t, kind="stable")
        t, eta = t[order], eta[order]
        keep = np.concatenate(([True], np.diff(t) > 0))
        blocks[i] = (t[keep], (eta - datum_offset)[keep])
        kept.append(i)

    if not kept:
        raise SystemExit("no usable coupling gauges")
    gmin = min(blocks[i][0].min() for i in kept)
    # Time-shift policy (single source of truth = config.landfall_s when given):
    #   we want the .bdy clock to place landfall at landfall_s, matching the .par
    #   tstart. GeoClaw puts landfall at raw t=0, so the shift is landfall_s.
    #   Guard: that shift must also keep the series start >= 0 (LISFLOOD rejects
    #   negative times), i.e. the GeoClaw window must begin no earlier than
    #   -landfall_s. If it doesn't, the two clocks would silently disagree.
    if time_offset is not None:
        toff = time_offset
    elif landfall_s is not None:
        toff = landfall_s
        if gmin + toff < -1e-6:
            raise SystemExit(
                f"clock mismatch: GeoClaw starts at {gmin:.0f}s, but landfall_s="
                f"{landfall_s:.0f} would shift the series to start at "
                f"{gmin + toff:.0f}s (< 0). The .bdy and .par tstart would "
                f"disagree — set landfall_s to {-gmin:.0f} or extend the GeoClaw window.")
    else:
        toff = -gmin

    with open(bdy_out, "w") as f:
        f.write("comment\n")
        for i in kept:
            t, eta = blocks[i]
            tw = t + toff                                  # model-clock timestamps
            vout = eta
            if tide is not None:
                tide_t, tide_v = tide                      # both on the model clock
                vout = eta - surge_baseline + np.interp(tw, tide_t, tide_v)
            f.write(f"bc{i}\n{len(t)}\t\tseconds\n")
            for v, tt in zip(vout, tw):
                f.write(f"{v:.7g}\t{tt:.5f}\t\n")

    keep_tags = {f"bc{i}" for i in kept}
    nin = nout = 0
    with open(bci_in) as fin, open(bci_out + ".tmp", "w") as fout:
        for line in fin:
            s = line.split()
            if s and s[0] == "P":
                nin += 1
                if s[-1] in keep_tags:
                    fout.write(line); nout += 1
            else:
                fout.write(line)
    os.replace(bci_out + ".tmp", bci_out)

    return {"kept": kept, "dropped": dropped, "time_offset": toff,
            "bci_points": (nin, nout), "bdy": bdy_out, "bci": bci_out}


def build_tide_only_bdy(bci_in, tide, bdy_out, bci_out):
    """Tide-only boundary (no surge / no GeoClaw): apply a make_tide series to every
    boundary point in the .bci. Models nuisance / king-tide flooding — the bottom rung
    of the tide ladder. `tide` is a make_tide path or (t_model_s, level_m) arrays."""
    import shutil
    if isinstance(tide, str):
        tide = _read_tide(tide)
    tide_t, tide_v = tide
    ids = _coupling_ids_from_bci(bci_in)
    if not ids:
        raise SystemExit(f"no boundary points in {bci_in}")
    with open(bdy_out, "w") as f:
        f.write("comment\n")
        for i in ids:
            f.write(f"bc{i}\n{len(tide_t)}\t\tseconds\n")
            for v, tt in zip(tide_v, tide_t):
                f.write(f"{v:.7g}\t{tt:.5f}\t\n")
    shutil.copy2(bci_in, bci_out)
    return {"kept": ids, "bdy": bdy_out, "bci": bci_out, "mode": "tide-only"}


def from_config(cfg, output_dir, bci_in, bdy_out, bci_out, n_coupling=None,
                tide_file=None):
    """Adapter: pull coupling knobs from a Scenario config.

    n_coupling defaults to None -> gauges are discovered on disk. landfall_s is
    taken from the config so the .bdy time shift is the single source of truth.
    When forcing.tide is set, the tide series is fetched (make_tide) if no
    tide_file is given, and superposed with surge_baseline = geoclaw.sea_level
    (the static level the surge run used).
    """
    tide = surge_baseline = None
    if cfg.forcing.tide is not None:
        if tide_file is None:
            from ..preprocess.make_tide import from_config as _tide_from_config
            tide_file = _tide_from_config(cfg)
        tide = tide_file
        surge_baseline = cfg.geoclaw.sea_level
    return build_bdy(output_dir, bci_in, bdy_out, bci_out,
                     n_coupling=n_coupling,
                     dry_thresh=cfg.coupling.dry_thresh,
                     datum_offset=cfg.coupling.datum_offset_m,
                     landfall_s=cfg.coupling.landfall_s,
                     tide=tide, surge_baseline=surge_baseline or 0.0)
