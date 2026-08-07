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
import pathlib
import re
import numpy as np

# Model time zero for the Matthew runs, UTC.
#
# The LISFLOOD clock is shifted relative to GeoClaw's, so neither the scenario's landfall_utc
# (2016-10-08T06:00, GeoClaw t=0) nor the par's tstart gives it directly. This value was measured:
# cross-correlating the GeoClaw boundary surge against the observed non-tidal residual at Fort
# Pulaski (8670870) peaks at r = 0.885 for this epoch, putting the modelled surge maximum at
# 2016-10-08 06:55 UTC.
#
# Anything converting between model seconds and real time needs it: tide superposition, boundary
# extension, river hydrographs, and any comparison against observations.
MODEL_T0_UTC = "2016-10-06T13:00:00"


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


# ---------------------------------------------------------------- window extension
def _coops(product, begin, end, station="8670870"):
    """NOAA CO-OPS series. product is 'predictions' (astronomical tide) or 'water_level'."""
    import json, urllib.request
    from datetime import datetime, timezone
    u = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
         f"product={product}&application=CORAL&begin_date={begin}&end_date={end}"
         f"&datum=NAVD&station={station}&time_zone=gmt&units=metric&format=json")
    d = json.load(urllib.request.urlopen(u, timeout=90))
    rows = d["predictions"] if product == "predictions" else d["data"]
    t = [datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc) for r in rows]
    v = [float(r["v"]) for r in rows if r["v"] not in ("", None)]
    return np.array([x.timestamp() for x in t[:len(v)]]), np.array(v)


def extend_bdy_observed(bdy_in, bdy_out, *, t_end_s, t0_utc=MODEL_T0_UTC, station="8670870",
                        begin=None, end=None):
    """Extend every block to `t_end_s` using observed water level at a tide gauge.

    GeoClaw is barotropic and stops at the end of its run, so the boundary returns to its static
    datum. Observations do not: after Matthew the non-tidal residual stayed 20-58 cm above normal
    until 14 October (PHAWL, Park et al. 2024), and the tide keeps running. Extending with a held
    or decayed value would miss both.

    Each block keeps its own offset from the gauge, measured at the splice time:

        stage_i(t) = observed(t) + [stage_i(t_splice) - observed(t_splice)]

    so the alongshore structure GeoClaw provides is preserved while the temporal behaviour comes
    from the observation. The offset also means nothing is double counted: whatever the boundary
    already held at the splice is carried forward, not added again.

    t0_utc is model time zero as an ISO UTC string. begin/end default to a window around the
    extension, in CO-OPS YYYYMMDD form.
    """
    from datetime import datetime, timedelta, timezone
    t0 = datetime.fromisoformat(t0_utc).replace(tzinfo=timezone.utc)
    lines = pathlib.Path(bdy_in).read_text().splitlines()

    # splice time is the last sample of the first block
    i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    n0 = int(lines[i + 1].split()[0])
    t_splice = float(lines[i + 1 + n0].split()[1])
    if t_end_s <= t_splice:
        raise SystemExit(f"t_end_s {t_end_s} is not beyond the existing end {t_splice}")

    # Order matters: superpose the tide first, then extend. The extension carries the full tidal
    # swing from the observation, so a tide-free input would change character at the splice --
    # continuous in value, but flat before and oscillating after.
    _c = lines[i + 2:i + 2 + n0]
    _v = np.array([float(r.split()[0]) for r in _c[-int(min(n0, 4000)):]])
    if _v.size > 100 and (_v.max() - _v.min()) < 0.5:
        print("  WARNING: the input boundary looks tide-free (last-day range "
              f"{_v.max() - _v.min():.2f} m). Run build_bdy with tide= first, then extend.")

    b = begin or (t0 + timedelta(seconds=t_splice - 86400)).strftime("%Y%m%d")
    e = end or (t0 + timedelta(seconds=t_end_s + 86400)).strftime("%Y%m%d")
    ot, ov = _coops("water_level", b, e, station)

    def _obs_at(secs):
        """Observed level interpolated onto model seconds."""
        return np.interp(np.array([(t0 + timedelta(seconds=float(s))).timestamp() for s in secs]),
                         ot, ov)

    # One M2 cycle. The offset is a MEAN over the last cycle, not the value at the splice instant:
    # a single sample sits at some tidal phase, so an instantaneous offset absorbs the tide instead
    # of the spatial difference between the boundary point and the gauge, and biases the extension.
    cyc = 12.42 * 3600.0

    out, i, nblk, skipped = [lines[0]], 1, 0, 0
    while i < len(lines):
        name = lines[i].strip()
        if not name:
            i += 1; continue
        parts = lines[i + 1].split()
        n, unit = int(parts[0]), (parts[1] if len(parts) > 1 else "seconds")
        rows = lines[i + 2:i + 2 + n]
        rv = np.array([float(r.split()[0]) for r in rows])
        rt = np.array([float(r.split()[1]) for r in rows])

        # Splice each block at its OWN last sample. GeoClaw writes one gauge series per block on
        # the solver's timesteps, so block end times differ. Splicing every block at the first
        # block's end appends rows that predate the last sample of any longer block, and LISFLOOD
        # rejects the file with "times should be in increasing order".
        tb = float(rt[-1])
        new_secs = np.arange(tb, t_end_s + 1, 360.0)[1:]
        if new_secs.size == 0:
            out += [name, f"{n}\t\t{unit}"] + list(rows)
            i += 2 + n; nblk += 1; skipped += 1
            continue

        obs_cycle_mean = float(np.mean(_obs_at(np.arange(tb - cyc, tb, 360.0))))
        sel = rv[rt >= tb - cyc]
        bdy_cycle_mean = float(sel.mean()) if sel.size else float(rv[-1])
        off = bdy_cycle_mean - obs_cycle_mean

        rows = list(rows) + [f"{v + off:.4f}\t{s:.1f}"
                             for v, s in zip(_obs_at(new_secs), new_secs)]
        out += [name, f"{len(rows)}\t\t{unit}"] + rows
        i += 2 + n; nblk += 1

    if skipped:
        print(f"  {skipped} of {nblk} blocks already reached {t_end_s:.0f} s and were left as-is")
    obs_rest = _obs_at(np.arange(t_splice, t_end_s + 1, 360.0)[1:])

    pathlib.Path(bdy_out).write_text("\n".join(out) + "\n")
    print(f"extended {nblk} blocks from {t_splice:.0f} to {t_end_s:.0f} s using {station}; "
          f"observed {obs_rest.min():.2f}-{obs_rest.max():.2f} m -> {bdy_out}")
    return bdy_out


def trim_bdy_to_zero(bdy_in, bdy_out=None):
    """Drop samples at or before t=0 from every block, interpolating one sample at t=0.

    LISFLOOD sets the previous time to -1 s before reading a block (`input.cpp:1860`), so a
    block whose first sample is negative fails the monotonic check on its first row and the
    run exits before the first timestep. Gauge records placed on the model clock routinely
    start before model time zero.

    Values before t=0 are outside the simulated window, so trimming loses nothing. The
    interpolated sample keeps the series covering the run start.
    """
    lines = pathlib.Path(bdy_in).read_text().splitlines()
    out, i, changed = [lines[0]], 1, []
    while i < len(lines):
        name = lines[i].strip()
        if not name:
            i += 1; continue
        parts = lines[i + 1].split()
        n, unit = int(parts[0]), (parts[1] if len(parts) > 1 else "seconds")
        rows = lines[i + 2:i + 2 + n]
        t = np.array([float(r.split()[1]) for r in rows])
        if t[0] < 0.0:
            v = np.array([float(r.split()[0]) for r in rows])
            keep = int((t <= 0.0).sum())
            if keep == len(t):
                raise SystemExit(f"block '{name}' lies entirely at or before t=0")
            v0 = float(np.interp(0.0, t, v))
            rows = [f"{v0:.6f}\t0.0\t"] + rows[keep:]
            changed.append((name, keep, t[0]))
        out += [name, f"{len(rows)}\t\t{unit}"] + list(rows)
        i += 2 + n
    dest = bdy_out or bdy_in
    pathlib.Path(dest).write_text("\n".join(out) + "\n")
    for name, k, t0 in changed:
        print(f"  block {name!r}: dropped {k} samples, first was t={t0:.1f} s")
    print(f"trimmed {len(changed)} block(s) to start at t=0 -> {dest}")
    return dest


if __name__ == "__main__" and "--trim-bdy" in __import__("sys").argv:
    import argparse
    ap = argparse.ArgumentParser(description="Drop .bdy samples at or before t=0")
    ap.add_argument("--trim-bdy", required=True)
    ap.add_argument("--out", default=None, help="default: edit in place")
    a = ap.parse_args()
    trim_bdy_to_zero(a.trim_bdy, a.out)
