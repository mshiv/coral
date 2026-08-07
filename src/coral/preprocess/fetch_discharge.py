"""USGS river discharge on the model clock, for a fluvial inflow boundary.

Completes the compound forcing set: surge, tide, rainfall, river. LISFLOOD takes a time-varying
inflow as a point source, `P <lon> <lat> QVAR <name>` in the .bci with a matching .bdy block, and
adds it as depth each timestep (iterateq.cpp):

    dh = q(t) * dx * dt / dA

so q is discharge PER UNIT WIDTH, m2/s. Gauge discharge is divided by the inflow width,
which is the width of the channel where it enters the domain. Getting this wrong scales the river
by the width in metres, which can produce silent and large errors.

Gauge records for Matthew show why this is : the Ogeechee at
Eden rises from 180 to 3,720 cfs but peaks on 12 October, three to four days after landfall, and
contributes 2-5% of boundary flux inside the simulated window. It matters for the recession and
duration.
"""
import argparse
import json
import urllib.request

import numpy as np
from datetime import datetime, timezone

CFS_TO_CMS = 0.0283168
NWIS = "https://waterservices.usgs.gov/nwis/iv/"        # instantaneous values


def fetch(site, start, end, param="00060"):
    """Return (times_utc, values_cfs) for a USGS gauge. Falls back to daily if 15-min is absent."""
    for service, tag in ((NWIS, "iv"), (NWIS.replace("/iv/", "/dv/"), "dv")):
        url = (f"{service}?sites={site}&startDT={start}&endDT={end}"
               f"&parameterCd={param}&format=json")
        try:
            d = json.load(urllib.request.urlopen(url, timeout=90))
            ts = d["value"]["timeSeries"]
            if not ts:
                continue
            vals = ts[0]["values"][0]["value"]
            tt = [datetime.fromisoformat(v["dateTime"]).astimezone(timezone.utc) for v in vals]
            vv = [float(v["value"]) for v in vals]
            keep = [(a, b) for a, b in zip(tt, vv) if b >= 0]
            if keep:
                print(f"  {site}: {len(keep)} {tag} samples, {min(b for _, b in keep):.0f}"
                      f"-{max(b for _, b in keep):.0f} cfs")
                return [a for a, _ in keep], [b for _, b in keep]
        except Exception as e:
            print(f"  {site} ({tag}): {e}")
    raise SystemExit(f"no discharge for {site}")


def to_model_clock(times, values, t0_utc, width_m, tstart_s=0.0):
    """Convert to (model_seconds, q_m2s). `t0_utc` is model time zero.

    Samples before t=0 are dropped, with one interpolated sample placed at t=0 so the
    series still covers the run start. LISFLOOD initialises the previous time to -1 s
    (`input.cpp:1860`), so a block whose first sample is negative fails the monotonic
    check on its first row and the run exits before the first timestep. A gauge record
    fetched around an event routinely starts before model time zero.
    """
    t0 = datetime.fromisoformat(t0_utc).replace(tzinfo=timezone.utc)
    secs = [(t - t0).total_seconds() + tstart_s for t in times]
    q = [v * CFS_TO_CMS / width_m for v in values]

    if secs and secs[0] < 0.0:
        # Drop everything at or before t=0 and prepend a single interpolated sample there, so the
        # series starts exactly at 0 and stays strictly increasing even when a sample lands on 0.
        n_drop = sum(1 for s in secs if s <= 0.0)
        if n_drop == len(secs):
            raise SystemExit("the whole discharge record predates model time zero; "
                             "check --t0 and the fetch window")
        q0 = float(np.interp(0.0, secs, q))
        secs, q = [0.0] + secs[n_drop:], [q0] + q[n_drop:]
        print(f"  dropped {n_drop} samples at or before t=0, "
              f"interpolated q={q0:.6f} m2/s at t=0")
    return secs, q


def write_bdy_block(path, name, secs, q, mode="a"):
    """Append one QVAR block. LISFLOOD reads the first line as a comment, so a new file gets one."""
    new = mode == "w"
    with open(path, mode) as f:
        if new:
            f.write("comment\n")
        f.write(f"{name}\n{len(secs)}\t\tseconds\n")
        for s, v in zip(secs, q):
            f.write(f"{v:.6f}\t{s:.1f}\t\n")
    print(f"  wrote block {name!r}: {len(secs)} samples, "
          f"{min(q):.4f}-{max(q):.4f} m2/s -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="USGS gauge id, e.g. 02202500 Ogeechee nr Eden")
    ap.add_argument("--name", required=True, help="boundary name used in the .bci")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--t0", required=True, help="model time zero, ISO UTC, e.g. 2016-10-06T00:00:00")
    ap.add_argument("--width-m", type=float, required=True,
                    help="inflow width; discharge is divided by this to give m2/s")
    ap.add_argument("--tstart", type=float, default=0.0, help="model clock offset (par tstart)")
    ap.add_argument("--out-bdy", required=True)
    ap.add_argument("--new", action="store_true", help="start a new .bdy instead of appending")
    a = ap.parse_args()
    tt, vv = fetch(a.site, a.start, a.end)
    secs, q = to_model_clock(tt, vv, a.t0, a.width_m, a.tstart)
    write_bdy_block(a.out_bdy, a.name, secs, q, mode="w" if a.new else "a")
    print(f"  add to the .bci:  P\t<lon>\t<lat>\tQVAR\t{a.name}")


if __name__ == "__main__":
    main()
