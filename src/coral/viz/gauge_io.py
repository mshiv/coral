"""Read GeoClaw gauge files.

One reader for every gauge figure. The gauge header carries the location, so latitude comes
from the file rather than from a separate table, and a run with a different gauge set needs no
other change.
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np

_LOC = re.compile(r"location=\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)")


def read_gauge(path):
    """(lon, lat, t_s, eta) from one gauge file. Arrays are empty when nothing parsed."""
    lon = lat = None
    t, eta = [], []
    for line in open(path):
        if line.startswith("#"):
            m = _LOC.search(line)
            if m:
                lon, lat = float(m.group(1)), float(m.group(2))
            continue
        p = line.split()
        if len(p) >= 6:
            try:
                t.append(float(p[1])); eta.append(float(p[5]))
            except ValueError:
                pass
    return lon, lat, np.array(t), np.array(eta)


def read_gauge_set(output_dir, ids):
    """Every gauge in `ids` that exists, as a list of (id, lon, lat, t, eta)."""
    out = []
    for i in ids:
        p = Path(output_dir) / f"gauge{i:05d}.txt"
        if not p.exists():
            continue
        lon, lat, t, eta = read_gauge(p)
        if lat is None or len(t) < 2:
            continue
        out.append((i, lon, lat, t, eta))
    return out
