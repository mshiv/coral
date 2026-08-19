"""Add open-ocean edge boundaries to a 30 m LISFLOOD run, so a tide can ebb.

A `P ... HVAR` point source hard-sets stage in one cell and passes no flux. With every domain
edge closed, the domain has no outlet: invisible under a monotonic surge, fatal under a tide.
The 2026-08-08 investigation measured Verror/Vol reaching 8 to 11 percent, 99.9 percent of the
domain wet and 77.9 m maximum depth, against 1e-16 for the tide-free run on the same grid.

The fix, configuration E1b, keeps the point sources and ADDS edge segments on the open-ocean
edges. That matters: the coupling gauges sit on a curve inside the domain, so edge-only forcing
would replace the alongshore structure GeoClaw resolves at the coast with whatever the 30 m
solver produces from an edge-imposed stage. Keeping both preserves the coupling and provides the
outlet, and it measured Verror/Vol 1e-15 throughout.

Edge conditions also require a primed initial state. In `SGC_BCs` (sgc.cpp:1354) an edge computes
flux only where the cell is already wet, so an edge added to a dry domain is inert forever. Use
`make_startfile --level` and set both `startfile` and a bare `startelev` in the .par.

The edge blocks reuse the first coupling gauge's series, which is what the working run did.

    python -m coral.couple.edge_bc --bdy <run>/<name>.bdy --bci <run>/<name>_coastline.bci \\
        --config configs/scenarios/<name>.yaml
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

EDGE_NAMES = {"E": "east", "W": "west", "N": "north", "S": "south"}


def read_block(bdy_path, name):
    """(header_line, [data lines]) for one .bdy block, or None if absent."""
    lines = Path(bdy_path).read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == name:
            count = int(lines[i + 1].split()[0])
            return lines[i + 1], lines[i + 2:i + 2 + count]
    return None


def block_names(bdy_path):
    """Every block name in the .bdy, in order. Line 1 is the file comment."""
    lines = Path(bdy_path).read_text().splitlines()
    out, i = [], 1
    while i < len(lines):
        name = lines[i].strip()
        if not name:
            i += 1
            continue
        try:
            count = int(lines[i + 1].split()[0])
        except (IndexError, ValueError):
            break
        out.append(name)
        i += 2 + count
    return out


def value_at(bdy_path, name, t_s):
    """Linear interpolation of a block's series at model time t_s."""
    blk = read_block(bdy_path, name)
    if blk is None:
        raise SystemExit(f"{bdy_path}: no block named {name}")
    v = [(float(a), float(b)) for a, b in (ln.split()[:2] for ln in blk[1])]
    v = [(t, x) for x, t in v]                      # file is "value time"
    v.sort()
    if t_s <= v[0][0]:
        return v[0][1]
    for (t0, x0), (t1, x1) in zip(v, v[1:]):
        if t0 <= t_s <= t1:
            return x0 if t1 == t0 else x0 + (x1 - x0) * (t_s - t0) / (t1 - t0)
    return v[-1][1]


def add_ocean_block(bdy_path, source="bc1", name="ocean", backup=True):
    """Append a block duplicating `source`. Idempotent."""
    if name in block_names(bdy_path):
        print(f"  .bdy already has a {name!r} block, leaving it")
        return False
    blk = read_block(bdy_path, source)
    if blk is None:
        raise SystemExit(f"{bdy_path}: no block named {source} to copy")
    if backup:
        shutil.copy2(bdy_path, str(bdy_path) + ".pre-edge")
    with open(bdy_path, "a") as f:
        f.write(f"{name}\n{blk[0]}\n" + "\n".join(blk[1]) + "\n")
    print(f"  appended {name!r} to the .bdy, copied from {source} ({len(blk[1])} samples)")
    return True


def add_edge_lines(bci_path, bbox, edges="ES", block="ocean", backup=True):
    """Append tab-separated edge segments. bbox is [W, E, S, N]. Idempotent."""
    w, e, s, n = bbox
    text = Path(bci_path).read_text()
    existing = {ln.split()[0] for ln in text.splitlines() if ln[:1] in EDGE_NAMES}
    todo = [c for c in edges if c not in existing]
    if not todo:
        print(f"  .bci already carries edges {sorted(existing)}, leaving it")
        return False
    if backup:
        shutil.copy2(bci_path, str(bci_path) + ".pre-edge")
    # An E or W edge spans a latitude range; N or S spans a longitude range.
    span = {"E": (s, n), "W": (s, n), "N": (w, e), "S": (w, e)}
    with open(bci_path, "a") as f:
        if not text.endswith("\n"):
            f.write("\n")
        for c in todo:
            a, b = span[c]
            f.write(f"{c}\t{a:.4f}\t{b:.4f}\tHVAR\t{block}\n")
            print(f"  appended {c} edge ({EDGE_NAMES[c]}) {a:.4f} to {b:.4f} -> {block}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bdy", required=True)
    ap.add_argument("--bci", required=True)
    ap.add_argument("--config", default=None, help="scenario YAML; supplies domain.bbox")
    ap.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("W", "E", "S", "N"))
    ap.add_argument("--edges", default="ES",
                    help="which edges are open ocean; the working 30 m run used ES")
    ap.add_argument("--source", default="bc1", help="block the edges reuse")
    ap.add_argument("--tstart", type=float, default=None,
                    help="report the boundary level here, for make_startfile --level")
    a = ap.parse_args()

    bbox, tstart = a.bbox, a.tstart
    if a.config:
        from ..config import load
        cfg = load(a.config)
        bbox = bbox or list(cfg.domain.bbox)
        tstart = tstart if tstart is not None else cfg.tstart
    if bbox is None:
        raise SystemExit("need --config or --bbox")

    print(f"bdy {a.bdy}")
    add_ocean_block(a.bdy, source=a.source)
    print(f"bci {a.bci}")
    add_edge_lines(a.bci, bbox, edges=a.edges, block="ocean")

    if tstart is not None:
        lv = value_at(a.bdy, a.source, tstart)
        print(f"\nboundary level at tstart {tstart:.0f} s: {lv:.3f} m")
        print("pass that to make_startfile --level, then set both in the .par:")
        print("    startfile   <the startfile>")
        print("    startelev")
        print("`startelev` is a bare flag, not an argument to a filename.")


if __name__ == "__main__":
    main()
