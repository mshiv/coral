"""Clip the domain DEM from a source DEM, driven by a scenario config.

Replaces clip_dem.sh. bbox + source DEM come from cfg.domain; no hardcoded extent.

  python -m coral.preprocess.clip_dem --config configs/scenarios/savannah_matthew_LR.yaml \
      --out runs/savannah_matthew_LR/inputs/SUB_DEM_savannah_matthew_LR.asc
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from coral import config


def clip(src_dem: str, bbox, out: str):
    """bbox = [W, E, S, N]. Writes an ESRI-ASCII clip (EPSG:4326, nodata -9999)."""
    W, E, S, N = bbox
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with rasterio.open(src_dem) as src:
        win = from_bounds(W, S, E, N, src.transform)
        a = src.read(1, window=win)
        tr = src.window_transform(win)
        nod = src.nodata if src.nodata is not None else -9999
    # write ESRI-ASCII (AAIGrid) so LISFLOOD reads it directly
    h, w = a.shape
    with rasterio.open(out, "w", driver="AAIGrid", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr,
                       nodata=nod) as dst:
        dst.write(a.astype("float32"), 1)
    print(f"wrote {out}  ({w}x{h}, bbox {W},{E},{S},{N})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="scenario YAML (provides DEM + bbox)")
    ap.add_argument("--dem", help="source DEM (overrides config)")
    ap.add_argument("--bbox", nargs=4, type=float, help="W E S N (overrides config)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dem, bbox = a.dem, a.bbox
    if a.config:
        c = config.load(a.config)
        dem = dem or c.domain.dem
        bbox = bbox or c.domain.bbox
    if not (dem and bbox):
        raise SystemExit("need --config or (--dem and --bbox)")
    clip(dem, bbox, a.out)


if __name__ == "__main__":
    main()
