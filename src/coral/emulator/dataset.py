"""Emulator data contract: LISFLOOD runs -> (input channels, depth target) tensors.

The forward emulator learns:  (static maps + scalar forcing) -> 2-D max-depth field.

Interventions need no special encoding — a seawall/wetland/retreat IS an edit to the
DEM / Manning / infiltration grids, so the static channels already carry it. That is
why the same channel stack covers baseline *and* intervention scenarios.

Input channels (aligned to the DEM grid, land-masked):
  0  DEM elevation (m)
  1  Manning's n
  2  infiltration Ksat (mm/hr)      [0 if absent]
  3  infiltration capacity AWC (mm) [0 if absent]
  4  distance-to-coast proxy (DEM<=sea_level -> 0)   (cheap connectivity hint)
  5.. scalar forcings broadcast to constant planes (surge peak, rain total, SLR, ...)

Target: max flood depth (m) from LISFLOOD .max, sea cells and dry cells -> 0.

A dataset is described by a manifest (list of samples); each sample points at a run
dir + its scalar forcings. See build_manifest() for the helper.

Deps: numpy, torch (optional extra: pip install -e ".[emulator]").
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


def read_asc(path):
    """Read an ESRI ASCII grid -> (array with nodata=nan, header dict)."""
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
        a = np.loadtxt(f)
    nod = h.get("nodata_value", -9999)
    return np.where(a == nod, np.nan, a), h


@dataclass
class FloodSample:
    """One LISFLOOD run = one training example."""
    name: str
    dem: str                      # SUB_DEM ascii
    manning: str                  # Manning ascii
    maxfile: str                  # results .max (target)
    infil: str | None = None      # Ksat ascii (optional)
    infilcap: str | None = None   # AWC ascii (optional)
    forcing: dict = field(default_factory=dict)   # scalar knobs -> constant planes
    sea_level: float = 0.81


# scalar forcings become constant channels, in this fixed order (missing -> 0)
SCALAR_KEYS = ("surge_peak_m", "rain_total_mm", "slr_m", "infil_capped")


def sample_to_arrays(s: FloodSample):
    """Return (X [C,H,W] float32, y [H,W] float32, mask [H,W] bool)."""
    dem, _ = read_asc(s.dem)
    man, _ = read_asc(s.manning)
    ksat = read_asc(s.infil)[0] if s.infil else np.zeros_like(dem)
    awc = read_asc(s.infilcap)[0] if s.infilcap else np.zeros_like(dem)
    tgt, _ = read_asc(s.maxfile)

    land = np.isfinite(dem) & (dem > s.sea_level)
    dist = np.where(dem <= s.sea_level, 0.0, 1.0)          # crude coast proxy
    depth = np.where(land & np.isfinite(tgt), np.clip(tgt - 0.0, 0, None), 0.0)
    depth = np.nan_to_num(depth)

    planes = [np.nan_to_num(dem), np.nan_to_num(man),
              np.nan_to_num(ksat), np.nan_to_num(awc), dist]
    for k in SCALAR_KEYS:
        planes.append(np.full_like(dem, float(s.forcing.get(k, 0.0))))
    X = np.stack(planes).astype("float32")
    return X, depth.astype("float32"), land


class FloodDataset:
    """torch Dataset over FloodSamples. Channel-wise standardization from `stats`
    (fit on the train split, reused on val/test)."""

    def __init__(self, samples, stats=None):
        import torch  # noqa: F401  (fail early if extra not installed)
        self.samples = list(samples)
        self.cache = [sample_to_arrays(s) for s in self.samples]
        self.stats = stats or self._fit_stats()

    def _fit_stats(self):
        xs = np.stack([x for x, _, _ in self.cache])          # [N,C,H,W]
        m = xs.mean(axis=(0, 2, 3)); sd = xs.std(axis=(0, 2, 3)) + 1e-6
        return {"mean": m.astype("float32"), "std": sd.astype("float32")}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import torch
        X, y, mask = self.cache[i]
        m, sd = self.stats["mean"][:, None, None], self.stats["std"][:, None, None]
        Xn = (X - m) / sd
        return (torch.from_numpy(Xn),
                torch.from_numpy(y)[None],           # [1,H,W]
                torch.from_numpy(mask)[None])        # [1,H,W]


def partition(samples, is_test):
    """Split FloodSamples into (train, test) by a predicate on the sample — e.g.
    hold out an unseen SLR or intervention type to test GENERALIZATION, not fit.
        train, test = partition(samples, lambda s: s.forcing.get('slr_m') == 1.5)
    """
    tr = [s for s in samples if not is_test(s)]
    te = [s for s in samples if is_test(s)]
    return tr, te


def make_datasets(train_samples, test_samples=None):
    """Build (train_ds, test_ds); test reuses the TRAIN normalization stats so the
    held-out set isn't peeked at when standardizing."""
    tr = FloodDataset(train_samples)
    te = FloodDataset(test_samples, stats=tr.stats) if test_samples else None
    return tr, te


def build_manifest(runs):
    """Turn a list of dicts (run_dir, name, forcing, ...) into FloodSamples.

    Each dict: {name, run_dir, root='res_matthew_sav', forcing={...},
                infil?, infilcap?}. Static maps are looked up inside run_dir.
    """
    out = []
    for r in runs:
        d = Path(r["run_dir"]); root = r.get("root", "res_matthew_sav")
        out.append(FloodSample(
            name=r["name"],
            dem=str(next(d.glob("SUB_DEM*.asc"))),
            manning=str(next(d.glob("Manning*.asc"))),
            maxfile=str(d / f"results_matthew_sav/{root}.max"),
            infil=str(next(iter(d.glob("infil_*.asc")), "")) or None,
            infilcap=str(next(iter(d.glob("infilcap_*.asc")), "")) or None,
            forcing=r.get("forcing", {}),
        ))
    return out
