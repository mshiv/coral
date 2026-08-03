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
import os
import numpy as np


# Optional on-disk cache of decoded grids. Parsing a 1356x882 ESRI ASCII grid costs about
# 0.28 s, and a training sample reads five of them, so an epoch over 1506 members spends
# roughly 35 minutes in np.loadtxt alone and a 300-epoch run would take a week. The same grid
# reloads from .npy in 2.5 ms, 113x faster. The cache is keyed on the RESOLVED path, so the
# ensemble's symlink deduplication carries over: every member that did not edit a given grid
# shares one cache entry rather than storing its own copy.
_CACHE_DIR = None


def set_grid_cache(path):
    """Enable the decoded-grid cache. Pass None to disable."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path) if path else None
    if _CACHE_DIR:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_key(path):
    import hashlib
    rp = os.path.realpath(path)
    st = os.stat(rp)
    # size and mtime guard against a stale entry if a grid is regenerated
    sig = f"{rp}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(sig.encode()).hexdigest()[:20]


def read_asc_cached(path):
    """read_asc, backed by the .npy cache when one is enabled."""
    if _CACHE_DIR is None:
        return read_asc(path)
    f = _CACHE_DIR / f"{_cache_key(path)}.npy"
    if f.exists():
        try:
            return np.load(f, mmap_mode="r"), None
        except Exception:
            f.unlink(missing_ok=True)      # corrupt or partial write, fall through and rebuild
    a, h = read_asc(path)
    tmp = f.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, a.astype("float32"))      # atomic rename so concurrent workers cannot race
    os.replace(tmp, f)
    return a, h


def read_asc(path):
    """Read an ESRI ASCII grid -> (array with nodata=nan, header dict).

    Failures are re-raised naming the file. numpy's own message ("the number of columns
    changed from 1356 to 1045 at row 39") says nothing about which of 1506 members is at
    fault, which turns a one-line fix into a search.
    """
    h = {}
    try:
        with open(path) as f:
            for _ in range(6):
                k, v = f.readline().split()
                h[k.lower()] = float(v)
            a = np.loadtxt(f)
    except ValueError as e:
        raise ValueError(f"{path} is not a readable ESRI ASCII grid ({e}). A truncated grid "
                         "usually means the run was killed mid-write or hit a disk quota; "
                         "rerun that member.") from e
    exp = (int(h["nrows"]), int(h["ncols"]))
    if a.shape != exp:
        raise ValueError(f"{path} holds {a.shape} but its header declares {exp}; the file is "
                         "incomplete. Rerun that member.")
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
    dem, _ = read_asc_cached(s.dem)
    man, _ = read_asc_cached(s.manning)
    ksat = read_asc_cached(s.infil)[0] if s.infil else np.zeros_like(dem)
    awc = read_asc_cached(s.infilcap)[0] if s.infilcap else np.zeros_like(dem)
    tgt, _ = read_asc_cached(s.maxfile)

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
    (fit on the train split, reused on val/test).

    `lazy` controls whether the decoded arrays are held in RAM. Eager caching costs
    9 channels x H x W x 4 bytes per sample, which is 43 MB on the 1356x882 30 m grid
    and so 64 GB over a 1506-member ensemble. Anything past a few dozen members must
    run lazy, which re-reads the ASCII grids on each access instead; use DataLoader
    workers to hide that I/O. `stats_n` caps how many samples the normalization stats
    are fitted on, since the channel means are stable long before 1506 samples.
    """

    def __init__(self, samples, stats=None, lazy=True, stats_n=64, seed=0):
        import torch  # noqa: F401  (fail early if extra not installed)
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("FloodDataset got zero samples")
        self.lazy = lazy
        self.cache = None if lazy else [sample_to_arrays(s) for s in self.samples]
        self.stats = stats or self._fit_stats(stats_n, seed)

    def _fit_stats(self, stats_n, seed):
        if self.cache is not None:
            xs = np.stack([x for x, _, _ in self.cache])       # [N,C,H,W]
        else:
            rng = np.random.default_rng(seed)
            n = min(stats_n, len(self.samples))
            pick = rng.choice(len(self.samples), size=n, replace=False)
            xs = np.stack([sample_to_arrays(self.samples[i])[0] for i in pick])
        m = xs.mean(axis=(0, 2, 3)); sd = xs.std(axis=(0, 2, 3)) + 1e-6
        return {"mean": m.astype("float32"), "std": sd.astype("float32")}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import torch
        X, y, mask = self.cache[i] if self.cache is not None \
            else sample_to_arrays(self.samples[i])
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


def make_datasets(train_samples, test_samples=None, lazy=True):
    """Build (train_ds, test_ds); test reuses the TRAIN normalization stats so the
    held-out set isn't peeked at when standardizing."""
    tr = FloodDataset(train_samples, lazy=lazy)
    te = FloodDataset(test_samples, stats=tr.stats, lazy=lazy) if test_samples else None
    return tr, te


def build_manifest(runs, skip_missing=False):
    """Turn a list of dicts (run_dir, name, forcing, ...) into FloodSamples.

    Each dict: {name, run_dir, root='res_matthew_sav', forcing={...},
                infil?, infilcap?}. Static maps are looked up inside run_dir.

    `skip_missing` drops runs whose grids or `.max` are absent instead of raising, which
    is what an ensemble needs: sweep writes one manifest entry per planned member, but
    members that are still queued or that failed have no `.max` yet. Returns the samples;
    call `missing_runs` for the list of what was dropped and why.
    """
    out = []
    for r in runs:
        d = Path(r["run_dir"]); root = r.get("root", "res_matthew_sav")
        dem = next(iter(sorted(d.glob("SUB_DEM*.asc"))), None)
        man = next(iter(sorted(d.glob("Manning*.asc"))), None)
        mx = _find_max(d, root, r.get("results_dir"))
        if dem is None or man is None or not is_valid_max(mx):
            if skip_missing:
                continue
            what = ("DEM" if dem is None else "Manning grid" if man is None
                    else f"{mx} (absent, empty, or truncated)")
            raise FileNotFoundError(f"{r['name']}: missing {what} in {d}")
        out.append(FloodSample(
            name=r["name"],
            dem=str(dem),
            manning=str(man),
            maxfile=str(mx),
            infil=str(next(iter(sorted(d.glob("infil_*.asc"))), "")) or None,
            infilcap=str(next(iter(sorted(d.glob("infilcap_*.asc"))), "")) or None,
            forcing=r.get("forcing", {}),
        ))
    return out


def is_valid_max(path):
    """True if `path` looks like a complete ESRI ASCII grid.

    A member killed while writing, or one that hit a disk quota, leaves a .max that is either
    empty or truncated partway through the data block. A header check alone is not enough:
    such a file can carry a perfectly good six-line header and then stop mid-grid, which
    np.loadtxt only discovers when the column count changes at some row.

    The check is header parse plus a file-size floor computed from the header itself. Values
    are written as "0.0000 " and similar, so at least three bytes per cell is a conservative
    bound that a truncated file fails by a wide margin and a complete file passes easily. This
    is a stat, not a read, so it stays cheap across a 1506-member ensemble.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return False
        with open(p) as f:
            h = {}
            for _ in range(6):
                parts = f.readline().split()
                if len(parts) != 2:
                    return False
                h[parts[0].lower()] = float(parts[1])
        ncols, nrows = int(h["ncols"]), int(h["nrows"])
        if ncols <= 0 or nrows <= 0:
            return False
        return p.stat().st_size >= 3 * ncols * nrows
    except (OSError, ValueError, KeyError):
        return False


def _find_max(run_dir, root, results_dir=None):
    """Locate the `.max` inside a run dir. The results directory name comes from the par's
    `dirroot`, so it is `results_matthew_sav` for the 30 m ensemble but
    `results_pinpoint_highres_4m` for the 4 m one; glob rather than assume."""
    d = Path(run_dir)
    if results_dir:
        return d / results_dir / f"{root}.max"
    exact = sorted(d.glob(f"results_*/{root}.max"))
    if exact:
        return exact[0]
    any_max = sorted(d.glob("results_*/*.max"))
    return any_max[0] if any_max else d / f"results_matthew_sav/{root}.max"


def missing_runs(runs):
    """The complement of build_manifest(skip_missing=True): (name, reason) per dropped run.
    Use it to tell an incomplete ensemble apart from a broken path before training."""
    bad = []
    for r in runs:
        d = Path(r["run_dir"]); root = r.get("root", "res_matthew_sav")
        if not d.is_dir():
            bad.append((r["name"], f"run dir absent: {d}")); continue
        if not next(iter(d.glob("SUB_DEM*.asc")), None):
            bad.append((r["name"], "no SUB_DEM*.asc")); continue
        if not next(iter(d.glob("Manning*.asc")), None):
            bad.append((r["name"], "no Manning*.asc")); continue
        mx = _find_max(d, root, r.get("results_dir"))
        if not mx.exists():
            bad.append((r["name"], "no .max, member not finished")); continue
        if not is_valid_max(mx):
            bad.append((r["name"], "empty or truncated .max, member must be rerun")); continue
    return bad
