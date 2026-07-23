"""Hydrological connectivity + stormwater-drain siting from a DEM (numpy/scipy only).

'Hydrologically connected to Pin Point' is NOT 'near Pin Point' — it's the flow
network: the **upstream catchment** (everything that drains TO the pour point) plus
the **downstream flow path** (where Pin Point drains to). Interventions that change
Pin Point's flooding live in this connected region, which can reach far from it. This
replaces the crude radius focus with a routing-based one.

Also finds **depressions / local minima** within the catchment — candidate sites for
stormwater drains (where water ponds / flow concentrates).

Method: D8 steepest-descent routing on the DEM (no depression fill — the domain drains
to the sea, which is nodata/low, so the flow-to-ocean structure is captured; add a
priority-flood fill for closed inland basins). For production-grade routing use
**pysheds**, **richdem**, or **WhiteboxTools** (C-accelerated, depression handling).

Deps: numpy, scipy.
"""
from __future__ import annotations
import numpy as np

# 8 D8 neighbour offsets
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def d8_receiver(dem, nodata):
    """Flat index of each cell's steepest-descent downslope neighbour (self if a
    sink/outlet). nodata cells -> -1."""
    ny, nx = dem.shape
    Z = np.where(nodata, np.inf, dem.astype("float64"))
    flat = np.arange(ny * nx).reshape(ny, nx)
    recv = flat.copy()
    best = np.zeros((ny, nx))
    Zp = np.pad(Z, 1, constant_values=np.inf)
    Ip = np.pad(flat, 1, constant_values=-1)
    for di, dj in DIRS:
        Zn = Zp[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        In = Ip[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        drop = (Z - Zn) / np.hypot(di, dj)
        b = drop > best
        recv[b] = In[b]; best[b] = drop[b]
    recv[nodata] = -1
    return recv.ravel()


def flow_accum(recv, dem, nodata):
    """Number of upstream cells draining through each cell (elevation-ordered)."""
    Z = np.where(nodata, -np.inf, dem).ravel()
    acc = np.ones(dem.size)
    for c in np.argsort(-Z):                 # high -> low
        r = recv[c]
        if r >= 0 and r != c:
            acc[r] += acc[c]
    return acc.reshape(dem.shape)


def catchment(recv, shape, pour_flat):
    """Boolean mask of all cells draining to `pour_flat` (upstream contributing area).
    Vectorized fixpoint: a cell joins the catchment once its receiver is in it."""
    valid = recv >= 0
    tgt = np.where(valid, recv, 0)          # dummy target for invalid cells
    m = np.zeros(recv.size, bool); m[pour_flat] = True
    n = 1
    while True:
        m = m | (valid & m[tgt])            # propagate one step upstream
        if m.sum() == n:
            break
        n = m.sum()
    return m.reshape(shape)


def downstream(recv, shape, start_flat, maxsteps=200000):
    """Cells along the flow path downstream of `start_flat` to its outlet."""
    path = [start_flat]; c = start_flat
    for _ in range(maxsteps):
        r = recv[c]
        if r < 0 or r == c:
            break
        c = r; path.append(c)
    m = np.zeros(recv.size, bool); m[path] = True
    return m.reshape(shape)


def snap_pourpoint(acc, rc, radius=12):
    """Snap a (row,col) to the highest-flow-accumulation cell within `radius` — puts
    the pour point on the actual drainage line, not an arbitrary hillside cell."""
    r, c = rc; ny, nx = acc.shape
    r0, r1 = max(0, r - radius), min(ny, r + radius + 1)
    c0, c1 = max(0, c - radius), min(nx, c + radius + 1)
    sub = acc[r0:r1, c0:c1]
    dr, dc = np.unravel_index(np.argmax(sub), sub.shape)
    return (r0 + dr, c0 + dc)


def connected_to(dem, nodata, rc, downstream_too=True, snap_radius=12):
    """The region hydrologically connected to a point (row,col): its upstream catchment
    (+ downstream flow path). Returns (mask, pour_rc, accumulation)."""
    recv = d8_receiver(dem, nodata)
    acc = flow_accum(recv, dem, nodata)
    pr, pc = snap_pourpoint(acc, rc, snap_radius)
    pflat = pr * dem.shape[1] + pc
    mask = catchment(recv, dem.shape, pflat)
    if downstream_too:
        mask = mask | downstream(recv, dem.shape, pflat)
    return mask, (pr, pc), acc


def hydraulic_connectivity(dem, point_rc, level, *, buffer_cells=20, sea_level=0.81):
    """Region hydraulically connected to `point` at water `level` (bathtub connectivity):
    the connected low-lying network (creeks + floodable land <= level continuous with the
    point) plus adjacent land within buffer_cells. For a tidal community like Pin Point,
    surge/tide propagates THROUGH the creek network, so this — not the D8 land catchment
    (tiny for a peninsula) — is the right 'connected to Pin Point' notion for surge. Use a
    D8 catchment on top for the pluvial (rainfall) contribution."""
    from scipy import ndimage
    low = np.isfinite(dem) & (dem <= level)
    lbl, _ = ndimage.label(low, structure=np.ones((3, 3)))
    r, c = point_rc
    if not low[r, c]:                                   # snap to nearest low/creek cell
        ij = ndimage.distance_transform_edt(~low, return_distances=False, return_indices=True)
        r, c = int(ij[0][r, c]), int(ij[1][r, c])
    comp = lbl == lbl[r, c]
    land = np.isfinite(dem) & (dem > sea_level)
    adj = ndimage.binary_dilation(comp, iterations=int(buffer_cells)) & land
    return comp | adj


def drain_candidates(dem, mask, *, footprint=7, min_relief=0.3, min_accum=None, acc=None):
    """Candidate stormwater-drain sites: local minima (ponding-prone) within `mask`,
    with local relief >= min_relief; optionally require high flow accumulation (where
    water concentrates). Returns a boolean mask of candidate cells."""
    from scipy import ndimage
    Zlo = np.where(np.isnan(dem), np.inf, dem)
    Zhi = np.where(np.isnan(dem), -np.inf, dem)
    localmin = (dem == ndimage.minimum_filter(Zlo, size=footprint)) & mask
    relief = ndimage.maximum_filter(Zhi, size=footprint * 3) - dem
    cand = localmin & (relief >= min_relief)
    if min_accum is not None and acc is not None:
        cand = cand & (acc >= min_accum)
    return cand
