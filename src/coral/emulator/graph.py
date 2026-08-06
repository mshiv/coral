"""Raster to graph conversion for the GNN emulator.

Cells become nodes, hydraulic adjacency becomes edges. The U-Net is grid-locked: its kernels
assume a fixed cell size, so a model trained at 30 m cannot run at 4 m. A graph has no such
assumption provided the edges carry physical scale.

Edge features are length and elevation difference. Without them a message-passing hop has no
physical meaning and the model cannot transfer between resolutions, which is the whole reason for
the architecture. This is the load-bearing design decision in the module.

Node features match the U-Net's per-cell channels (elevation, Manning's n, Ksat, storage) so the
two are benchmarked on identical information. Scalar forcing is attached per node the same way.

Boundary nodes carry the .bci/.bdy stage series. That is where surge and tide enter, and it is the
part with no precedent in the flood-GNN literature: SWE-GNN and its successors are fluvial or
pluvial.

Subgraph sampling exists because a 30 m grid is 1.2 M nodes and about 4.8 M edges per sample, which
does not fit in GPU memory. Samples are connected neighbourhoods, with boundary nodes oversampled
since they carry the forcing.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FloodGraph:
    """Graph form of one run. Arrays are numpy; convert to torch at the training boundary."""
    node_feat: np.ndarray      # [N, F]
    edge_index: np.ndarray     # [2, E] source, target
    edge_feat: np.ndarray      # [E, 2] length_m, elevation difference (target - source)
    target: np.ndarray         # [N] peak depth
    is_boundary: np.ndarray    # [N] bool
    node_rc: np.ndarray        # [N, 2] row, col, for rasterising predictions back
    shape: tuple               # (nrows, ncols) of the source grid


def cell_size_m(header):
    """Cell size in metres. Degrees are converted at the grid's own latitude.

    LISFLOOD grids here are lon/lat, so a degree of longitude is shorter than a degree of
    latitude and both shrink with latitude. Getting this wrong silently rescales every edge.
    """
    cs = header["cellsize"]
    if cs > 0.01:                       # already metres
        return float(cs), float(cs)
    lat = header["yllcorner"] + 0.5 * header["nrows"] * cs
    return float(cs * 111320.0 * np.cos(np.radians(lat))), float(cs * 110540.0)


def build_graph(dem, manning, ksat, awc, target, header, *, scalars=None,
                boundary_mask=None, sea_level=0.81):
    """One raster stack -> FloodGraph. Nodes are cells that can hold water.

    Permanently dry high ground is dropped: it never wets, contributes no flux, and inflating the
    graph with it costs memory that subgraph sampling then has to work around.
    """
    nr, nc = dem.shape
    valid = np.isfinite(dem)
    node_mask = valid & (dem < sea_level + 12.0)      # 12 m above datum covers any surge here
    idx = np.full((nr, nc), -1, dtype=np.int64)
    idx[node_mask] = np.arange(int(node_mask.sum()))
    rr, cc = np.nonzero(node_mask)

    feats = [dem[node_mask], manning[node_mask], ksat[node_mask], awc[node_mask]]
    for v in (scalars or {}).values():
        feats.append(np.full(rr.shape, float(v)))
    node_feat = np.stack(feats, axis=1).astype("float32")

    dx_m, dy_m = cell_size_m(header)
    src, dst, length, dz = [], [], [], []
    for (dr, dc), L in (((0, 1), dx_m), ((1, 0), dy_m)):
        a = idx[max(0, -dr):nr - dr, max(0, -dc):nc - dc]
        b = idx[dr:, dc:] if (dr or dc) else idx
        both = (a >= 0) & (b >= 0)
        ai, bi = a[both], b[both]
        # Both directions: floodplain flow is not one-way, and an undirected pair lets message
        # passing carry water either way as the head gradient dictates.
        src += [ai, bi]; dst += [bi, ai]
        z = dem[node_mask]
        length += [np.full(ai.shape, L), np.full(ai.shape, L)]
        dz += [z[bi] - z[ai], z[ai] - z[bi]]

    edge_index = np.stack([np.concatenate(src), np.concatenate(dst)]).astype("int64")
    edge_feat = np.stack([np.concatenate(length), np.concatenate(dz)], axis=1).astype("float32")

    isb = np.zeros(rr.shape, dtype=bool)
    if boundary_mask is not None:
        isb = boundary_mask[node_mask]

    return FloodGraph(node_feat=node_feat, edge_index=edge_index, edge_feat=edge_feat,
                      target=np.nan_to_num(target[node_mask]).astype("float32"),
                      is_boundary=isb, node_rc=np.stack([rr, cc], axis=1), shape=(nr, nc))


def boundary_mask_from_bci(bci_path, header):
    """Mark the cells named in a .bci as boundary nodes.

    Handles both forms: `P lon lat ...` point sources, and `N|S|E|W start end ...` edge segments,
    which cover a stretch of one domain edge rather than a single cell.
    """
    nr, nc, cs = header["nrows"], header["ncols"], header["cellsize"]
    x0, y1 = header["xllcorner"], header["yllcorner"] + nr * cs
    m = np.zeros((nr, nc), dtype=bool)
    for ln in open(bci_path):
        p = ln.split()
        if len(p) < 4:
            continue
        side = p[0].upper()
        if side == "P":
            c = int((float(p[1]) - x0) / cs); r = int((y1 - float(p[2])) / cs)
            if 0 <= r < nr and 0 <= c < nc:
                m[r, c] = True
        elif side in ("N", "S", "E", "W"):
            a, b = sorted((float(p[1]), float(p[2])))
            if side in ("N", "S"):
                c0 = max(int((a - x0) / cs), 0); c1 = min(int((b - x0) / cs) + 1, nc)
                m[0 if side == "N" else nr - 1, c0:c1] = True
            else:
                r0 = max(int((y1 - b) / cs), 0); r1 = min(int((y1 - a) / cs) + 1, nr)
                m[r0:r1, 0 if side == "W" else nc - 1] = True
    return m


def sample_subgraph(g, n_nodes, rng, boundary_frac=0.25):
    """Connected subgraph of about `n_nodes`, grown from seeds by breadth-first expansion.

    Boundary nodes are oversampled because they carry the forcing: a subgraph with none of them
    has no inflow and teaches the model nothing about the boundary condition.
    """
    N = g.node_feat.shape[0]
    if n_nodes >= N:
        return g
    nb = [[] for _ in range(N)]
    for s, d in zip(g.edge_index[0], g.edge_index[1]):
        nb[s].append(d)

    bidx = np.flatnonzero(g.is_boundary)
    nseed = max(1, n_nodes // 200)
    seeds = []
    if bidx.size:
        seeds += list(rng.choice(bidx, size=max(1, int(nseed * boundary_frac)), replace=True))
    seeds += list(rng.integers(0, N, size=nseed))

    seen = set(seeds); frontier = list(seeds)
    while frontier and len(seen) < n_nodes:
        nxt = []
        for u in frontier:
            for v in nb[u]:
                if v not in seen:
                    seen.add(v); nxt.append(v)
                    if len(seen) >= n_nodes:
                        break
            if len(seen) >= n_nodes:
                break
        frontier = nxt
    keep = np.fromiter(seen, dtype=np.int64)
    keep.sort()

    remap = np.full(N, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size)
    em = (remap[g.edge_index[0]] >= 0) & (remap[g.edge_index[1]] >= 0)
    return FloodGraph(node_feat=g.node_feat[keep],
                      edge_index=np.stack([remap[g.edge_index[0][em]], remap[g.edge_index[1][em]]]),
                      edge_feat=g.edge_feat[em], target=g.target[keep],
                      is_boundary=g.is_boundary[keep], node_rc=g.node_rc[keep], shape=g.shape)


def to_raster(g, values, fill=np.nan):
    """Scatter node values back onto the source grid, for comparison with .max output."""
    out = np.full(g.shape, fill, dtype="float32")
    out[g.node_rc[:, 0], g.node_rc[:, 1]] = values
    return out
