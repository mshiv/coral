"""Training loop for the graph emulator.

Mirrors emulator/train.py so the two are benchmarked on the same split, loss and schedule, and any
difference is the architecture.

Graphs are built once per member and cached, since raster-to-graph is the expensive part and the
topology never changes within a member. Each epoch draws a fresh subgraph per member, which acts as
augmentation and keeps memory bounded.
"""
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import read_asc_cached
from .graph import build_graph, boundary_mask_from_bci, sample_subgraph
from .gnn import FloodGNN, masked_tail_mse


def graph_from_sample(s, bci_path=None, sea_level=0.81):
    """FloodSample -> FloodGraph, reading the same rasters the U-Net uses."""
    dem, hdr = read_asc_cached(s.dem)
    man, _ = read_asc_cached(s.manning)
    ksat = read_asc_cached(s.infil)[0] if s.infil else np.zeros_like(dem)
    awc = read_asc_cached(s.infilcap)[0] if s.infilcap else np.zeros_like(dem)
    tgt, _ = read_asc_cached(s.maxfile)
    bm = boundary_mask_from_bci(bci_path, hdr) if bci_path else None
    return build_graph(dem, man, ksat, awc, tgt, hdr, scalars=s.forcing,
                       boundary_mask=bm, sea_level=sea_level)


def _to_torch(g, device):
    return (torch.from_numpy(g.node_feat).to(device),
            torch.from_numpy(g.edge_index).to(device),
            torch.from_numpy(g.edge_feat).to(device),
            torch.from_numpy(g.is_boundary).to(device),
            torch.from_numpy(g.target).to(device))


def standardise(graphs):
    """Per-feature mean and std over a sample of graphs.

    Node features span very different magnitudes (elevation in metres, Ksat in hundreds of mm/hr),
    so without this the large-magnitude channels dominate the first layer.
    """
    X = np.concatenate([g.node_feat for g in graphs[:32]], axis=0)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    E = np.concatenate([g.edge_feat for g in graphs[:32]], axis=0)
    emu, esd = E.mean(0), E.std(0)
    esd[esd < 1e-6] = 1.0
    return mu.astype("float32"), sd.astype("float32"), emu.astype("float32"), esd.astype("float32")


def train(train_samples, val_samples, *, bci_path=None, epochs=200, lr=1e-3, hidden=64,
          layers=8, sub_nodes=60000, patience=8, eval_every=5, ckpt="gnn.pt",
          tail_gamma=1.0, device=None, seed=0, history=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    tr = [graph_from_sample(s, bci_path) for s in train_samples]
    va = [graph_from_sample(s, bci_path) for s in val_samples]
    mu, sd, emu, esd = standardise(tr)
    for g in tr + va:
        g.node_feat[:] = (g.node_feat - mu) / sd
        g.edge_feat[:] = (g.edge_feat - emu) / esd
    print(f"  graphs built in {time.time()-t0:.0f}s: {len(tr)} train, {len(va)} val, "
          f"{tr[0].node_feat.shape[1]} node feats, {tr[0].node_feat.shape[0]} nodes/member")

    model = FloodGNN(in_dim=tr[0].node_feat.shape[1], hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, bad = float("inf"), 0

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for g in tr:
            sub = sample_subgraph(g, sub_nodes, rng)
            nf, ei, ef, ib, y = _to_torch(sub, device)
            loss = masked_tail_mse(model(nf, ei, ef, ib), y, torch.ones_like(y, dtype=torch.bool),
                                   tail_gamma)
            opt.zero_grad(); loss.backward()
            # Message passing over many layers can produce large gradients on the first batches,
            # before the residual stream settles.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()

        if ep % eval_every == 0 or ep == 1:
            model.eval(); se = n = 0.0
            with torch.no_grad():
                for g in va:
                    nf, ei, ef, ib, y = _to_torch(g, device)
                    p = model(nf, ei, ef, ib).clamp(min=0)
                    se += ((p - y) ** 2).sum().item(); n += y.numel()
            rmse = (se / max(n, 1)) ** 0.5
            flag = ""
            if rmse < best - 1e-5:
                best, bad = rmse, 0
                torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                            "emu": emu, "esd": esd, "cfg": dict(hidden=hidden, layers=layers,
                                                                in_dim=tr[0].node_feat.shape[1])},
                           ckpt)
                flag = "  *best, saved"
            else:
                bad += 1
            print(f"ep {ep:4d}  train_loss {tot/len(tr):.4f}  val RMSE {rmse:.3f} m{flag}")
            if history is not None:
                history.append({"epoch": ep, "train_loss": tot / len(tr), "val_rmse": rmse})
            if patience and bad >= patience:
                print(f"early stop at ep {ep}: no val improvement for {patience} evaluations")
                break

    print(f"saved {ckpt} (best val RMSE {best:.3f} m)")
    return model, best
