"""Stage-2 GNN flood emulator (PyTorch Geometric).

Every grid cell is a node; grid adjacency is the edges (hydraulic connectivity). Node
features carry elevation + surface properties + forcing; the network predicts per-node
flood depth via message passing. A GraphSAGE backbone — the dependency-light entry point
toward mSWE-GNN / FloodGNN-GRU (the wiki's try-order). Why a GNN over the CNN: native to
unstructured/adaptive meshes, generalizes to unseen geometry, and interventions are
literally node/edge edits. See docs/EMULATOR_PLAN.md.

Deps: torch, torch_geometric (pip install torch-geometric).
"""
from __future__ import annotations
import numpy as np


def grid_edge_index(H, W, diagonal=False):
    """4- (or 8-) connectivity edge_index (2, E) for an H x W grid; node id = r*W + c.
    Undirected (both directions included)."""
    import torch
    idx = np.arange(H * W).reshape(H, W)
    e = [np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),      # right
         np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])]      # down
    if diagonal:
        e += [np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()]),
              np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()])]
    ei = np.concatenate(e, axis=1)
    ei = np.concatenate([ei, ei[::-1]], axis=1)                    # make undirected
    return torch.from_numpy(np.ascontiguousarray(ei)).long()


class FloodGNN:
    """Factory -> an nn.Module GraphSAGE net (lazy import so importing this file doesn't
    require torch/PyG until a model is built). forward(x[N,C], edge_index) -> depth[N]."""

    def __new__(cls, in_channels=9, hidden=64, layers=3):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv

        class _GNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                c = in_channels
                for _ in range(layers):
                    self.convs.append(SAGEConv(c, hidden)); c = hidden
                self.head = nn.Linear(hidden, 1)

            def forward(self, x, edge_index):
                for conv in self.convs:
                    x = torch.relu(conv(x, edge_index))
                return self.head(x).squeeze(-1)   # linear head (clamp >=0 at inference;
                #   an output ReLU here dead-ends the gradient if init is negative)

        return _GNN()
