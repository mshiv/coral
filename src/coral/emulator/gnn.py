"""Message-passing emulator over the flood graph.

Baseline to beat: U-Net at RMSE 0.033 m, CSI 0.973 on corrected 30 m physics, same split.

Each layer updates a node from its neighbours, weighted by edge features. Following SWE-GNN, a
hop is a finite-volume flux exchange, so depth is the number of cells information can travel and
should be set from how far water moves in the modelled window, not tuned blindly.

Edge features enter every message. That makes the model resolution-agnostic.

"""
import torch
import torch.nn as nn


def _mlp(cin, cout, hidden=None):
    h = hidden or cout
    return nn.Sequential(nn.Linear(cin, h), nn.SiLU(), nn.Linear(h, cout))


class MessageLayer(nn.Module):
    """One flux exchange. Message from edge features and both endpoint states, summed per node."""

    def __init__(self, dim, edge_dim):
        super().__init__()
        self.msg = _mlp(2 * dim + edge_dim, dim)
        self.upd = _mlp(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_feat):
        s, d = edge_index[0], edge_index[1]
        m = self.msg(torch.cat([x[s], x[d], edge_feat], dim=-1))
        agg = torch.zeros_like(x).index_add_(0, d, m)
        # Residual: without it, depth beyond a few layers washes out node identity and the model
        # predicts a near-uniform field.
        return self.norm(x + self.upd(torch.cat([x, agg], dim=-1)))


class FloodGNN(nn.Module):
    """Node features -> peak depth.

    `boundary_dim` reserves inputs for the stage series at boundary nodes. Interior nodes get
    zeros there. This is how surge and tide enter, and it is the piece with no precedent in the
    flood-GNN literature.
    """

    def __init__(self, in_dim, edge_dim=2, hidden=64, layers=8, boundary_dim=0):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.enc = _mlp(in_dim + boundary_dim + 1, hidden)     # +1 = boundary flag
        self.edge_enc = _mlp(edge_dim, edge_dim)
        self.layers = nn.ModuleList(MessageLayer(hidden, edge_dim) for _ in range(layers))
        self.head = nn.Linear(hidden, 1)

    def forward(self, node_feat, edge_index, edge_feat, is_boundary, boundary_feat=None):
        n = node_feat.shape[0]
        flag = is_boundary.float().unsqueeze(-1)
        parts = [node_feat, flag]
        if self.boundary_dim:
            bf = boundary_feat if boundary_feat is not None else \
                node_feat.new_zeros(n, self.boundary_dim)
            parts.insert(1, bf * flag)                        # forcing only where it applies
        x = self.enc(torch.cat(parts, dim=-1))
        e = self.edge_enc(edge_feat)
        for layer in self.layers:
            x = layer(x, edge_index, e)
        # Linear head, clamped at inference only. A ReLU head gives zero gradient once a prediction
        # falls below zero and the loss plateaus -- the same failure already recorded for the U-Net.
        return self.head(x).squeeze(-1)


def masked_tail_mse(pred, target, mask, gamma=0.0):
    """MSE over valid nodes, optionally upweighting deep water by target**gamma.

    Matches the U-Net loss so the benchmark compares architectures, not objectives.
    """
    d = (pred - target) ** 2
    if gamma:
        d = d * (1.0 + target.clamp(min=0) ** gamma)
    m = mask.float()
    return (d * m).sum() / m.sum().clamp(min=1)


def mass_penalty(pred, edge_index, edge_feat, cell_area):
    """Penalise net flux imbalance implied by neighbouring predicted depths.

    A soft stand-in for the continuity check the physics satisfies to ~1e-8. Reported separately
    from skill: it constrains the field, it does not by itself make predictions accurate.
    """
    s, d = edge_index[0], edge_index[1]
    dh = pred[d] - pred[s] + edge_feat[:, 1]           # water surface difference
    flux = torch.zeros_like(pred).index_add_(0, d, dh)
    return (flux ** 2).mean() * cell_area
