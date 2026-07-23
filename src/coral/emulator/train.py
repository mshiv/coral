"""Training + evaluation loop for the flood emulator (Stage 1 U-Net baseline).

Loss is tail-aware: deep cells are weighted up so the emulator doesn't under-predict
the extremes (where adaptation decisions bite — Longo 2026 / Hermans 2025). Eval
reports RMSE on wet cells and CSI (critical success index) on flood extent, the two
metrics the physics comparison cares about.

This is the wiring; a real model needs the LISFLOOD ensemble (see docs/EMULATOR_PLAN.md
Stage 0). With the current handful of same-grid runs it demonstrates the loop only.

Run: python -m coral.emulator.train  (after pip install -e ".[emulator]")
"""
from __future__ import annotations


def masked_tail_mse(pred, target, mask, tail_gamma=1.0):
    """Weighted MSE over masked (land) cells; weight = 1 + gamma*target so deeper
    water contributes more. gamma=0 -> plain masked MSE."""
    import torch
    w = (1.0 + tail_gamma * target) * mask.float()
    se = (pred - target) ** 2 * w
    return se.sum() / (w.sum() + 1e-6)


def metrics(pred, target, mask, thresh=0.10):
    """RMSE on wet land cells + CSI of flood extent at `thresh` metres."""
    import torch
    m = mask.bool()
    wet = m & (target > thresh)
    rmse = torch.sqrt(((pred - target) ** 2)[wet].mean()) if wet.any() else torch.tensor(float("nan"))
    pf, tf = (pred > thresh) & m, (target > thresh) & m
    tp = (pf & tf).sum().float(); fp = (pf & ~tf).sum().float(); fn = (~pf & tf).sum().float()
    csi = tp / (tp + fp + fn + 1e-6)
    return float(rmse), float(csi)


def train(dataset, val=None, *, epochs=200, lr=1e-3, base=32, tail_gamma=1.0,
          device=None, ckpt="emulator_unet.pt"):
    import torch
    from torch.utils.data import DataLoader
    from .models import UNet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = dataset[0][0].shape[0]
    model = UNet(in_channels=in_ch, base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(dataset, batch_size=1, shuffle=True)

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for X, y, mask in dl:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            loss = masked_tail_mse(model(X), y, mask, tail_gamma)
            loss.backward(); opt.step(); tot += loss.item()
        if ep % 20 == 0 or ep == 1:
            msg = f"ep {ep:3d}  train_loss {tot/len(dl):.4f}"
            if val is not None:
                model.eval()
                with torch.no_grad():
                    rs, cs = [], []
                    for X, y, mask in DataLoader(val, batch_size=1):
                        r, c = metrics(model(X.to(device)), y.to(device), mask.to(device))
                        rs.append(r); cs.append(c)
                import numpy as np
                msg += f"  val RMSE {np.nanmean(rs):.3f} m  CSI {np.nanmean(cs):.3f}"
            print(msg)
    torch.save({"model": model.state_dict(), "stats": dataset.stats,
                "in_channels": in_ch, "base": base}, ckpt)
    print(f"saved {ckpt}")
    return model


if __name__ == "__main__":
    # Wiring demo on the existing Matthew runs (same grid — proves the loop, not a
    # trained model). Real training: generate the ensemble first (EMULATOR_PLAN Stage 0).
    from .dataset import build_manifest, FloodDataset
    WF = "/Users/smurugan9/research/coastalFlood/savannah_matthew_workflow"
    runs = [
        {"name": "surge", "run_dir": f"{WF}/lisflood", "forcing": {"surge_peak_m": 2.3}},
        {"name": "rain", "run_dir": f"{WF}/lisflood_compound_run",
         "forcing": {"surge_peak_m": 2.3, "rain_total_mm": 257}},
        {"name": "awccap", "run_dir": f"{WF}/lisflood_compound_infilcap_run",
         "forcing": {"surge_peak_m": 2.3, "rain_total_mm": 257, "infil_capped": 1}},
    ]
    from .dataset import partition, make_datasets
    samples = build_manifest(runs)
    # hold out an unseen config as the TEST set (generalization, not fit) — here the
    # AWC-capped run; with a real ensemble, hold out unseen SLR / intervention placements.
    train_s, test_s = partition(samples, lambda s: s.name == "awccap")
    tr, te = make_datasets(train_s, test_s or train_s)
    print(f"train {len(tr)} / test {len(te)} samples, {tr[0][0].shape[0]} channels")
    train(tr, val=te, epochs=40)
