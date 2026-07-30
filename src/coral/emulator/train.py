"""Training + evaluation loop for the flood emulator (Stage 1 U-Net baseline).

Loss is tail-aware: deep cells are weighted up so the emulator doesn't under-predict
the extremes (where adaptation decisions bite — Longo 2026 / Hermans 2025). Eval
reports RMSE on wet cells and CSI (critical success index) on flood extent, the two
metrics the physics comparison cares about.

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


def evaluate(model, dataset, device, thresh=0.10, workers=0):
    """Per-sample (rmse, csi) over `dataset`, in dataset order. Returns a list of dicts
    keyed by sample name so a report can show which scenarios the emulator handles worst."""
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    rows = []
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    with torch.no_grad():
        for s, (X, y, mask) in zip(dataset.samples, dl):
            r, c = metrics(model(X.to(device)), y.to(device), mask.to(device), thresh)
            rows.append({"name": s.name, "rmse_m": r, "csi": c,
                         "slr_m": float(s.forcing.get("slr_m", 0.0)),
                         "slr_label": s.forcing.get("slr_label")})
    return rows


def train(dataset, val=None, *, epochs=200, lr=1e-3, base=32, tail_gamma=1.0,
          device=None, ckpt="emulator_unet.pt", workers=0, eval_every=5,
          patience=None, history=None, log=print):
    """Fit the U-Net. `workers` sets DataLoader workers, which matter once the dataset is
    lazy: each sample re-reads several ASCII grids, so without workers the GPU waits on I/O.
    `patience` stops early when the validation RMSE has not improved for that many
    evaluations, and the best-scoring weights are what gets saved. `history`, if a list is
    passed, collects one row per evaluation so a caller can plot the learning curve, which
    is how we decide whether 50 realisations per adaptation is enough or 100 is needed."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .models import UNet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = dataset[0][0].shape[0]
    model = UNet(in_channels=in_ch, base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=workers,
                    persistent_workers=bool(workers))
    vdl = (DataLoader(val, batch_size=1, num_workers=workers,
                      persistent_workers=bool(workers)) if val is not None else None)

    def save(state_dict):
        torch.save({"model": state_dict, "stats": dataset.stats,
                    "in_channels": in_ch, "base": base}, ckpt)

    best, best_ep, stale, best_state = float("inf"), 0, 0, None
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for X, y, mask in dl:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            loss = masked_tail_mse(model(X), y, mask, tail_gamma)
            loss.backward(); opt.step(); tot += loss.item()
        tr_loss = tot / len(dl)
        if ep % eval_every == 0 or ep == 1 or ep == epochs:
            row = {"epoch": ep, "train_loss": tr_loss}
            msg = f"ep {ep:4d}  train_loss {tr_loss:.4f}"
            if vdl is not None:
                model.eval()
                with torch.no_grad():
                    rs, cs = [], []
                    for X, y, mask in vdl:
                        r, c = metrics(model(X.to(device)), y.to(device), mask.to(device))
                        rs.append(r); cs.append(c)
                vr, vc = float(np.nanmean(rs)), float(np.nanmean(cs))
                row.update({"val_rmse_m": vr, "val_csi": vc})
                msg += f"  val RMSE {vr:.3f} m  CSI {vc:.3f}"
                if vr < best:
                    best, best_ep, stale = vr, ep, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    save(best_state)
                    msg += "  *best, saved"
                else:
                    stale += 1
            if history is not None:
                history.append(row)
            log(msg)
            if patience and vdl is not None and stale >= patience:
                log(f"early stop at ep {ep}: no val improvement for {stale} evaluations")
                break
    if vdl is None:                       # nothing to select on, so the last state is it
        save(model.state_dict()); best_ep = epochs
    elif best_state is not None:
        # restore the best weights from memory rather than re-reading the checkpoint: the
        # checkpoint carries numpy normalization stats, so torch.load would need
        # weights_only=False, and there is no reason to round-trip through disk here.
        model.load_state_dict(best_state)
    log(f"saved {ckpt} (best val RMSE {best:.3f} m at ep {best_ep})" if val is not None
        else f"saved {ckpt}")
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
