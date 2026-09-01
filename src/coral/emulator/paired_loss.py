"""Paired intervention-response loss and diagnostics for the flood U-Net.

The production emulator predicts absolute peak depth.  This module preserves that target and
adds one controlled training signal: for an intervention member and the no-intervention member
at the same sea level, their predicted difference should match the parent-model difference.
"""
from __future__ import annotations

import math

import numpy as np

from .dataset import FloodDataset
from .train import masked_tail_mse


def forcing_key(sample):
    """Stable key linking an intervention to its matched no-intervention member."""
    label = sample.forcing.get("slr_label")
    return ("label", str(label)) if label else ("slr_m", round(float(sample.forcing.get("slr_m", 0.0)), 6))


def is_baseline(sample):
    return sample.name.endswith("_base")


def baseline_map(samples):
    """Return one completed baseline per forcing key; fail on missing or duplicate keys."""
    result = {}
    for sample in samples:
        if not is_baseline(sample):
            continue
        key = forcing_key(sample)
        if key in result:
            raise ValueError(f"duplicate baselines for {key}: {result[key].name}, {sample.name}")
        result[key] = sample
    return result


def validate_pairs(samples, baselines):
    missing = sorted({forcing_key(s) for s in samples if not is_baseline(s) and forcing_key(s) not in baselines})
    if missing:
        raise ValueError(f"no completed baseline for {len(missing)} forcing keys: {missing}")


class PairedFloodDataset:
    """Original samples plus a same-forcing baseline for the auxiliary difference loss.

    Each original member remains the primary example once. Baselines therefore retain the same
    sampling frequency as the control training; the additional baseline forward pass exists only
    to form the paired prediction difference.
    """

    def __init__(self, samples, *, stats=None, lazy=True, stats_n=64, seed=0,
                 baseline_pool=None):
        self.primary = FloodDataset(samples, stats=stats, lazy=lazy, stats_n=stats_n, seed=seed)
        pool = list(baseline_pool if baseline_pool is not None else samples)
        self.baselines = baseline_map(pool)
        validate_pairs(self.primary.samples, self.baselines)
        needed = []
        seen = set()
        for sample in self.primary.samples:
            if is_baseline(sample):
                continue
            base = self.baselines[forcing_key(sample)]
            if base.name not in seen:
                needed.append(base); seen.add(base.name)
        self.baseline_ds = FloodDataset(needed or [next(iter(self.baselines.values()))],
                                        stats=self.primary.stats, lazy=lazy)
        self.baseline_index = {s.name: i for i, s in enumerate(self.baseline_ds.samples)}
        self.samples = self.primary.samples
        self.stats = self.primary.stats

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, index):
        import torch
        current = self.primary[index]
        sample = self.primary.samples[index]
        if is_baseline(sample):
            return (*current, *current, torch.tensor(False))
        base = self.baselines[forcing_key(sample)]
        return (*current, *self.baseline_ds[self.baseline_index[base.name]], torch.tensor(True))


def paired_difference_loss(pred, target, mask, base_pred, base_target, base_mask,
                           response_threshold=0.02):
    """MSE on the full paired difference plus equal emphasis on true response cells.

    The global term penalizes invented off-site changes. The response term prevents the many
    zero-change cells from overwhelming the localized intervention signal.
    """
    import torch
    valid = mask.bool() | base_mask.bool()
    true_delta = target - base_target
    pred_delta = pred - base_pred
    global_loss = ((pred_delta - true_delta) ** 2)[valid].mean()
    response = valid & (true_delta.abs() >= response_threshold)
    if response.any():
        response_loss = ((pred_delta - true_delta) ** 2)[response].mean()
        return 0.5 * global_loss + 0.5 * response_loss
    return global_loss


def train_paired(dataset, val=None, *, delta_lambda=1.0, delta_train_threshold=0.02,
                 epochs=200, lr=1e-3, base=32, tail_gamma=1.0, device=None,
                 ckpt="emulator_unet_paired.pt", workers=0, eval_every=5,
                 patience=None, history=None, log=print, seed=0):
    """Train with the production absolute-depth loss plus a paired difference term.

    Checkpoint selection remains validation absolute-depth RMSE, matching the control run and
    preventing a favorable delta result from being purchased by degrading the hazard field.
    """
    import torch
    from torch.utils.data import DataLoader
    from .models import UNet
    from .train import metrics

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = dataset.primary[0][0].shape[0]
    model = UNet(in_channels=in_ch, base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    dl = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=workers,
                    persistent_workers=bool(workers), generator=generator)
    vdl = DataLoader(val.primary, batch_size=1, shuffle=False, num_workers=workers,
                     persistent_workers=bool(workers)) if val is not None else None

    def save(state_dict):
        torch.save({"model": state_dict, "stats": dataset.stats, "in_channels": in_ch,
                    "base": base, "objective": "absolute_plus_paired_delta",
                    "delta_lambda": delta_lambda,
                    "delta_train_threshold_m": delta_train_threshold}, ckpt)

    best, best_ep, stale, best_state = float("inf"), 0, 0, None
    for ep in range(1, epochs + 1):
        model.train(); total = total_abs = total_delta = 0.0
        for X, y, mask, X0, y0, mask0, has_pair in dl:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            pred = model(X)
            absolute = masked_tail_mse(pred, y, mask, tail_gamma)
            delta = torch.zeros((), device=device)
            if bool(has_pair.item()):
                X0, y0, mask0 = X0.to(device), y0.to(device), mask0.to(device)
                delta = paired_difference_loss(pred, y, mask, model(X0), y0, mask0,
                                               delta_train_threshold)
            loss = absolute + delta_lambda * delta
            loss.backward(); opt.step()
            total += loss.item(); total_abs += absolute.item(); total_delta += delta.item()
        n = len(dl)
        if ep % eval_every == 0 or ep == 1 or ep == epochs:
            row = {"epoch": ep, "train_loss": total / n,
                   "train_absolute_loss": total_abs / n,
                   "train_delta_loss": total_delta / n}
            msg = (f"ep {ep:4d} loss {total/n:.4f} absolute {total_abs/n:.4f} "
                   f"delta {total_delta/n:.4f}")
            if vdl is not None:
                model.eval(); rmses = []; csis = []
                with torch.no_grad():
                    for X, y, mask in vdl:
                        rmse, csi = metrics(model(X.to(device)), y.to(device), mask.to(device))
                        rmses.append(rmse); csis.append(csi)
                vr, vc = float(np.nanmean(rmses)), float(np.nanmean(csis))
                row.update({"val_rmse_m": vr, "val_csi": vc})
                msg += f" val RMSE {vr:.3f} m CSI {vc:.3f}"
                if vr < best:
                    best, best_ep, stale = vr, ep, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    save(best_state); msg += " *best, saved"
                else:
                    stale += 1
            if history is not None:
                history.append(row)
            log(msg)
            if patience and vdl is not None and stale >= patience:
                log(f"early stop at ep {ep}: no absolute-depth validation improvement")
                break
    if vdl is None:
        save(model.state_dict()); best_ep = epochs
    elif best_state is not None:
        model.load_state_dict(best_state)
    log(f"saved {ckpt} (best absolute-depth val RMSE {best:.3f} m at ep {best_ep})")
    return model


def difference_metrics(model, dataset, device, thresholds=(0.05, 0.10), workers=0,
                       cell_m=None):
    """Per-member signed response metrics on held-out intervention members."""
    import torch
    from torch.utils.data import DataLoader

    model.eval(); rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        persistent_workers=bool(workers))
    area = float(cell_m) ** 2 if cell_m else 1.0
    with torch.no_grad():
        for sample, batch in zip(dataset.samples, loader):
            X, y, mask, X0, y0, mask0, has_pair = batch
            if not bool(has_pair.item()):
                continue
            pred = model(X.to(device)); pred0 = model(X0.to(device))
            true_delta = y.to(device) - y0.to(device)
            pred_delta = pred - pred0
            valid = mask.to(device).bool() | mask0.to(device).bool()
            err = pred_delta - true_delta
            rec = {"name": sample.name, "forcing_key": str(forcing_key(sample)),
                   "delta_rmse_all_m": float(torch.sqrt((err[valid] ** 2).mean())),
                   "delta_mae_all_m": float(err[valid].abs().mean()),
                   "true_net_volume_change_m3": float(true_delta[valid].sum() * area),
                   "pred_net_volume_change_m3": float(pred_delta[valid].sum() * area)}
            for threshold in thresholds:
                affected = valid & (true_delta.abs() >= threshold)
                pred_affected = valid & (pred_delta.abs() >= threshold)
                suffix = f"{threshold:.2f}".replace(".", "p")
                if affected.any():
                    rec[f"delta_rmse_response_{suffix}_m"] = float(torch.sqrt((err[affected] ** 2).mean()))
                    rec[f"delta_mae_response_{suffix}_m"] = float(err[affected].abs().mean())
                    rec[f"sign_accuracy_{suffix}"] = float((torch.sign(pred_delta[affected]) == torch.sign(true_delta[affected])).float().mean())
                else:
                    rec[f"delta_rmse_response_{suffix}_m"] = math.nan
                    rec[f"delta_mae_response_{suffix}_m"] = math.nan
                    rec[f"sign_accuracy_{suffix}"] = math.nan
                inter = (affected & pred_affected).sum().float()
                union = (affected | pred_affected).sum().float()
                rec[f"response_iou_{suffix}"] = float(inter / union) if union else math.nan
                rec[f"n_response_cells_{suffix}"] = int(affected.sum())
            rows.append(rec)
    return rows
