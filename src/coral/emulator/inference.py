"""Emulator inference: trained model -> predicted flood-depth raster -> deliverables.

Closes the loop to the collaborator formats. The emulator's output is the same object
as a LISFLOOD `.max` (a depth array on the grid), so it drops straight into the existing
postprocess: predict -> write an ESRI-ASCII `.max`-equivalent -> 01b_make_flood_cog
(COG) / 02_make_zarr (Zarr) / 03_point_timeseries (CSV/JSON). That's the CHORUS handoff:
(intervention/TC params -> flood-depth raster), at emulator speed instead of LISFLOOD's.

Run: python -m coral.emulator.inference --ckpt emulator_unet.pt --run <run_dir> --out pred.asc
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .dataset import FloodSample, sample_to_arrays
from .models import UNet


def load_model(ckpt, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=device)
    model = UNet(in_channels=st["in_channels"], base=st["base"]).to(device)
    model.load_state_dict(st["model"]); model.eval()
    return model, st["stats"], device


def predict(model, stats, sample: FloodSample, device):
    """Return the predicted max-depth array (m) on the sample's grid, land-masked."""
    import torch
    X, _, land = sample_to_arrays(sample)
    m, sd = stats["mean"][:, None, None], stats["std"][:, None, None]
    Xn = torch.from_numpy((X - m) / sd)[None].to(device)
    with torch.no_grad():
        pred = model(Xn)[0, 0].cpu().numpy()
    return np.where(land, np.clip(pred, 0, None), np.nan)


def predict_to_asc(ckpt, sample: FloodSample, out_asc, device=None):
    """Predict and write an ESRI-ASCII depth grid (header copied from the sample DEM) so
    the existing postprocess scripts consume it exactly like a LISFLOOD `.max`."""
    model, stats, device = load_model(ckpt, device)
    depth = predict(model, stats, sample, device)
    with open(sample.dem) as f:
        hdr = [f.readline() for _ in range(6)]
    Path(out_asc).parent.mkdir(parents=True, exist_ok=True)
    with open(out_asc, "w") as f:
        f.writelines(hdr)
        np.savetxt(f, np.where(np.isnan(depth), -9999, depth), fmt="%.4f")
    print(f"predicted depth -> {out_asc} (max {np.nanmax(depth):.2f} m) — "
          f"feed to 01b_make_flood_cog / 02_make_zarr / 03_point_timeseries")
    return out_asc


if __name__ == "__main__":
    import argparse
    from .dataset import build_manifest
    ap = argparse.ArgumentParser(description="Emulator inference -> depth raster")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True, help="a run dir (provides the input grids)")
    ap.add_argument("--name", default="pred"); ap.add_argument("--out", default="pred.asc")
    ap.add_argument("--forcing", default="{}", help="JSON scalar forcings")
    a = ap.parse_args()
    import json
    s = build_manifest([{"name": a.name, "run_dir": a.run,
                         "forcing": json.loads(a.forcing)}])[0]
    predict_to_asc(a.ckpt, s, a.out)
