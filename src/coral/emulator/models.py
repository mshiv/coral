"""Emulator architectures. Stage 1 = a compact U-Net (image-to-image), the
CASPIAN-style CNN baseline the GNN must beat. Stage 2 (GNN, PyTorch Geometric)
lands in gnn.py once the baseline + data pipeline are proven.

All PyTorch (advisor constraint). Keep the baseline deliberately small — with a
modest LISFLOOD ensemble, a heavy net overfits.
"""
from __future__ import annotations


def _double_conv(cin, cout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet:
    """Factory returning an nn.Module U-Net (defined lazily so importing this
    module doesn't require torch until you actually build a model)."""

    def __new__(cls, in_channels=9, base=32, depth=4):
        import torch
        import torch.nn as nn

        class _UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.downs = nn.ModuleList()
                self.pools = nn.ModuleList()
                c = in_channels
                for d in range(depth):
                    w = base * (2 ** d)
                    self.downs.append(_double_conv(c, w))
                    self.pools.append(nn.MaxPool2d(2))
                    c = w
                self.bottleneck = _double_conv(c, c * 2)
                self.ups = nn.ModuleList(); self.up_convs = nn.ModuleList()
                for d in reversed(range(depth)):
                    w = base * (2 ** d)
                    self.ups.append(nn.ConvTranspose2d(c * 2 if d == depth - 1 else w * 2, w, 2, stride=2))
                    self.up_convs.append(_double_conv(w * 2, w))
                    c = w
                self.head = nn.Conv2d(base, 1, 1)

            def forward(self, x):
                # pad H,W to a multiple of 2**depth so pooling/upsampling align
                import torch.nn.functional as F
                h, w = x.shape[-2:]; m = 2 ** depth
                ph, pw = (m - h % m) % m, (m - w % m) % m
                x = F.pad(x, (0, pw, 0, ph))
                skips = []
                for conv, pool in zip(self.downs, self.pools):
                    x = conv(x); skips.append(x); x = pool(x)
                x = self.bottleneck(x)
                for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
                    x = up(x)
                    x = conv(torch.cat([x, skip], dim=1))
                x = self.head(x)
                x = F.relu(x)                       # depth >= 0
                return x[..., :h, :w]               # unpad

        return _UNet()
