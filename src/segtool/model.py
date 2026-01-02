from __future__ import annotations
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    """Lightweight U-Net for 256x144 sanity baseline."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels*2, base_channels*4

        self.d1 = DoubleConv(in_channels, c1)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(c1, c2)
        self.p2 = nn.MaxPool2d(2)

        self.b = DoubleConv(c2, c3)

        self.u2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.c2 = DoubleConv(c2 + c2, c2)

        self.u1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.c1 = DoubleConv(c1 + c1, c1)

        self.out = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        xb = self.b(self.p2(x2))

        x = self.u2(xb)
        x = self.c2(torch.cat([x, x2], dim=1))

        x = self.u1(x)
        x = self.c1(torch.cat([x, x1], dim=1))

        return self.out(x)
