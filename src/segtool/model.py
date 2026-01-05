from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import center_crop_like, make_bilinear_upsample_weight

class DoubleConvValid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UNetValid(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.d1 = DoubleConvValid(in_channels, base_channels) # 64
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConvValid(base_channels, base_channels * 2) # 128
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConvValid(base_channels * 2, base_channels * 4) # 256
        self.p3 = nn.MaxPool2d(2)

        self.d4 = DoubleConvValid(base_channels * 4, base_channels * 8) # 512
        self.p4 = nn.MaxPool2d(2)

        self.b = DoubleConvValid(base_channels * 8, base_channels * 16) # 1024

        self.u4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.c4 = DoubleConvValid(base_channels * 16, base_channels * 8)

        self.u3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.c3 = DoubleConvValid(base_channels * 8, base_channels * 4)

        self.u2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.c2 = DoubleConvValid(base_channels * 4, base_channels * 2)

        self.u1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.c1 = DoubleConvValid(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        in_hw = x.shape[-2:]

        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        xb = self.b(self.p4(x4))

        y4 = self.u4(xb)
        x4c = center_crop_like(x4, y4)
        y4 = self.c4(torch.cat([x4c, y4], dim=1))

        y3 = self.u3(y4)
        x3c = center_crop_like(x3, y3)
        y3 = self.c3(torch.cat([x3c, y3], dim=1))

        y2 = self.u2(y3)
        x2c = center_crop_like(x2, y2)
        y2 = self.c2(torch.cat([x2c, y2], dim=1))

        y1 = self.u1(y2)
        x1c = center_crop_like(x1, y1)
        y1 = self.c1(torch.cat([x1c, y1], dim=1))

        logits = self.out(y1)

        if logits.shape[-2:] != in_hw:
            logits = F.interpolate(
                logits,
                size=in_hw,
                mode="bilinear",
                align_corners=False
                )


        return logits
    


class DoubleConv(nn.Module):
    """(Conv3x3(pad=1) -> ReLU) * 2  => spatial size 유지"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """U-Net (same conv version): crop 없이 skip concat 가능"""
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.d1 = DoubleConv(in_channels, base_channels)           # 64
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(base_channels, base_channels * 2)     # 128
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(base_channels * 2, base_channels * 4) # 256
        self.p3 = nn.MaxPool2d(2)

        self.d4 = DoubleConv(base_channels * 4, base_channels * 8) # 512
        self.p4 = nn.MaxPool2d(2)

        self.b  = DoubleConv(base_channels * 8, base_channels * 16) # 1024

        self.u4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.c4 = DoubleConv(base_channels * 16, base_channels * 8)  # (skip 512 + up 512) = 1024

        self.u3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.c3 = DoubleConv(base_channels * 8, base_channels * 4)   # (256 + 256) = 512

        self.u2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.c2 = DoubleConv(base_channels * 4, base_channels * 2)   # (128 + 128) = 256

        self.u1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.c1 = DoubleConv(base_channels * 2, base_channels)       # (64 + 64) = 128

        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_hw = x.shape[-2:]

        x1 = self.d1(x)               # (B, 64, H,   W)
        x2 = self.d2(self.p1(x1))      # (B,128, H/2, W/2)
        x3 = self.d3(self.p2(x2))      # (B,256, H/4, W/4)
        x4 = self.d4(self.p3(x3))      # (B,512, H/8, W/8)
        xb = self.b(self.p4(x4))       # (B,1024,H/16,W/16)

        y4 = self.u4(xb)              # (B,512, H/8, W/8)
        if x4.shape[-2:] != y4.shape[-2:]:
            x4 = F.interpolate(x4, size=y4.shape[-2:], mode="bilinear", align_corners=False)
        y4 = self.c4(torch.cat([x4, y4], dim=1))  # (B,512, H/8, W/8)

        y3 = self.u3(y4)              # (B,256, H/4, W/4)
        if x3.shape[-2:] != y3.shape[-2:]:
            x3 = F.interpolate(x3, size=y3.shape[-2:], mode="bilinear", align_corners=False)
        y3 = self.c3(torch.cat([x3, y3], dim=1))  # (B,256, H/4, W/4)

        y2 = self.u2(y3)              # (B,128, H/2, W/2)
        if x2.shape[-2:] != y2.shape[-2:]:
            x2 = F.interpolate(x2, size=y2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.c2(torch.cat([x2, y2], dim=1))  # (B,128, H/2, W/2)

        y1 = self.u1(y2)              # (B,64, H, W)
        if x1.shape[-2:] != y1.shape[-2:]:
            x1 = F.interpolate(x1, size=y1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.c1(torch.cat([x1, y1], dim=1))  # (B,64, H, W)

        logits = self.out(y1)         # (B, num_classes, H, W)

        # 혹시라도 입력이 홀수 크기라 up/down 과정에서 1픽셀 어긋나면 마지막에 입력 크기로 맞춤
        if logits.shape[-2:] != in_hw:
            logits = F.interpolate(logits, size=in_hw, mode="bilinear", align_corners=False)

        return logits
