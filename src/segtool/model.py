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


# Fully Convolutional Networks (FCN)
class FCN(nn.Module):
    """Simple FCN baseline using DoubleConv blocks - lightweight 2-stage architecture."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c1, c2 = base_channels, base_channels*2

        # Encoder (downsampling) - 2 stages
        self.encoder1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c2, c2*2)

        # Decoder (upsampling) - 2 stages
        self.upconv2 = nn.ConvTranspose2d(c2*2, c2, 2, 2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, 2, 2)

        # Skip connection processing
        self.skip_conv2 = nn.Conv2d(c2, c2, 1)  # pool1 output
        
        # Final classifier
        self.classifier = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.encoder1(x)      # Skip for final fusion
        pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)  # Skip for upconv2
        pool2 = self.pool2(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(pool2)

        # Decoder with skip connections (FCN-style)
        up2 = self.upconv2(bottleneck)
        skip2 = self.skip_conv2(enc2)
        up2 = up2 + skip2  # Element-wise addition (FCN style)

        up1 = self.upconv1(up2)
        up1 = up1 + enc1  # Final skip connection
        return self.classifier(up1)