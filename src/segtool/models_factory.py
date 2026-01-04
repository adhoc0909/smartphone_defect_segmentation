from __future__ import annotations
import torch.nn as nn
from .model import UNet

def build_model(name: str, base_channels: int = 32) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(base_channels=base_channels)

    if name == "fcn":
        return FCN
    raise ValueError(f"Unknown model name: {name}")