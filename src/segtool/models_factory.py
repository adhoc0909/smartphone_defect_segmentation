from __future__ import annotations
import torch.nn as nn
from .model import UNet, FCN, DeepLabv1

def build_model(name: str, base_channels: int = 32) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(base_channels=base_channels)
    elif name == "fcn":
        return FCN(base_channels=base_channels)
    elif name == "deeplabv1":
        return DeepLabv1(base_channels=base_channels)
    raise ValueError(f"Unknown model name: {name}")