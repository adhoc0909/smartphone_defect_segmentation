from __future__ import annotations
import torch.nn as nn
from .model import *

def build_model(name: str, base_channels: int = 32) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(base_channels=base_channels)
    if name == "fcn8s":
        return FCN8s(in_channels=3, num_classes=1, base_channels=base_channels)
    if name == 'deeplabv1':
        return DeepLabV1(in_channels=3, num_classes=1, base_channels=base_channels)
    raise ValueError(f"Unknown model name: {name}")