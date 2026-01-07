from __future__ import annotations
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 수정: mps 추가
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, extra: Optional[Dict[str, Any]] = None) -> None:
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))

def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(str(path), map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt

def center_crop_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
   
    _, _, H, W = x.shape
    _, _, Hr, Wr = ref.shape
    if H == Hr and W == Wr:
        return x
    top = (H - Hr) // 2
    left = (W - Wr) // 2
    return x[:, :, top:top + Hr, left:left + Wr]

def make_bilinear_upsample_weight(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
    """
    Create weights for ConvTranspose2d to perform bilinear upsampling (FCN paper style init).
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = torch.arange(kernel_size).float()
    filt = (1 - torch.abs(og - center) / factor)
    w2d = filt[:, None] * filt[None, :]  # [k,k]

    weight = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
    for i in range(min(in_channels, out_channels)):
        weight[i, i] = w2d
    return weight

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best is None:
            self.best = value
            return False

        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
