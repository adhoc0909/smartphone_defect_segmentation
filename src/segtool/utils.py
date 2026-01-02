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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
