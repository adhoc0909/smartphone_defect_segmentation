from __future__ import annotations
from dataclasses import dataclasses
from pathlib import Path

@dataclass
class DataConfig:
    base_path: Path = Path('/content')
    img_size_h: int = 144
    img_size_w: int = 256
    split_ratio: float = 0.8
    seed: int = 42

@dataclass
class TriningConfig:
    data: DataConfig = DataConfig()
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 2
    threshold: float = 0.5

    out_dir: Path = Path("./outputs")
    run_name: str = "unet_baseline_256x144"
    save_every: int = 1

    model_name: str = "unet"
    base_channels: int = 32

    # wandb
    use_wandb: bool = False
    wandb_project: str = "mobile-defect-segmentation"
    wandb_entity: str | None = None
