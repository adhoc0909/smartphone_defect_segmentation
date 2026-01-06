from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class DatasetPaths:
    img_dirs: Dict[str, Path]
    mask_dirs: Dict[str, Path]

def default_paths(base: Path) -> DatasetPaths:
    img_dirs = {
        "scratch": base / "scratch",
        "oil": base / "oil",
        "stain": base / "stain",
    }
    mask_dirs = {
        "scratch": base / "ground_truth_1",
        "stain": base / "ground_truth_1",
        "oil": base / "ground_truth_2",
    }
    return DatasetPaths(img_dirs=img_dirs, mask_dirs=mask_dirs)

def _list_images(folder: Path) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    out: List[Path] = []
    for e in exts:
        out.extend([Path(p) for p in glob.glob(str(folder / e))])
    return sorted(out)


def _rand_uniform(a: float, b: float) -> float:
    return float(a + (b - a) * random.random())

def _apply_photometric_aug(img_pil: Image.Image, aug_config: dict) -> Image.Image:
    """
    Photometric-only augmentations (mask unaffected).
    aug_config:
      - augs: list[str] among {"bcg","blur","noise","specular","colorjitter"}
      - p: probability for each enabled aug (default 0.5)
      - brightness: (min,max) e.g. (0.8,1.2)
      - contrast: (min,max)
      - gamma: (min,max)
      - blur_sigma: (min,max)
      - noise_std: (min,max) in [0,1] scale (applied to float image)
      - specular: dict with strength/radius
      - colorjitter: dict with saturation/hue
    """
    augs = set([a.lower() for a in aug_config.get("augs", [])])
    if not augs:
        return img_pil

    p = float(aug_config.get("p", 0.5))

    # 1) brightness/contrast/gamma
    if "bcg" in augs and random.random() < p:
        bmin, bmax = aug_config.get("brightness", (0.8, 1.2))
        cmin, cmax = aug_config.get("contrast", (0.8, 1.2))
        gmin, gmax = aug_config.get("gamma", (0.8, 1.2))

        img_pil = ImageEnhance.Brightness(img_pil).enhance(_rand_uniform(bmin, bmax))
        img_pil = ImageEnhance.Contrast(img_pil).enhance(_rand_uniform(cmin, cmax))

        gamma = _rand_uniform(gmin, gmax)
        if abs(gamma - 1.0) > 1e-6:
            arr = np.asarray(img_pil).astype(np.float32) / 255.0
            arr = np.clip(arr, 0.0, 1.0) ** gamma
            img_pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    # 2) blur (focus shake approximation)
    if "blur" in augs and random.random() < p:
        smin, smax = aug_config.get("blur_sigma", (0.6, 1.8))
        sigma = _rand_uniform(smin, smax)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

    # Convert to float array once if needed
    arr = None

    # 3) noise
    if "noise" in augs and random.random() < p:
        nmin, nmax = aug_config.get("noise_std", (0.01, 0.05))
        std = _rand_uniform(nmin, nmax)
        arr = np.asarray(img_pil).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        img_pil = Image.fromarray((arr * 255.0).astype(np.uint8))
        arr = None  # reset

    # 4) specular highlight (strong reflection-ish)
    if "specular" in augs and random.random() < p:
        # simple: add one or two gaussian blobs
        cfg = aug_config.get("specular", {})
        strength = float(cfg.get("strength", 0.8))     # 0~1
        radius_min = int(cfg.get("radius_min", 20))
        radius_max = int(cfg.get("radius_max", 120))
        n_blobs = int(cfg.get("n_blobs", 2))

        arr = np.asarray(img_pil).astype(np.float32) / 255.0
        h, w = arr.shape[0], arr.shape[1]
        yy, xx = np.mgrid[0:h, 0:w]

        for _ in range(n_blobs):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            r = random.randint(radius_min, max(radius_min, radius_max))
            # gaussian falloff
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            blob = np.exp(-dist2 / (2.0 * (r ** 2))).astype(np.float32)
            blob = blob[..., None]  # (H,W,1)
            arr = np.clip(arr + strength * blob, 0.0, 1.0)

        img_pil = Image.fromarray((arr * 255.0).astype(np.uint8))
        arr = None

    # 5) color jitter (saturation + hue)
    if "colorjitter" in augs and random.random() < p:
        cfg = aug_config.get("colorjitter", {})
        smin, smax = cfg.get("saturation", (0.8, 1.2))
        hue = float(cfg.get("hue", 0.03))  # max hue shift in [-hue, +hue] (fraction of 1.0)
        img_pil = ImageEnhance.Color(img_pil).enhance(_rand_uniform(smin, smax))

        # Hue shift via HSV
        arr = np.asarray(img_pil).astype(np.uint8)
        hsv = Image.fromarray(arr, mode="RGB").convert("HSV")
        hsv_np = np.asarray(hsv).astype(np.uint8)
        shift = int(_rand_uniform(-hue, hue) * 255)  # HSV hue in [0,255)
        hsv_np[..., 0] = (hsv_np[..., 0].astype(int) + shift) % 256
        img_pil = Image.fromarray(hsv_np, mode="HSV").convert("RGB")

    return img_pil


def find_mask(img_path: Path, defect_type: str, mask_dirs: Dict[str, Path]) -> Optional[Path]:
    if defect_type == "good":
        return None
    fname = img_path.name
    if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg"):
        fname = os.path.splitext(fname)[0] + ".png"
    mask_path = mask_dirs[defect_type] / fname
    return mask_path if mask_path.exists() else None

class DefectSegDataset(Dataset):
    """ Binary Segmenetation을 위한 데이터셋 클래스. (img, mask, cls, img_path) 리턴"""

    def __init__(
        self,
        base_path: Path,
        split: str, # "train | val | test"
        img_size_hw: Tuple[int, int],
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
        seed: int = 42,
        aug_config: Optional[dict] = None,
    )  -> None:
        assert split in {"train", "val", "test"}
        self.base_path = base_path
        self.img_h, self.img_w = img_size_hw
        paths = default_paths(base_path)
        self.img_dirs = paths.img_dirs
        self.mask_dirs = paths.mask_dirs

        self.aug_config = aug_config or {}

        rng = np.random.default_rng(seed)

        self.samples: List[Tuple[Path, str]] = []

        for cls, folder in self.img_dirs.items():
            imgs = _list_images(folder)

            rng.shuffle(imgs)

            n_total = len(imgs)
            n_train = int(n_total * train_ratio)
            n_test = int(n_total * test_ratio)
            
            train_imgs = imgs[:n_train]
            test_imgs = imgs[n_train: n_train + n_test]
            val_imgs = imgs[n_train + n_test:]

            if split == 'train':
                chosen = train_imgs
            elif split == 'test':
                chosen = test_imgs
            else:
                chosen = val_imgs

            self.samples.extend([(p, cls) for p in chosen])

    def __len__(self) -> ing:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> np.ndarray:
        img = (
            Image.open(path).convert("RGB").resize((self.img_w, self.img_h), Image.BILINEAR)
        )
        return np.asarray(img)

    def _load_mask(self, img_path: Path, cls: str) -> np.ndarray:
        if cls == "good":
            return np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        mpath = find_mask(img_path, cls, self.mask_dirs)
        if mpath is None:
            return np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        mask = (
            Image.open(mpath)
            .convert("L")
            .resize((self.img_w, self.img_h), Image.NEAREST)
        )
        m = np.asarray(mask)
    
        return (m > 0).astype(np.uint8) * 255

    def __getitem__(self, idx: int): # !!! normalization이 없음!
        img_path, cls = self.samples[idx]

        img = self._load_rgb(img_path)
        mask = self._load_mask(img_path, cls)
        img = np.asarray(img).copy()
        mask = np.asarray(mask).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask_t = (mask_t > 0.5).float()
        

        return img_t, mask_t, cls, str(img_path)

def make_loaders(
    base_path: Path,
    img_size_hw: Tuple[int, int],
    train_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int = 2,
    aug_config: Optional[dict] = None,
):
    train_ds = DefectSegDataset(base_path, "train", img_size_hw, train_ratio, test_ratio, seed, aug_config=aug_config)
    test_ds = DefectSegDataset(base_path, "test", img_size_hw, train_ratio, test_ratio, seed, aug_config=aug_config)
    val_ds = DefectSegDataset(base_path, "val", img_size_hw, train_ratio, test_ratio, seed, aug_config=aug_config)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader