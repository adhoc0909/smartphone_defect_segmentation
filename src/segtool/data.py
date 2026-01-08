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

def default_paths(base: Path, include_augmented: bool = True) -> DatasetPaths:
    img_dirs = {
        "scratch": base / "scratch",
        "oil": base / "oil",
        "stain": base / "stain",
        "good": base / "good",  # good 클래스 추가: 제거해도 무방
    }
    mask_dirs = {
        "scratch": base / "ground_truth_1",
        "stain": base / "ground_truth_1",
        "oil": base / "ground_truth_2",
        "good": None,  # good은 마스크 없음: 제거해도 무방
    }

    # 증강 데이터는 각 클래스별로 별도 처리하므로 여기서는 제거

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
    - p_img: probability to apply ANY augmentation to this image (default 0.2)
    - p: probability for each enabled aug (default 0.3)
    - brightness: (min,max) e.g. (0.8,1.2)
    - contrast: (min,max)
    - gamma: (min,max)
    - blur_sigma: (min,max)
    - noise_std: (min,max) in [0,1]
    - specular: dict with strength/radius
    - colorjitter: dict with saturation/hue
    """
    augs = set([a.lower() for a in aug_config.get("augs", [])])
    if not augs:
        return img_pil

    # ✅ 대부분은 원본 유지: 이미지 단위 gate
    p_img = float(aug_config.get("p_img", 0.2))  # ex) 0.2 => 80% 원본
    if random.random() >= p_img:
        return img_pil

    # ✅ aug별 확률 (여러 개 동시에 걸릴 수도 있지만 낮게)
    p = float(aug_config.get("p", 0.3))

    # 1) brightness/contrast/gamma
    if "bcg" in augs and random.random() < p:
        bmin, bmax = aug_config.get("brightness", (0.9, 1.1))
        cmin, cmax = aug_config.get("contrast", (0.9, 1.1))
        gmin, gmax = aug_config.get("gamma", (0.95, 1.05))

        img_pil = ImageEnhance.Brightness(img_pil).enhance(_rand_uniform(bmin, bmax))
        img_pil = ImageEnhance.Contrast(img_pil).enhance(_rand_uniform(cmin, cmax))

        gamma = _rand_uniform(gmin, gmax)
        if abs(gamma - 1.0) > 1e-6:
            arr = np.asarray(img_pil).astype(np.float32) / 255.0  # ✅ 수정됨
            arr = np.clip(arr, 0.0, 1.0) ** gamma
            img_pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    # 2) blur
    if "blur" in augs and random.random() < p:
        smin, smax = aug_config.get("blur_sigma", (0.5, 1.2))
        sigma = _rand_uniform(smin, smax)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 3) noise
    if "noise" in augs and random.random() < p:
        nmin, nmax = aug_config.get("noise_std", (0.005, 0.02))
        std = _rand_uniform(nmin, nmax)
        arr = np.asarray(img_pil).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        img_pil = Image.fromarray((arr * 255.0).astype(np.uint8))

    # 4) specular highlight
    if "specular" in augs and random.random() < p:
        cfg = aug_config.get("specular", {})
        strength = float(cfg.get("strength", 0.5))
        radius_min = int(cfg.get("radius_min", 20))
        radius_max = int(cfg.get("radius_max", 80))
        n_blobs = int(cfg.get("n_blobs", 1))

        arr = np.asarray(img_pil).astype(np.float32) / 255.0
        h, w = arr.shape[0], arr.shape[1]
        yy, xx = np.mgrid[0:h, 0:w]

        for _ in range(n_blobs):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            r = random.randint(radius_min, max(radius_min, radius_max))
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            blob = np.exp(-dist2 / (2.0 * (r ** 2))).astype(np.float32)
            blob = blob[..., None]
            arr = np.clip(arr + strength * blob, 0.0, 1.0)

        img_pil = Image.fromarray((arr * 255.0).astype(np.uint8))

    # 5) color jitter
    if "colorjitter" in augs and random.random() < p:
        cfg = aug_config.get("colorjitter", {})
        smin, smax = cfg.get("saturation", (0.9, 1.1))
        hue = float(cfg.get("hue", 0.02))
        img_pil = ImageEnhance.Color(img_pil).enhance(_rand_uniform(smin, smax))

        arr = np.asarray(img_pil).astype(np.uint8)
        hsv = Image.fromarray(arr, mode="RGB").convert("HSV")
        hsv_np = np.asarray(hsv).astype(np.uint8)
        shift = int(_rand_uniform(-hue, hue) * 255)
        hsv_np[..., 0] = (hsv_np[..., 0].astype(int) + shift) % 256
        img_pil = Image.fromarray(hsv_np, mode="HSV").convert("RGB")

    return img_pil

def find_mask(img_path: Path, defect_type: str, mask_dirs: Dict[str, Path]) -> Optional[Path]:
    if defect_type == "good":
        return None

    fname = img_path.name

    # 증강 데이터인지 확인 (aug_로 시작하는 경우)
    if fname.startswith("aug_"):
        # 증강 데이터의 마스크는 aug_mask 폴더에서 찾기
        base_dir = img_path.parent.parent  # aug 폴더의 부모 디렉토리
        aug_mask_dir = base_dir / "aug_mask"
        mask_fname = os.path.splitext(fname)[0] + ".png"
        mask_path = aug_mask_dir / mask_fname
        return mask_path if mask_path.exists() else None

    # 원본 데이터 처리
    if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg"):
        fname = os.path.splitext(fname)[0] + ".png"
    mask_path = mask_dirs[defect_type] / fname
    return mask_path if mask_path.exists() else None

class DefectSegDataset(Dataset):  # ✅ 수정됨
    """Binary Segmentation dataset. returns (img, mask, cls, img_path)"""

    def __init__(
        self,
        base_path: Path,
        split: str,
        img_size_hw: Tuple[int, int],
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
        seed: int = 42,
        include_augmented: bool = True,
        aug_config: Optional[dict] = None,
    )  -> None:
        assert split in {"train", "val", "test"}
        self.split = split
        self.base_path = base_path
        self.img_h, self.img_w = img_size_hw

        # 기본 경로 설정 (증강 데이터 제외)
        paths = default_paths(base_path, False)
        use_augmented = include_augmented and (split == "train")
        self.img_dirs = paths.img_dirs
        self.mask_dirs = paths.mask_dirs

        self.aug_config = aug_config or {}

        rng = np.random.default_rng(seed)
        self.samples: List[Tuple[Path, str]] = []

        for cls, folder in self.img_dirs.items():
            # 원본 이미지만 수집
            original_imgs = _list_images(folder)
            
            # 증강 데이터 수집 (모든 split에서 분할용으로 사용)
            aug_imgs = []
            if use_augmented and (base_path / "aug").exists():
                aug_dir = base_path / "aug"
                aug_imgs = [p for p in _list_images(aug_dir) if p.name.startswith(f"aug_{cls}_")]
            
            # 원본만으로 train/val/test 분할
            rng.shuffle(original_imgs)
            n_total = len(original_imgs)
            n_train = int(n_total * train_ratio)
            n_test = int(n_total * test_ratio)

            train_original = original_imgs[:n_train]
            test_original = original_imgs[n_train: n_train + n_test]
            val_original = original_imgs[n_train + n_test:]

            # split에 따라 선택
            if split == "train":
                # train에서는 원본 + 증강 데이터 모두 포함
                chosen = train_original + aug_imgs
            elif split == "test":
                # test에서는 원본만
                chosen = test_original
            else:  # val
                # val에서는 원본만
                chosen = val_original

            self.samples.extend([(p, cls) for p in chosen])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb_pil(self, path: Path) -> Image.Image:  # ✅ 추가됨
        return Image.open(path).convert("RGB").resize((self.img_w, self.img_h), Image.BILINEAR)

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

        mask = Image.open(mpath).convert("L").resize((self.img_w, self.img_h), Image.NEAREST)
        m = np.asarray(mask)
        return (m > 0).astype(np.uint8) * 255

    def __getitem__(self, idx: int):
        img_path, cls = self.samples[idx]  # ✅ 중복 제거됨

        # ✅ PIL로 로드
        img_pil = self._load_rgb_pil(img_path)

        # ✅ train에서만 photometric aug 적용 (val/test는 절대 X)
        if self.split == "train" and self.aug_config.get("augs"):
            img_pil = _apply_photometric_aug(img_pil, self.aug_config)

        img = np.asarray(img_pil).copy()  # ✅ 중복 제거됨
        mask = self._load_mask(img_path, cls)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # ✅ 수정됨
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # ✅ 수정됨
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
    include_augmented: bool = True,
):
    # ✅ train만 aug_config 사용, val/test는 None(=augmentation 완전 OFF)
    train_ds = DefectSegDataset(base_path, "train", img_size_hw, train_ratio, test_ratio, seed, include_augmented, aug_config)  # ✅ 수정됨
    test_ds  = DefectSegDataset(base_path, "test",  img_size_hw, train_ratio, test_ratio, seed, include_augmented, None)  # ✅ 수정됨
    val_ds   = DefectSegDataset(base_path, "val",   img_size_hw, train_ratio, test_ratio, seed, include_augmented, None)  # ✅ 수정됨

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  # ✅ 수정됨
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader