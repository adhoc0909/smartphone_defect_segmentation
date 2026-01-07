from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

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
        "good": base / "good",  # good 클래스 추가
    }
    mask_dirs = {
        "scratch": base / "ground_truth_1",
        "stain": base / "ground_truth_1",
        "oil": base / "ground_truth_2",
        "good": None,  # good은 마스크 없음
    }
    
    # 증강 데이터 추가 (새 스크래치 클래스 포함)
    if include_augmented and (base / "aug").exists():
        img_dirs["scratch_synthetic"] = base / "aug"  # 새 스크래치 클래스
        mask_dirs["scratch_synthetic"] = base / "aug_mask"
    
    return DatasetPaths(img_dirs=img_dirs, mask_dirs=mask_dirs)

def _list_images(folder: Path) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    out: List[Path] = []
    for e in exts:
        out.extend([Path(p) for p in glob.glob(str(folder / e))])
    return sorted(out)


def find_mask(img_path: Path, defect_type: str, mask_dirs: Dict[str, Path]) -> Optional[Path]:
    if defect_type == "good":
        return None
    
    fname = img_path.name
    
    # 새 스크래치 클래스 처리
    if defect_type == "scratch_synthetic":
        mask_fname = os.path.splitext(fname)[0] + ".png"
        mask_path = mask_dirs[defect_type] / mask_fname
        return mask_path if mask_path.exists() else None
    
    # 원본 데이터 처리
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
        include_augmented: bool = True,
    )  -> None:
        assert split in {"train", "val", "test"}
        self.base_path = base_path
        self.img_h, self.img_w = img_size_hw
        
        # 증강 데이터 포함 (디버깅 완료)
        use_augmented = include_augmented
        paths = default_paths(base_path, use_augmented)
        self.img_dirs = paths.img_dirs
        self.mask_dirs = paths.mask_dirs

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

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> np.ndarray:
        img = (
            Image.open(path).convert("RGB").resize((self.img_w, self.img_h), Image.BILINEAR)
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
        img_path, cls = self.samples[idx]

        img = self._load_rgb(img_path)
        mask = self._load_mask(img_path, cls)
        img = np.asarray(img).copy()
        mask = np.asarray(mask).copy()
        img = np.asarray(img).copy()
        mask = np.asarray(mask).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask_t = (mask_t > 0.5).float()
        
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
    include_augmented: bool = True,
):
    train_ds = DefectSegDataset(base_path, "train", img_size_hw, train_ratio, test_ratio, seed)
    test_ds = DefectSegDataset(base_path, "test", img_size_hw, train_ratio, test_ratio, seed)
    val_ds = DefectSegDataset(base_path, "val", img_size_hw, train_ratio, test_ratio, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader