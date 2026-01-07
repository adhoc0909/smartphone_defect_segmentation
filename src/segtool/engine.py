from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics, reduce_mean, Metrics

@dataclass
class EpochResult:
    loss: float
    metrics_all: Metrics
    metrics_defect_only: Metrics

def _defect_only_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> Metrics:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    ms: List[Metrics] = []
    for b in range(targets.size(0)):
        if targets[b].sum() > 0:
            ms.append(compute_metrics(logits[b:b+1], targets[b:b+1], threshold=threshold))
    return reduce_mean(ms)

def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device, threshold: float) -> EpochResult:
    model.train()
    total_loss = 0.0
    ms_all: List[Metrics] = []
    ms_def: List[Metrics] = []

    for imgs, masks, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
    

        loss = criterion(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        ms_all.append(compute_metrics(logits.detach(), masks, threshold=threshold))
        ms_def.append(_defect_only_metrics(logits.detach(), masks, threshold=threshold))

    return EpochResult(
        loss=total_loss / max(len(loader), 1),
        metrics_all=reduce_mean(ms_all),
        metrics_defect_only=reduce_mean(ms_def),
    )

@torch.no_grad()
def validate(model, loader: DataLoader, criterion, device, threshold: float) -> EpochResult:
    model.eval()
    total_loss = 0.0
    ms_all: List[Metrics] = []
    ms_def: List[Metrics] = []

    for imgs, masks, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, masks)

        total_loss += float(loss.item())
        ms_all.append(compute_metrics(logits, masks, threshold=threshold))
        ms_def.append(_defect_only_metrics(logits, masks, threshold=threshold))

    return EpochResult(
        loss=total_loss / max(len(loader), 1),
        metrics_all=reduce_mean(ms_all),
        metrics_defect_only=reduce_mean(ms_def),
    )