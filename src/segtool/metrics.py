from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch

@dataclass
class Metrics:
    dice: float
    iou: float
    precision: float
    recall: float

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> Metrics:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    tp = (pred * targets).sum()
    fp = (pred * (1 - targets)).sum()
    fn = ((1 - pred) * targets).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return Metrics(float(dice.item()), float(iou.item()), float(precision.item()), float(recall.item()))

def reduce_mean(metrics_list: List[Metrics]) -> Metrics:
    if not metrics_list:
        return Metrics(0.0, 0.0, 0.0, 0.0)
    d = sum(m.dice for m in metrics_list) / len(metrics_list)
    i = sum(m.iou for m in metrics_list) / len(metrics_list)
    p = sum(m.precision for m in metrics_list) / len(metrics_list)
    r = sum(m.recall for m in metrics_list) / len(metrics_list)
    return Metrics(d, i, p, r)
