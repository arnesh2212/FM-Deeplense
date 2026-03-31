"""
Reconstruction-oriented losses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target).square()
    if mask is None:
        return loss.mean()
    mask = mask.to(dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target).abs()
    if mask is None:
        return loss.mean()
    mask = mask.to(dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def smooth_l1_jepa_loss(predicted_target: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(predicted_target, target, beta=beta)
