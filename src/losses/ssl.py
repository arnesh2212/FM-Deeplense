"""
Additional self-supervised learning losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .common import off_diagonal


def cosine_similarity_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 1.0 - (p * z).sum(dim=-1).mean()


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    return off_diagonal(cov).square().mean()


@dataclass
class VICRegLossOutput:
    total: torch.Tensor
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor


def vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> VICRegLossOutput:
    inv = F.mse_loss(x, y)
    var = variance_loss(x) + variance_loss(y)
    cov = covariance_loss(x) + covariance_loss(y)
    total = sim_coeff * inv + var_coeff * var + cov_coeff * cov
    return VICRegLossOutput(total=total, invariance=inv, variance=var, covariance=cov)
