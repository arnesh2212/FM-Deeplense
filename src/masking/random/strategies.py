"""
Random masking strategies over a patch grid.
"""
from __future__ import annotations

from typing import Tuple

import torch


GridSize = Tuple[int, int]


class RandomTokenMasking:
    """
    Select exactly ``mask_ratio`` fraction of tokens uniformly at random.
    """

    def __init__(self, mask_ratio: float = 0.4):
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1.")
        self.mask_ratio = mask_ratio

    def __call__(self, batch_size: int, grid_size: GridSize, device: torch.device | None = None) -> torch.Tensor:
        height, width = grid_size
        num_patches = height * width
        num_masked = max(1, int(num_patches * self.mask_ratio))

        noise = torch.rand(batch_size, num_patches, device=device)
        indices = noise.argsort(dim=1)[:, :num_masked]
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)
        return mask.view(batch_size, height, width)


class BernoulliMasking:
    """
    Independently mask each patch with probability ``mask_prob``.
    """

    def __init__(self, mask_prob: float = 0.4):
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")
        self.mask_prob = mask_prob

    def __call__(self, batch_size: int, grid_size: GridSize, device: torch.device | None = None) -> torch.Tensor:
        return torch.rand(batch_size, *grid_size, device=device) < self.mask_prob
