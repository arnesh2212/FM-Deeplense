"""
Shared masking helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


GridSize = Tuple[int, int]


@dataclass
class MaskingConfig:
    image_size: int = 64
    patch_size: int = 8

    @property
    def grid_size(self) -> GridSize:
        side = self.image_size // self.patch_size
        return side, side

    @property
    def num_patches(self) -> int:
        height, width = self.grid_size
        return height * width


def flatten_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError("Expected mask shape [batch, height, width].")
    return mask.flatten(1)
