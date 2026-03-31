"""
Block-style masking utilities.
"""
from __future__ import annotations

from typing import Tuple

import torch


GridSize = Tuple[int, int]


class BlockMasking:
    """
    Sample one contiguous random rectangle per image.
    """

    def __init__(self, min_block_size: int = 2, max_block_size: int = 4):
        if min_block_size < 1 or max_block_size < min_block_size:
            raise ValueError("Invalid block size range.")
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    def __call__(self, batch_size: int, grid_size: GridSize, device: torch.device | None = None) -> torch.Tensor:
        height, width = grid_size
        mask = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            block_h = int(torch.randint(self.min_block_size, self.max_block_size + 1, (1,), device=device).item())
            block_w = int(torch.randint(self.min_block_size, self.max_block_size + 1, (1,), device=device).item())
            block_h = min(block_h, height)
            block_w = min(block_w, width)

            top = int(torch.randint(0, height - block_h + 1, (1,), device=device).item())
            left = int(torch.randint(0, width - block_w + 1, (1,), device=device).item())
            mask[batch_idx, top : top + block_h, left : left + block_w] = True

        return mask


class CenterBlockMasking:
    """
    Mask a centered rectangular region with configurable relative size.
    """

    def __init__(self, height_ratio: float = 0.5, width_ratio: float = 0.5):
        if not 0.0 < height_ratio <= 1.0 or not 0.0 < width_ratio <= 1.0:
            raise ValueError("height_ratio and width_ratio must be in (0, 1].")
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def __call__(self, batch_size: int, grid_size: GridSize, device: torch.device | None = None) -> torch.Tensor:
        height, width = grid_size
        block_h = max(1, int(height * self.height_ratio))
        block_w = max(1, int(width * self.width_ratio))
        top = (height - block_h) // 2
        left = (width - block_w) // 2

        mask = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
        mask[:, top : top + block_h, left : left + block_w] = True
        return mask


class MultiBlockMasking:
    """
    Union of multiple random block masks.
    """

    def __init__(self, num_blocks: int = 3, min_block_size: int = 2, max_block_size: int = 4):
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        self.num_blocks = num_blocks
        self.block_masking = BlockMasking(
            min_block_size=min_block_size,
            max_block_size=max_block_size,
        )

    def __call__(self, batch_size: int, grid_size: GridSize, device: torch.device | None = None) -> torch.Tensor:
        combined = torch.zeros(batch_size, *grid_size, dtype=torch.bool, device=device)
        for _ in range(self.num_blocks):
            combined |= self.block_masking(batch_size=batch_size, grid_size=grid_size, device=device)
        return combined
