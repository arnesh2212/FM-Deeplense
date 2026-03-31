"""
Structured / deterministic mask layouts.
"""
from __future__ import annotations

from typing import Tuple

import torch


GridSize = Tuple[int, int]


class CheckerboardMasking:
    """
    Alternate masked / visible tokens in a checkerboard pattern.
    """

    def __init__(self, invert: bool = False):
        self.invert = invert

    def __call__(self, batch_size: int, grid_size: GridSize, device: torch.device | None = None) -> torch.Tensor:
        height, width = grid_size
        rows = torch.arange(height, device=device).unsqueeze(1)
        cols = torch.arange(width, device=device).unsqueeze(0)
        pattern = (rows + cols) % 2 == 0
        if self.invert:
            pattern = ~pattern
        return pattern.unsqueeze(0).expand(batch_size, -1, -1)
