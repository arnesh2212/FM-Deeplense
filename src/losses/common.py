"""
Shared helpers for representation-learning losses.
"""
from __future__ import annotations

import torch


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix.")
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
