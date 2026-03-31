"""
LeJEPA losses based on the official minimal example and paper description.

LeJEPA combines:
- an invariance term across multiple views / augmentations
- SIGReg (Sketched Isotropic Gaussian Regularization)

References:
- LeJEPA paper: https://arxiv.org/abs/2511.08544
- Official minimal implementation:
  https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _move_view_axis(proj: torch.Tensor, view_axis: int) -> torch.Tensor:
    if proj.ndim != 3:
        raise ValueError("Expected `proj` to have shape [views, batch, dim] or [batch, views, dim].")
    return proj.movedim(view_axis, 0)


def invariance_loss(proj: torch.Tensor, view_axis: int = 0) -> torch.Tensor:
    """
    Compute the LeJEPA invariance loss across multiple augmented views.

    Parameters
    ----------
    proj:
        Projected features with shape ``[views, batch, dim]`` or ``[batch, views, dim]``.
    view_axis:
        Axis corresponding to the view dimension.
    """
    proj = _move_view_axis(proj, view_axis=view_axis)
    return (proj.mean(dim=0, keepdim=True) - proj).square().mean()


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    This follows the official minimal LeJEPA implementation, but avoids hardcoding
    CUDA so it can run on CPU or GPU transparently.
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0, num_random_projections: int = 256):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be at least 2.")
        if num_random_projections < 1:
            raise ValueError("num_random_projections must be at least 1.")

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.num_random_projections = num_random_projections
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor, view_axis: int = 0) -> torch.Tensor:
        proj = _move_view_axis(proj, view_axis=view_axis)
        _, _, dim = proj.shape

        random_matrix = torch.randn(
            dim,
            self.num_random_projections,
            device=proj.device,
            dtype=proj.dtype,
        )
        random_matrix = random_matrix / random_matrix.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        x_t = (proj @ random_matrix).unsqueeze(-1) * self.t.to(device=proj.device, dtype=proj.dtype)
        err = (x_t.cos().mean(dim=-3) - self.phi.to(device=proj.device, dtype=proj.dtype)).square()
        err = err + x_t.sin().mean(dim=-3).square()

        statistic = (err @ self.weights.to(device=proj.device, dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean()


@dataclass
class LeJEPALossOutput:
    total: torch.Tensor
    invariance: torch.Tensor
    sigreg: torch.Tensor


class LeJEPALoss(nn.Module):
    """
    Combined LeJEPA objective:

    ``loss = (1 - lambda) * invariance_loss + lambda * sigreg_loss``
    """

    def __init__(
        self,
        lamb: float = 0.1,
        knots: int = 17,
        t_max: float = 3.0,
        num_random_projections: int = 256,
    ):
        super().__init__()
        if not 0.0 <= lamb <= 1.0:
            raise ValueError("lamb must be in [0, 1].")
        self.lamb = lamb
        self.sigreg = SIGReg(
            knots=knots,
            t_max=t_max,
            num_random_projections=num_random_projections,
        )

    def forward(self, proj: torch.Tensor, view_axis: int = 0) -> LeJEPALossOutput:
        inv = invariance_loss(proj, view_axis=view_axis)
        sig = self.sigreg(proj, view_axis=view_axis)
        total = (1.0 - self.lamb) * inv + self.lamb * sig
        return LeJEPALossOutput(total=total, invariance=inv, sigreg=sig)
