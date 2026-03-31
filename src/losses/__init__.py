"""
Reusable loss functions for self-supervised and representation-learning experiments.
"""

from .common import off_diagonal
from .lejepa import LeJEPALoss, SIGReg, invariance_loss
from .reconstruction import masked_l1_loss, masked_mse_loss, smooth_l1_jepa_loss
from .ssl import (
    VICRegLossOutput,
    cosine_similarity_loss,
    covariance_loss,
    variance_loss,
    vicreg_loss,
)

__all__ = [
    "LeJEPALoss",
    "SIGReg",
    "VICRegLossOutput",
    "cosine_similarity_loss",
    "covariance_loss",
    "invariance_loss",
    "masked_l1_loss",
    "masked_mse_loss",
    "off_diagonal",
    "smooth_l1_jepa_loss",
    "variance_loss",
    "vicreg_loss",
]
