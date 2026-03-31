"""
Reusable loss functions for self-supervised and representation-learning experiments.
"""

from .lejepa import LeJEPALoss, SIGReg, invariance_loss

__all__ = ["LeJEPALoss", "SIGReg", "invariance_loss"]
