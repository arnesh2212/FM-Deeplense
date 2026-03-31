"""
Reusable masking strategies for self-supervised image experiments.
"""

from .base import MaskingConfig, flatten_mask
from .block import BlockMasking, CenterBlockMasking, MultiBlockMasking
from .custom import CheckerboardMasking
from .random import BernoulliMasking, RandomTokenMasking

__all__ = [
    "BernoulliMasking",
    "BlockMasking",
    "CenterBlockMasking",
    "CheckerboardMasking",
    "MaskingConfig",
    "MultiBlockMasking",
    "RandomTokenMasking",
    "flatten_mask",
]
