"""
Shared evaluation metrics for image quality assessment.
Provides PSNR, SSIM computation and an AverageMeter for tracking training stats.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(channels, 1, window_size, window_size).to(pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
