"""Masked inverse-square generalized Dice for hard or soft foreground targets."""

import torch
from torch import Tensor


def masked_generalized_dice(probability: Tensor, foreground: Tensor, valid: Tensor) -> Tensor:
    """Return one GDL value per sample for tensors shaped [N,1,*spatial]."""
    axes = tuple(range(2, probability.ndim))
    predicted = torch.cat((1 - probability, probability), dim=1)
    truth = torch.cat((1 - foreground, foreground), dim=1)
    volumes = (truth * valid).sum(axes)
    weights = torch.where(volumes > 0, volumes.clamp_min(1).reciprocal().square(), 0)
    weights = weights / weights.amax(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    intersection = (predicted * truth * valid).sum(axes)
    size = ((predicted + truth) * valid).sum(axes)
    return 1 - (2 * (weights * intersection).sum(dim=1) + 1e-6) / (
        (weights * size).sum(dim=1) + 1e-6)
