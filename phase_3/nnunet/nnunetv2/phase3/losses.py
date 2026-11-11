"""Per-case masked segmentation and final-scale projection losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .projection import ProjectionConsistencyLoss


@dataclass
class LossBreakdown:
    total: Tensor
    segmentation: Tensor
    dice: Tensor
    bce: Tensor
    segmentation_scales: tuple[Tensor, ...]
    projection: Tensor
    projection_weighted: Tensor
    valid_rays: Tensor | None
    valid_sample_counts: tuple[int, ...]


def _validate_target(logits: Tensor, target: Tensor) -> Tensor:
    if logits.ndim != 5 or logits.shape[1] != 2:
        raise ValueError("Segmentation loss requires [B,2,D,H,W] logits")
    if target.ndim == 4:
        target = target[:, None]
    if target.shape != logits[:, :1].shape:
        raise ValueError("Segmentation target must match logits grid")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError("Segmentation logits and target must be finite")
    if not torch.all((target == 0) | (target == 1) | (target == 2)):
        raise ValueError("Target labels must be 0, 1, or ignore=2")
    return target


class MaskedForegroundDiceBCE(nn.Module):
    def forward(self, logits: Tensor, target: Tensor, *, allow_empty: bool = False) -> LossBreakdown:
        target = _validate_target(logits, target)
        z = (logits[:, 1:2] - logits[:, :1]).float()
        y = (target == 1).float()
        mask = (target != 2).float()
        axes = (1, 2, 3, 4)
        counts = mask.sum(axes)
        if not allow_empty and (counts == 0).any():
            raise ValueError("No valid voxels in sample")
        probability = torch.sigmoid(z)
        bce_per_case = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * mask).sum(axes) / counts.clamp_min(1)
        dice_per_case = 1 - (2 * (mask * probability * y).sum(axes) + 1) / (
            (mask * probability).sum(axes) + (mask * y).sum(axes) + 1)
        bce = bce_per_case.mean()
        dice = dice_per_case.mean()
        total = bce + dice
        zero = total * 0
        return LossBreakdown(total, total, dice, bce, (total,), zero, zero, None,
                             (int((counts > 0).sum().item()),))


class Phase3CompositeLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.segmentation = MaskedForegroundDiceBCE()
        self.projection = ProjectionConsistencyLoss(config) if config.projection.enabled else None

    def breakdown(self, outputs: Tensor | list[Tensor],
                  targets: Tensor | list[Tensor]) -> LossBreakdown:
        heads = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        labels = list(targets) if isinstance(targets, (list, tuple)) else [targets]
        if len(heads) != len(labels):
            raise ValueError("Deep supervision heads and targets must have equal counts")
        if len(heads) not in (1, 4):
            raise ValueError("Expected one or four deep supervision scales")
        if len(heads) == 4 and not self.config.deep_supervision.enabled:
            raise ValueError("Deep supervision is disabled in the experiment config")
        weights = self.config.deep_supervision.weights if len(heads) == 4 else (1.0,)
        scales = []
        counts = []
        first = None
        for index, (logits, target, weight) in enumerate(zip(heads, labels, weights)):
            target = _validate_target(logits, target)
            counts.append(int((target != 2).flatten(1).any(dim=1).sum().item()))
            if weight == 0:
                scales.append(heads[0].sum() * 0)
                continue
            result = self.segmentation(logits, target, allow_empty=index > 0)
            if first is None:
                first = result
            scales.append(result.total)
        segmentation = sum(weight * scale for weight, scale in zip(weights, scales))
        projection = (self.projection(heads[0], labels[0]) if self.projection is not None
                      else segmentation * 0)
        weighted = self.config.projection.weight * projection
        return LossBreakdown(segmentation + weighted, segmentation, first.dice, first.bce,
                             tuple(scales), projection, weighted,
                             self.projection.last_valid_ray_counts if self.projection is not None else None,
                             tuple(counts))

    def forward(self, outputs: Tensor | list[Tensor], targets: Tensor | list[Tensor]) -> Tensor:
        return self.breakdown(outputs, targets).total
