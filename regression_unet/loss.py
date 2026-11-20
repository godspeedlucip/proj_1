"""Sparse float projection regression loss and metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                       foreground_weight: float = 100.0, beta: float = 0.05) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if foreground_weight < 1 or beta <= 0:
        raise ValueError("foreground_weight must be >=1 and beta positive")
    weights = 1 + (foreground_weight - 1) * (target > 0).to(prediction.dtype)
    error = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return (error * weights).sum() / weights.sum()


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Report intensity error on support and false-positive energy off support."""
    prediction = prediction.clamp_min(0)
    foreground = target > 0
    background = ~foreground
    return {
        "foreground_mae": float((prediction[foreground] - target[foreground]).abs().mean())
        if foreground.any() else 0.0,
        "background_mean_prediction": float(prediction[background].mean())
        if background.any() else 0.0,
    }


def mask_depth_loss(details: dict[str, torch.Tensor], target: torch.Tensor,
                    binary_mask: torch.Tensor | None = None,
                    positive_weight: float = 100.0,
                    mask_only: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise support and conditional projection intensity as separate tasks."""
    target = target.float()
    mask = (target > 0).float() if binary_mask is None else binary_mask.float()
    if mask.shape != target.shape or not torch.equal(mask, (target > 0).float()):
        raise ValueError("binary mask must equal the positive support of the continuous target")
    logits = details["mask_logits"].float()
    probability = details["mask_probability"].float()
    conditional_depth = details["depth"].float()
    projection = details["projection"].float()
    if any(value.shape != target.shape for value in (logits, probability, conditional_depth, projection)):
        raise ValueError("cascade output and target shapes differ")
    weight = torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=weight)
    intersection = (probability * mask).sum(dim=(2, 3))
    dice = 1 - ((2 * intersection + 1) /
                (probability.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) + 1)).mean()
    foreground_error = F.smooth_l1_loss(conditional_depth, target, reduction="none", beta=0.05)
    foreground_error = (foreground_error * mask).sum() / mask.sum().clamp_min(1)
    background_error = (projection * (1 - mask)).mean()
    total = bce + dice if mask_only else bce + dice + foreground_error + background_error
    components = {"mask_bce": float(bce.detach()), "mask_dice_loss": float(dice.detach()),
                  "depth_foreground_loss": float(foreground_error.detach()),
                  "background_leakage": float(background_error.detach())}
    return total, components


def mask_metrics(probability: torch.Tensor, binary_mask: torch.Tensor) -> dict[str, float]:
    predicted = probability.float() >= 0.5
    target = binary_mask.bool()
    true_positive = (predicted & target).sum().float()
    predicted_count = predicted.sum().float()
    target_count = target.sum().float()
    return {"mask_dice": float((2 * true_positive + 1) /
                               (predicted_count + target_count + 1)),
            "mask_precision": float((true_positive + 1) / (predicted_count + 1)),
            "mask_recall": float((true_positive + 1) / (target_count + 1))}
