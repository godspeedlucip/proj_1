"""Shared-view detector with trainable epipolar heatmap refinement."""

import torch
from torch import nn

from ..epipolar import EpipolarSupport


class JointEpipolarMarkerNet(nn.Module):
    def __init__(self, backbone: nn.Module, support: EpipolarSupport) -> None:
        super().__init__()
        self.backbone = backbone
        self.support = support
        self.quality = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.refine = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1)
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        self.last_view_weights: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5 or images.shape[1:3] != (3, 1):
            raise ValueError("joint detector input must have shape [B,3,1,H,W]")
        batch, _, _, height, width = images.shape
        initial = self.backbone(images.reshape(batch * 3, 1, height, width))
        if initial.shape != (batch * 3, 1, height, width):
            raise ValueError("backbone must return one full-resolution heatmap per view")
        initial = initial.reshape(batch, 3, 1, height, width)
        probabilities = torch.sigmoid(initial)
        support = self.support(probabilities)
        own_flat = probabilities.flatten(3)
        support_flat = support.flatten(3)
        summary = torch.stack(
            (
                own_flat.mean(-1).squeeze(2),
                own_flat.amax(-1).squeeze(2),
                support_flat.mean((2, 3)),
                support_flat.amax((2, 3)),
            ),
            dim=-1,
        )
        weights = torch.softmax(self.quality(summary).squeeze(-1), dim=1)
        self.last_view_weights = weights
        refined = []
        for target in range(3):
            sources = [source for source in range(3) if source != target]
            channels = torch.cat(
                (
                    initial[:, target],
                    support[:, target, 0:1] * weights[:, sources[0], None, None, None],
                    support[:, target, 1:2] * weights[:, sources[1], None, None, None],
                ),
                dim=1,
            )
            refined.append(initial[:, target] + self.refine(channels))
        return initial, torch.stack(refined, dim=1)
