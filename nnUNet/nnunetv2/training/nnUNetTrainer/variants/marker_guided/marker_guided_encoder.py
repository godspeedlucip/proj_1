from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MarkerGuidedPriorEncoder(nn.Module):
    """Compact two-channel encoder projected onto the plans-defined skip scales."""

    def __init__(self, input_channels: int, feature_channels: list[int], stage_indices: list[int],
                 prior_feature_ratio: float = 0.25):
        super().__init__()
        if not 0 < prior_feature_ratio <= 1:
            raise ValueError("prior_feature_ratio must be in (0, 1]")
        prior_width = max(8, int(min(feature_channels) * prior_feature_ratio))
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, prior_width, 3, padding=1, bias=False),
            nn.InstanceNorm2d(prior_width, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(prior_width, prior_width, 3, padding=1, bias=False),
            nn.InstanceNorm2d(prior_width, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        if not stage_indices or any(i < 0 or i >= len(feature_channels) for i in stage_indices):
            raise ValueError("stage_indices must select valid feature stages")
        self.stage_indices = list(stage_indices)
        self.projections = nn.ModuleDict({str(i): nn.Conv2d(prior_width, feature_channels[i], 1)
                                          for i in self.stage_indices})

    def project_level(self, encoded: torch.Tensor, index: int, size) -> torch.Tensor:
        level = F.interpolate(encoded, size=size, mode="bilinear", align_corners=False)
        return self.projections[str(index)](level)

    def forward(self, priors: torch.Tensor, skip_shapes: list[tuple[int, int]]) -> dict[int, torch.Tensor]:
        encoded = self.stem(priors)
        if len(skip_shapes) <= max(self.stage_indices):
            raise ValueError("skip shape list does not contain the selected plans stages")
        return {i: self.project_level(encoded, i, skip_shapes[i]) for i in self.stage_indices}
