from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualPriorAdapter(nn.Module):
    """Validity-gated, identity-initialized residual modulation of an image skip."""

    def __init__(self, feature_channels: int, prior_channels: int, beta_init: float = 0.05,
                 beta_max: float = 0.5):
        super().__init__()
        if feature_channels < 1 or prior_channels < 1:
            raise ValueError("feature_channels and prior_channels must be positive")
        if not 0 < beta_init < beta_max:
            raise ValueError("beta_init must be positive and smaller than beta_max")
        hidden = max(8, min(feature_channels, prior_channels * 2))
        self.attention = nn.Sequential(
            nn.Conv2d(feature_channels + prior_channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, feature_channels, kernel_size=1),
        )
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)
        self.beta_max = float(beta_max)
        ratio = beta_init / beta_max
        self.beta_logit = nn.Parameter(torch.tensor(math.log(ratio / (1 - ratio)), dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_max * torch.sigmoid(self.beta_logit)

    def forward(self, image_features: torch.Tensor, prior_features: torch.Tensor,
                gate_map: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 4 or prior_features.ndim != 4 or gate_map.ndim != 4:
            raise ValueError("adapter expects NCHW tensors")
        if image_features.shape[0] != prior_features.shape[0] or image_features.shape[0] != gate_map.shape[0]:
            raise ValueError("adapter batch dimensions must match")
        size = image_features.shape[-2:]
        if prior_features.shape[-2:] != size:
            prior_features = F.interpolate(prior_features, size=size, mode="bilinear", align_corners=False)
        if gate_map.shape[-2:] != size:
            gate_map = F.interpolate(gate_map, size=size, mode="bilinear", align_corners=False)
        if gate_map.shape[1] != 1:
            raise ValueError("gate_map must have one channel")
        gate_map = gate_map.clamp(0, 1)
        attention = torch.sigmoid(self.attention(torch.cat((image_features, prior_features), dim=1)))
        return image_features * (1 + gate_map * self.beta * (2 * attention - 1))
