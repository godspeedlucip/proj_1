"""Residual multiscale refinement of the final two-class segmentation logits."""

import torch
from torch import nn
from torch.nn import functional as F


class TubularEnhancement(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Tubular enhancement requires at least two class logits")
        hidden = max(2 * num_classes, 8)
        self.dil1 = nn.Conv3d(num_classes, hidden, 3, dilation=1, padding=1)
        self.dil2 = nn.Conv3d(num_classes, hidden, 3, dilation=2, padding=2)
        self.dil3 = nn.Conv3d(num_classes, hidden, 3, dilation=3, padding=3)
        self.dil5 = nn.Conv3d(num_classes, hidden, 3, dilation=5, padding=5)
        self.norm_dil1 = nn.InstanceNorm3d(hidden)
        self.norm_dil2 = nn.InstanceNorm3d(hidden)
        self.norm_dil3 = nn.InstanceNorm3d(hidden)
        self.norm_dil5 = nn.InstanceNorm3d(hidden)
        self.fusion = nn.Sequential(
            nn.Conv3d(hidden * 4, num_classes, 1),
            nn.InstanceNorm3d(num_classes),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(num_classes, num_classes, 1),
        )
        nn.init.normal_(self.fusion[3].weight, std=1e-4)
        nn.init.zeros_(self.fusion[3].bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != self.dil1.in_channels:
            raise ValueError("Expected five-dimensional class logits")
        branches = [
            F.leaky_relu(norm(conv(logits)), inplace=True)
            for conv, norm in ((self.dil1, self.norm_dil1), (self.dil2, self.norm_dil2),
                               (self.dil3, self.norm_dil3), (self.dil5, self.norm_dil5))
        ]
        return logits + self.fusion(torch.cat(branches, dim=1))
