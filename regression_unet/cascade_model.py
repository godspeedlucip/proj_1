"""Standalone PyTorch U-Nets for mask-first continuous stent projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class Encoder(nn.Module):
    def __init__(self, input_channels: int, widths: tuple[int, ...]):
        super().__init__()
        self.stages = nn.ModuleList([ConvBlock(input_channels, widths[0])])
        for previous, current in zip(widths[:-1], widths[1:]):
            self.stages.append(nn.Sequential(
                nn.Conv2d(previous, previous, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(previous, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                ConvBlock(previous, current)))

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for stage in self.stages:
            image = stage(image)
            skips.append(image)
        return skips


class Decoder(nn.Module):
    def __init__(self, widths: tuple[int, ...]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for current, skip in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.ups.append(nn.ConvTranspose2d(current, skip, 2, stride=2))
            self.blocks.append(ConvBlock(skip * 2, skip))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        value = skips[-1]
        for up, block, skip in zip(self.ups, self.blocks, reversed(skips[:-1])):
            value = up(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return self.head(value)


class PlainUNet(nn.Module):
    def __init__(self, input_channels: int, widths: tuple[int, ...]):
        super().__init__()
        self.encoder = Encoder(input_channels, widths)
        self.decoder = Decoder(widths)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class MarkerMaskUNet(nn.Module):
    """Only Phase 1 point heatmaps are fused; the straight marker band is ignored."""

    def __init__(self, widths: tuple[int, ...]):
        super().__init__()
        self.encoder = Encoder(1, widths)
        self.decoder = Decoder(widths)
        self.point_stem = ConvBlock(1, 16)
        self.point_adapters = nn.ModuleList([nn.Conv2d(16, widths[0], 1),
                                             nn.Conv2d(16, widths[1], 1)])
        for adapter in self.point_adapters:
            nn.init.normal_(adapter.weight, std=0.01)
            nn.init.zeros_(adapter.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("expected [N,4,H,W]: image, point, band, validity")
        skips = self.encoder(inputs[:, :1])
        point_feature = self.point_stem(inputs[:, 1:2])
        validity = inputs[:, 3:4]
        for index, adapter in enumerate(self.point_adapters):
            shape = skips[index].shape[-2:]
            prior = F.interpolate(point_feature, size=shape, mode="bilinear", align_corners=False)
            gate = F.interpolate(validity, size=shape, mode="nearest")
            skips[index] = skips[index] + gate * adapter(prior)
        return self.decoder(skips)


class MaskThenDepthUNet(nn.Module):
    def __init__(self, mask_widths: tuple[int, ...] = (32, 64, 128, 256, 320),
                 depth_widths: tuple[int, ...] = (24, 48, 96, 192, 256)):
        super().__init__()
        if len(mask_widths) != len(depth_widths) or len(mask_widths) < 3:
            raise ValueError("cascade U-Nets require matching stage counts of at least three")
        self.mask_network = MarkerMaskUNet(mask_widths)
        self.depth_network = PlainUNet(2, depth_widths)

    @property
    def encoder(self):
        return self.mask_network.encoder

    def forward(self, inputs: torch.Tensor, return_aux: bool = False):
        mask_logits = self.mask_network(inputs)
        mask_probability = torch.sigmoid(mask_logits)
        masked_image = inputs[:, :1] * mask_probability
        depth = torch.sigmoid(self.depth_network(torch.cat((masked_image, mask_probability), dim=1)))
        projection = depth * mask_probability
        if return_aux:
            return {"mask_logits": mask_logits, "mask_probability": mask_probability,
                    "masked_image": masked_image, "depth": depth, "projection": projection}
        return projection
