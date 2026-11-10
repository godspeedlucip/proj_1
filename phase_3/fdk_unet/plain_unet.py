"""Five-level BN/ReLU FDK baseline with a single shared decoder."""
import torch
from torch import nn
from torch.nn import functional as F


def initialize_convolutions(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=True),
        nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1), nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=True),
        nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1), nn.ReLU(inplace=True),
    )


def check_same_grid(actual, reference):
    if (not isinstance(actual, torch.Tensor) or actual.shape != reference.shape
            or actual.dtype != reference.dtype or actual.device != reference.device):
        raise ValueError("Transform must preserve shape, dtype and device")
    return actual


class PlainUNet3D(nn.Module):
    channels = (16, 32, 64, 128, 256)

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(double_conv(a, b) for a, b in zip((1,) + self.channels, self.channels))
        self.pool = nn.MaxPool3d(2, 2)
        self.up_projection = nn.ModuleList(nn.Conv3d(self.channels[l + 1], self.channels[l], 3, padding=1)
                                           for l in (3, 2, 1, 0))
        self.decoder = nn.ModuleList(double_conv(2 * self.channels[l], self.channels[l]) for l in (3, 2, 1, 0))
        self.head = nn.Conv3d(16, 1, 1)
        self.apply(initialize_convolutions)

    def encode(self, image):
        if (image.ndim != 5 or image.shape[0] < 1 or image.shape[1] != 1
                or min(image.shape[2:]) < 32 or image.dtype != torch.float32
                or not torch.isfinite(image).all()):
            raise ValueError("Expected finite float32 [B,1,D,H,W], each spatial edge >= 32")
        features = []
        for level, block in enumerate(self.encoder):
            image = block(self.pool(image) if level else image)
            features.append(image)
        return tuple(features)

    def decode(self, features, *, bottleneck_transform=None, skip_transform=None):
        if len(features) != 5:
            raise ValueError("Expected five encoder features")
        first = features[0]
        if first.ndim != 5 or min(first.shape[2:]) < 32:
            raise ValueError("Invalid encoder feature shape")
        for level, (feature, channels) in enumerate(zip(features, self.channels)):
            expected = (first.shape[0], channels, *(s // (2 ** level) for s in first.shape[2:]))
            if (tuple(feature.shape) != expected or feature.device != first.device
                    or feature.dtype != first.dtype):
                raise ValueError("Encoder features have inconsistent shapes, dtype or device")
        x = features[-1]
        if bottleneck_transform is not None:
            x = check_same_grid(bottleneck_transform(x), x)
        for level, projection, block in zip((3, 2, 1, 0), self.up_projection, self.decoder):
            skip = features[level]
            up = projection(F.interpolate(x, size=skip.shape[2:], mode="nearest"))
            if skip_transform is not None:
                skip = check_same_grid(skip_transform(level, skip, up), skip)
            x = block(torch.cat((up, skip), dim=1))
        return self.head(x)

    def forward(self, image):
        return self.decode(self.encode(image))
