import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, widths=(32, 64, 128, 256, 320, 320)):
        super().__init__()
        widths = tuple(widths)
        if in_channels < 1 or num_classes < 2 or len(widths) < 2 or any(w < 1 for w in widths):
            raise ValueError("Expected positive channels, at least two classes, and at least two positive widths")

        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoder.append(_conv_block(previous, width))
            previous = width

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up_projection = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up_projection.append(nn.Conv3d(previous, width, kernel_size=1))
            self.decoder.append(_conv_block(2 * width, width))
            previous = width

        self.head = nn.Conv3d(widths[0], num_classes, kernel_size=1)

    def encode(self, image):
        features = []
        x = image
        for index, block in enumerate(self.encoder):
            if index:
                x = self.pool(x)
            x = block(x)
            features.append(x)
        return tuple(features)

    def forward(self, image):
        features = self.encode(image)
        x = features[-1]
        for projection, block, skip in zip(
            self.up_projection, self.decoder, reversed(features[:-1])
        ):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = projection(x)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)
