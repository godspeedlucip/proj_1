from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@contextmanager
def _temporary_batchnorm_buffers(module):
    originals = []
    try:
        for layer in module.modules():
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                for name in ("running_mean", "running_var", "num_batches_tracked"):
                    value = getattr(layer, name)
                    if value is not None:
                        originals.append((layer, name, value))
                        setattr(layer, name, value.clone())
        yield
    finally:
        for layer, name, value in originals:
            setattr(layer, name, value)


def checkpoint_forward(function, batchnorm_root, *inputs):
    # Recompute with training-mode batch statistics and let updates fall on
    # disposable copies of the running buffers.
    return checkpoint(
        function, *inputs, use_reentrant=False,
        context_fn=lambda: (nullcontext(), _temporary_batchnorm_buffers(batchnorm_root)),
    )


def _decode(up, layer, x, skip):
    x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
    return layer(torch.cat((skip, up(x)), dim=1))

class UNet(nn.Module):
    def __init__(self, dims, in_channels, channels, dropout=0, final='sigmoid', bn_eps=1e-5, bn_momentum=0.1, bias=True):
        super().__init__()
        conv, bn, pool = (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d) if dims == 2 else (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d)
        drop = nn.Dropout2d if dims == 2 else nn.Dropout3d
        self.channels = tuple(channels)
        def block(ci, co):
            layers = []
            for a in (ci, co):
                layers.extend([conv(a, co, 3, padding=1, bias=bias), bn(co, eps=bn_eps, momentum=bn_momentum)])
                if dropout: layers.append(drop(dropout))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
        self.encoder = nn.ModuleList()
        ci = in_channels
        for co in channels:
            self.encoder.append(block(ci,co)); ci = co
        self.pool = pool(2)
        self.upconv = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for co in reversed(channels[:-1]):
            self.upconv.append(conv(ci, co, 3, padding=1, bias=bias))
            self.decoder.append(block(2*co, co)); ci = co
        self.output = conv(ci, 1, 1, bias=bias)
        self.final = final
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, *, activation_checkpointing=False):
        recompute = activation_checkpointing and self.training and torch.is_grad_enabled()
        skips = []
        for index, layer in enumerate(self.encoder):
            if index: x = self.pool(x)
            x = checkpoint_forward(layer, layer, x) if recompute else layer(x)
            skips.append(x)
        for up, layer, skip in zip(self.upconv, self.decoder, reversed(skips[:-1])):
            if recompute:
                x = checkpoint_forward(_decode, layer, up, layer, x, skip)
            else:
                x = _decode(up, layer, x, skip)
        x = self.output(x)
        return torch.sigmoid(x) if self.final == 'sigmoid' else F.relu(x)
