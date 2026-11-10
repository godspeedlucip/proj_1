"""Pointwise, zero-residual-initialized feature fusion."""
import torch
from torch import nn
from .plain_unet import check_same_grid, initialize_convolutions


class _Fusion(nn.Module):
    def __init__(self, swin_channels, unet_channels, multiplier):
        super().__init__()
        self.align = nn.Conv3d(swin_channels, unet_channels, 1, bias=True)
        self.gate = nn.Sequential(nn.Conv3d(multiplier*unet_channels, max(unet_channels//4,16), 1),
                                  nn.ReLU(), nn.Conv3d(max(unet_channels//4,16), unet_channels, 1))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.apply(initialize_convolutions)
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)
        self.telemetry = {}

    def _forward(self, feature, prior, up=None):
        if prior.ndim != 5 or prior.shape[0] != feature.shape[0] or prior.shape[2:] != feature.shape[2:]:
            raise ValueError('Fusion features must cover the same spatial grid')
        aligned = self.align(prior)
        check_same_grid(aligned, feature)
        parts = (feature, aligned) if up is None else (feature, aligned, up)
        gate = torch.sigmoid(self.gate(torch.cat(parts, dim=1)))
        beta = torch.tanh(self.alpha).to(feature.dtype)
        residual = beta * gate * aligned
        with torch.no_grad():
            self.telemetry = {'beta': float(beta), 'gate_mean': float(gate.float().mean()),
                              'gate_saturation_fraction': float(((gate < .01) | (gate > .99)).float().mean()),
                              'residual_ratio': float(residual.float().norm() / (feature.float().norm()+1e-8))}
        return feature + residual


class BottleneckFusion(_Fusion):
    def __init__(self, swin_channels=384, unet_channels=256):
        super().__init__(swin_channels, unet_channels, 2)

    def forward(self, feature, prior):
        return self._forward(feature, prior)


class DecoderGuidedFusion(_Fusion):
    def __init__(self, swin_channels, unet_channels, *, decoder_conditioning):
        super().__init__(swin_channels, unet_channels, 3)
        self.decoder_conditioning = decoder_conditioning

    def forward(self, feature, prior, up):
        check_same_grid(up, feature)
        return self._forward(feature, prior, up if self.decoder_conditioning else torch.zeros_like(up))
