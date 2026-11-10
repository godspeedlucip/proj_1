"""Swin context feeds the bottleneck and the existing top-down decoder."""
from contextlib import nullcontext
import torch
from torch import nn
from .fusion import BottleneckFusion, DecoderGuidedFusion
from .initialization import named_rng
from .plain_unet import check_same_grid


class SwinPriorUNet3D(nn.Module):
    def __init__(self, unet, prior, *, fusion_config, seed=2026):
        super().__init__()
        self.unet = unet
        self.prior = prior
        self.prior.requires_grad_(False).eval()
        self.skips = nn.ModuleDict()
        for level, cs, cu in zip(fusion_config['unet_indices'], fusion_config['swin_channels'], fusion_config['unet_channels']):
            with named_rng(seed, f'fusion.scale{level}'):
                if level == 4:
                    self.bottleneck = BottleneckFusion(cs, cu)
                else:
                    self.skips[str(level)] = DecoderGuidedFusion(cs, cu, decoder_conditioning=fusion_config['decoder_conditioning'])
        if not fusion_config['bottleneck_enabled']:
            self.bottleneck = None

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.prior.parameters()):
            self.prior.eval()
        return self

    def forward(self, image_unet, image_swin):
        check_same_grid(image_swin, image_unet)
        if any(s % 32 for s in image_unet.shape[2:]) or not torch.isfinite(image_swin).all():
            raise ValueError('Fusion requires finite inputs with patch edges divisible by 32')
        features = self.unet.encode(image_unet)
        needs_grad = image_swin.requires_grad or any(p.requires_grad for p in self.prior.parameters())
        with nullcontext() if needs_grad else torch.no_grad():
            prior = self.prior(image_swin)
        def bottleneck(f):
            return self.bottleneck(f, prior[3]) if self.bottleneck is not None else f
        def skip(level, f, u):
            return self.skips[str(level)](f, prior[level-1], u) if str(level) in self.skips else f
        return self.unet.decode(features, bottleneck_transform=bottleneck, skip_transform=skip)

    def fusion_telemetry(self):
        modules = dict(self.skips.items())
        if self.bottleneck is not None:
            modules['4'] = self.bottleneck
        result = {}
        for level, module in modules.items():
            gradients = {name: float(p.grad.float().norm()) if p.grad is not None else None
                         for name, p in module.named_parameters()}
            result[level] = dict(module.telemetry, gradients=gradients)
        return result
