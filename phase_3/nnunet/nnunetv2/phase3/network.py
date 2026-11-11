"""Native PlainConvUNet with explicit Phase 3 feature contracts."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from contextlib import nullcontext

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .config import Phase3Config, resolve_config
from .initialization import named_rng
from .planning import validate_phase3_plans
from .fusion import BottleneckFusion, DecoderGuidedFusion
from .swin_prior import SwinPriorEncoder
from .tubular import TubularEnhancement


class Phase3Network(nn.Module):
    def __init__(self, plans_manager, configuration_manager, config: Phase3Config,
                 enable_deep_supervision: bool = True):
        super().__init__()
        self.config = config
        with named_rng(config.seed, "cnn"):
            native = get_network_from_plans(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                input_channels=1, output_channels=2, allow_init=True,
                deep_supervision=enable_deep_supervision,
            )
        self.encoder = native.encoder
        self.decoder = native.decoder
        self.prior = None
        self.bottleneck = None
        self.skip_fusions = nn.ModuleDict()
        self.tubular = None
        if config.swin.enabled:
            with named_rng(config.seed, "prior"):
                self.prior = SwinPriorEncoder(execute_through=config.swin.execute_through)
            self.prior.requires_grad_(False).eval()
            channels = tuple(config.cnn.channels)
            if config.fusion.bottleneck_enabled:
                with named_rng(config.seed, "fusion.scale4"):
                    self.bottleneck = BottleneckFusion(384, channels[4])
            for level, swin_channels in ((2, 96), (3, 192)):
                with named_rng(config.seed, f"fusion.scale{level}"):
                    self.skip_fusions[str(level)] = DecoderGuidedFusion(
                        swin_channels, channels[level],
                        decoder_conditioning=config.fusion.decoder_conditioning)
            for module in ([self.bottleneck] if self.bottleneck is not None else []) + list(self.skip_fusions.values()):
                module.alpha.data.fill_(config.fusion.alpha_init)
        if config.tubular.enabled:
            with named_rng(config.seed, "tubular"):
                self.tubular = TubularEnhancement(num_classes=2)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.prior is not None:
            self.prior.eval()
        return self

    def encode_cnn(self, data: Tensor) -> list[Tensor]:
        if data.ndim != 5 or data.shape[1] != 2:
            raise ValueError("Phase 3 network requires [B,2,D,H,W] input")
        return self.encoder(data[:, :1])

    def _prior_features(self, data: Tensor) -> dict[int, Tensor]:
        if self.prior is None:
            return {}
        if not torch.isfinite(data[:, 1:2]).all():
            raise ValueError("Swin input must be finite")
        needs_grad = data.requires_grad or any(p.requires_grad for p in self.prior.parameters())
        with nullcontext() if needs_grad else torch.no_grad():
            return self.prior(data[:, 1:2])

    def _postprocess_final(self, logits: Tensor) -> Tensor:
        return self.tubular(logits) if self.tubular is not None else logits

    def decode_cnn(self, skips: list[Tensor], prior: dict[int, Tensor] | None = None) -> Tensor | list[Tensor]:
        prior = prior or {}
        current = self.bottleneck(skips[-1], prior[3]) if self.bottleneck is not None else skips[-1]
        outputs = []
        for stage_index, stage in enumerate(self.decoder.stages):
            up = self.decoder.transpconvs[stage_index](current)
            level = 3 - stage_index
            skip = skips[-(stage_index + 2)]
            if str(level) in self.skip_fusions:
                skip = self.skip_fusions[str(level)](skip, prior[level - 1], up)
            current = torch.cat((up, skip), dim=1)
            current = stage(current)
            if self.decoder.deep_supervision:
                outputs.append(self.decoder.seg_layers[stage_index](current))
            elif stage_index == len(self.decoder.stages) - 1:
                outputs.append(self.decoder.seg_layers[-1](current))
        outputs.reverse()
        outputs[0] = self._postprocess_final(outputs[0])
        return outputs if self.decoder.deep_supervision else outputs[0]

    def forward(self, data: Tensor) -> Tensor | list[Tensor]:
        skips = self.encode_cnn(data)
        return self.decode_cnn(skips, self._prior_features(data))

    def fusion_telemetry(self) -> dict:
        modules = dict(self.skip_fusions.items())
        if self.bottleneck is not None:
            modules["4"] = self.bottleneck
        return {name: dict(module.telemetry, gradients={key: (
            float(param.grad.detach().float().norm()) if param.grad is not None else None)
            for key, param in module.named_parameters()}) for name, module in modules.items()}

    def set_deep_supervision(self, enabled: bool) -> None:
        self.decoder.deep_supervision = bool(enabled)

    def compute_conv_feature_map_size(self, input_size):
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def build_phase3_network(plans_manager, configuration_manager,
                         num_input_channels: int, num_output_channels: int,
                         enable_deep_supervision: bool = True) -> Phase3Network:
    validate_phase3_plans(plans_manager.plans)
    if (num_input_channels, num_output_channels) != (2, 2):
        raise ValueError("Phase 3 requires two input views and two class logits")
    if configuration_manager.data_identifier != plans_manager.get_configuration("3d_fullres").data_identifier:
        raise ValueError("Phase 3 requires its own 3d_fullres configuration")
    config = resolve_config(plans_manager.plans["phase3"]["config"])
    return Phase3Network(plans_manager, configuration_manager, config, enable_deep_supervision)
