from __future__ import annotations

import pydoc
from typing import Sequence

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from .marker_guided_encoder import MarkerGuidedPriorEncoder
from .residual_prior_adapter import ResidualPriorAdapter


def _resolve_architecture_kwargs(configuration_manager) -> dict:
    kwargs = dict(configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        value = kwargs.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            kwargs[key] = [pydoc.locate(item) if isinstance(item, str) else item for item in value]
        elif isinstance(value, str):
            kwargs[key] = pydoc.locate(value)
        if kwargs[key] is None:
            raise ImportError(f"cannot resolve nnU-Net architecture argument {key!r}: {value!r}")
    return kwargs


class MarkerGuidedPlainConvUNet(nn.Module):
    """Plans-configured 2D PlainConvUNet with a soft Marker prior branch."""

    def __init__(self, input_channels: int, num_classes: int, deep_supervision: bool,
                 fusion_stage_indices: Sequence[int] | None = None, prior_channels: int = 2,
                 gate_channel: int = 3, prior_feature_ratio: float = 0.25, **plain_conv_unet_kwargs):
        super().__init__()
        if input_channels != 4:
            raise ValueError("Marker-guided E5-U requires four input channels")
        if prior_channels != 2 or gate_channel != 3:
            raise ValueError("E5-U expects prior channels 1-2 and validity gate channel 3")
        kwargs = dict(plain_conv_unet_kwargs)
        features = kwargs.get("features_per_stage")
        n_stages = kwargs.get("n_stages")
        if features is None or n_stages is None:
            raise ValueError("PlainConvUNet plans must define n_stages and features_per_stage")
        if isinstance(features, int):
            features = [features] * n_stages
        else:
            features = list(features)
        if len(features) != n_stages:
            raise ValueError("features_per_stage length must match n_stages")
        kwargs["features_per_stage"] = features
        image_network = PlainConvUNet(input_channels=1, num_classes=num_classes,
                                      deep_supervision=deep_supervision, **kwargs)
        self.encoder = image_network.encoder
        self.decoder = image_network.decoder
        self.feature_channels = features
        self.fusion_stage_indices = list(fusion_stage_indices) if fusion_stage_indices is not None else list(range(min(2, n_stages)))
        if not self.fusion_stage_indices or any(i < 0 or i >= n_stages for i in self.fusion_stage_indices):
            raise ValueError("fusion_stage_indices must select valid encoder stages")
        self.prior_encoder = MarkerGuidedPriorEncoder(prior_channels, features,
                                                      self.fusion_stage_indices, prior_feature_ratio)
        self.adapters = nn.ModuleDict({str(i): ResidualPriorAdapter(features[i], features[i])
                                       for i in self.fusion_stage_indices})
        self.gate_channel = gate_channel

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"expected input [N,4,H,W], got {tuple(x.shape)}")
        skips = self.encoder(x[:, :1])
        priors = self.prior_encoder(x[:, 1:3], [tuple(skip.shape[-2:]) for skip in skips])
        gate = x[:, self.gate_channel:self.gate_channel + 1]
        for index in self.fusion_stage_indices:
            skips[index] = self.adapters[str(index)](skips[index], priors[index], gate)
        return self.decoder(skips)

    def set_deep_supervision_enabled(self, enabled: bool):
        self.decoder.deep_supervision = enabled


def build_marker_guided_network(configuration_manager, num_input_channels: int,
                                num_output_channels: int, enable_deep_supervision: bool = True):
    if configuration_manager.network_arch_class_name.split(".")[-1] != "PlainConvUNet":
        raise RuntimeError("Marker-guided Trainer supports only 2D PlainConvUNet plans")
    kwargs = _resolve_architecture_kwargs(configuration_manager)
    if kwargs.get("conv_op") is not torch.nn.Conv2d:
        raise RuntimeError("Marker-guided Trainer requires 2D Conv2d plans")
    network = MarkerGuidedPlainConvUNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **kwargs,
    )
    network.apply(PlainConvUNet.initialize)
    for adapter in network.adapters.values():
        nn.init.zeros_(adapter.attention[-1].weight)
        nn.init.zeros_(adapter.attention[-1].bias)
    return network
