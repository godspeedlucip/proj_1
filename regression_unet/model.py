"""Build the standalone mask/depth cascade or historical regression models."""

from __future__ import annotations

import json
import pydoc
from pathlib import Path

import torch

from .cascade_model import MaskThenDepthUNet


def load_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        configuration = plan["configurations"]["2d"]
        architecture = configuration["architecture"]
    except KeyError as exc:
        raise ValueError("plan needs a 2d PlainConvUNet configuration") from exc
    if architecture.get("network_class_name", "").split(".")[-1] != "PlainConvUNet":
        raise ValueError("plan 2d architecture must be PlainConvUNet")
    if len(configuration.get("patch_size", [])) != 2:
        raise ValueError("plan needs a two-dimensional patch_size")
    return plan


def build_unet(plan: dict, input_channels: int = 4,
               model_family: str = "marker_fusion") -> torch.nn.Module:
    config = plan["configurations"]["2d"]
    architecture = config["architecture"]
    if architecture["network_class_name"].split(".")[-1] != "PlainConvUNet":
        raise ValueError("plan 2d architecture must be PlainConvUNet")
    if model_family == "mask_then_depth":
        if input_channels != 4:
            raise ValueError("cascade requires four input channels")
        source = architecture["arch_kwargs"]["features_per_stage"]
        stages = min(5, architecture["arch_kwargs"]["n_stages"])
        mask_widths = tuple(min(int(value), 320) for value in source[:stages])
        depth_widths = tuple(max(8, min(256, round(value * 0.75 / 8) * 8))
                             for value in mask_widths)
        return MaskThenDepthUNet(mask_widths, depth_widths)
    if model_family == "marker_fusion":
        from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_network import build_marker_guided_network
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        configuration = PlansManager(plan).get_configuration("2d")
        return build_marker_guided_network(configuration, input_channels, 1,
                                           enable_deep_supervision=False)
    if model_family != "plain":
        raise ValueError(f"unsupported model family: {model_family}")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    kwargs = dict(architecture["arch_kwargs"])
    for name in architecture.get("_kw_requires_import", []):
        if kwargs.get(name) is not None:
            kwargs[name] = pydoc.locate(kwargs[name])
            if kwargs[name] is None:
                raise ImportError(f"cannot import architecture argument {name}")
    if kwargs.get("conv_op") is not torch.nn.Conv2d:
        raise ValueError("plan must use Conv2d")
    return PlainConvUNet(input_channels=input_channels, num_classes=1,
                         deep_supervision=False, **kwargs)
