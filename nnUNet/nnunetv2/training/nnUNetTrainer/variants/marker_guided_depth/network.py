"""E5-U segmentation followed by continuous depth regression on its predicted support."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..marker_guided.fusion import MarkerFusion
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_network import (
    build_marker_guided_network,
)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(inplace=True),
    )


class DepthRefiner(nn.Module):
    """Five-resolution 2D U-Net for image, mask and confidence inside the support."""

    def __init__(self):
        super().__init__()
        widths = (16, 32, 64, 128, 256)
        self.input_block = _conv_block(3, widths[0])
        self.encoder = nn.ModuleList(
            _conv_block(in_ch, out_ch) for in_ch, out_ch in zip(widths[:-1], widths[1:])
        )
        self.decoder = nn.ModuleList(
            _conv_block(widths[i] + widths[i - 1], widths[i - 1]) for i in reversed(range(1, len(widths)))
        )
        self.head = nn.Conv2d(widths[0], 1, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # Initial full-frame smoke checkpoints stored the input block as encoder.0.
        # Copy just these keys before remapping so adjacent stages never overwrite each other.
        if prefix + "encoder.4.0.weight" in state_dict and prefix + "input_block.0.weight" not in state_dict:
            encoder_prefix = prefix + "encoder."
            previous = {key: state_dict.pop(key) for key in list(state_dict) if key.startswith(encoder_prefix)}
            for key, value in previous.items():
                stage, layer = key[len(encoder_prefix):].split(".", 1)
                name = "input_block" if stage == "0" else f"encoder.{int(stage) - 1}"
                state_dict[prefix + name + "." + layer] = value
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, inputs: Tensor) -> Tensor:
        skips = []
        x = self.input_block(inputs)
        for block in self.encoder:
            skips.append(x)
            x = block(F.max_pool2d(x, 2))
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


class MarkerGuidedNNUNetDepthCascade(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.depth_refiner = DepthRefiner()

    @property
    def decoder(self):
        # Expose the native deep-supervision switch without registering another copy of its weights.
        return self.backbone.decoder

    def set_deep_supervision_enabled(self, enabled: bool):
        self.backbone.set_deep_supervision_enabled(enabled)

    def forward(self, inputs: Tensor, *, return_depth: bool = False):
        segmentation = self.backbone(inputs)
        if not return_depth:
            return segmentation
        logits_ds = segmentation if isinstance(segmentation, (list, tuple)) else [segmentation]
        logits = logits_ds[0]
        probability = logits.softmax(dim=1, dtype=torch.float32)[:, 1:2]
        hard_mask = (probability.detach() > 0.5).float()
        depth_inputs = torch.cat((inputs[:, :1] * hard_mask, hard_mask, probability * hard_mask), dim=1)
        conditional = F.softplus(self.depth_refiner(depth_inputs).float())
        return {
            "seg_logits": logits,
            "seg_logits_ds": logits_ds,
            "mask_probability": probability,
            "predicted_mask": hard_mask,
            "conditional_depth": conditional,
            "depth_normalized": conditional * hard_mask,
        }


def build_joint_network(config_manager, num_input_channels: int, num_output_channels: int,
                        enable_deep_supervision: bool = True,
                        fusion: MarkerFusion = MarkerFusion()) -> MarkerGuidedNNUNetDepthCascade:
    if num_output_channels != 2:
        raise ValueError("Depth cascade requires binary segmentation with two softmax logits")
    backbone = build_marker_guided_network(config_manager, num_input_channels, num_output_channels,
                                           enable_deep_supervision, fusion=fusion)
    return MarkerGuidedNNUNetDepthCascade(backbone)


def initialize_from_legacy(network, weights, *, joint):
    """Warm-start an expanded model from the legacy two-skip backbone or depth cascade."""
    backbone = network.backbone
    target = network if joint else backbone
    if backbone.fusion.mode == "skip":
        target.load_state_dict(weights, strict=True)
        return
    prefix = "backbone." if joint else ""
    added = [prefix + "axial_projections."]
    for level in backbone.fusion_stage_indices[2:]:
        added.extend((prefix + f"adapters.{level}.", prefix + f"prior_encoder.projections.{level}."))
    # Only fill the new modules. Strict loading still audits every legacy parameter
    # and executes the existing depth refiner migration for older smoke checkpoints.
    initial = {key: value for key, value in target.state_dict().items() if key.startswith(tuple(added))}
    if initial.keys() & weights.keys():
        raise ValueError("Warm start expects a legacy two-skip checkpoint; use resume for a trained variant")
    target.load_state_dict({**initial, **weights}, strict=True)


def _segmentation_loss(logits_ds, mask: Tensor, valid_pixels: Tensor) -> Tensor:
    """Native two-class CE minus foreground batch Dice, excluding padded pixels at every scale."""
    supervised = logits_ds[:-1] if len(logits_ds) > 1 else logits_ds
    weights = [0.5 ** i for i in range(len(supervised))]
    result = mask.new_zeros((), dtype=torch.float32)
    for weight, logits in zip(weights, supervised):
        shape = logits.shape[-2:]
        target, valid = mask, valid_pixels
        if shape != mask.shape[-2:]:
            target = F.interpolate(mask.float(), size=shape, mode="nearest")
            # Only use a downsampled cell when its entire footprint was on the original detector.
            valid = F.adaptive_max_pool2d((~valid_pixels).float(), shape) == 0

        target = torch.where(valid, target, 0).long()
        logits = logits.float()
        ce_map = F.cross_entropy(logits, target[:, 0], reduction="none")
        ce = (ce_map * valid[:, 0]).sum() / valid.sum().clamp_min(1)
        foreground = logits.softmax(dim=1)[:, 1:2] * valid
        intersection = (foreground * target).sum()
        dice = (2 * intersection + 1e-5) / (foreground.sum() + target.sum() + 1e-5)
        result = result + weight * (ce - dice * valid.any())
    return result / sum(weights)


def _mean_in_region(values: Tensor, region: Tensor) -> tuple[Tensor, Tensor]:
    areas = region.sum(dim=(1, 2, 3))
    means = (values * region).sum(dim=(1, 2, 3)) / areas.clamp_min(1)
    return means.mean(), (areas == 0).sum()


def joint_loss(outputs: dict, mask: Tensor | list[Tensor], depth: Tensor, valid_pixels: Tensor):
    """Full-resolution depth on all GT foreground (including misses) and valid background."""
    if isinstance(mask, (list, tuple)):
        mask = mask[0]
    prediction = outputs["depth_normalized"].float()
    if mask.shape != prediction.shape or depth.shape != prediction.shape or valid_pixels.shape != prediction.shape:
        raise ValueError("Full-resolution mask, depth target and validity must match the [B,1,H,W] depth prediction")
    valid = valid_pixels.bool()
    foreground = mask.bool()
    seg = _segmentation_loss(outputs["seg_logits_ds"], mask, valid)
    error = F.smooth_l1_loss(prediction, depth.float(), reduction="none", beta=0.05)
    fg, empty_fg = _mean_in_region(error, valid & foreground)
    bg, empty_bg = _mean_in_region(prediction.abs(), valid & ~foreground)
    depth_loss = fg + 0.1 * bg
    return seg + depth_loss, {
        "seg": seg, "fg": fg, "bg": bg, "depth": depth_loss, "empty_fg": empty_fg, "empty_bg": empty_bg,
    }
