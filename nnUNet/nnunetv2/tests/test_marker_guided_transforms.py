from __future__ import annotations

import torch
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform

from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_transforms import RawOnlyIntensityTransform


def test_raw_only_intensity_keeps_prior_channels_unchanged():
    image = torch.stack([torch.full((12, 12), float(i + 1)) for i in range(4)])
    transform = RawOnlyIntensityTransform(MultiplicativeBrightnessTransform(
        multiplier_range=(2.0, 2.0), synchronize_channels=False, p_per_channel=1))
    result = transform(image=image.clone())["image"]
    assert torch.all(result[0] == 2)  # projection values are multiplied by two
    assert torch.equal(result[1:], image[1:])


def test_spatial_transform_keeps_image_channels_and_mask_aligned():
    torch.manual_seed(10)
    base = torch.zeros(16, 16)
    base[2:5, 3:7] = 1
    image = torch.stack([base.clone() for _ in range(4)])
    result = MirrorTransform(allowed_axes=(0, 1))(image=image, segmentation=base[None].clone())
    transformed = result["image"]
    assert torch.equal(transformed[0], transformed[1])
    assert torch.equal(transformed[1], transformed[2])
    assert torch.equal(transformed[2], transformed[3])
    assert torch.equal(transformed[0], result["segmentation"][0])
