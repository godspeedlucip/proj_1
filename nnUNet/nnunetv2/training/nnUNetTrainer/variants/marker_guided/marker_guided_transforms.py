from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform


class RawOnlyIntensityTransform(BasicTransform):
    """Run an intensity transform on projection channel 0, preserving all priors."""

    def __init__(self, transform: BasicTransform):
        super().__init__()
        self.transform = transform

    def apply(self, data_dict: dict, **params) -> dict:
        image = data_dict.get("image")
        if image is None:
            return data_dict
        if image.ndim < 3 or image.shape[0] != 4:
            raise ValueError("raw-only augmentation requires four-channel CxHxW/CxDxHxW images")
        transformed = self.transform(image=image[:1].clone())["image"]
        result = image.clone()
        result[0:1] = transformed
        data_dict["image"] = result
        return data_dict


class ClampPriorChannelsTransform(BasicTransform):
    def apply(self, data_dict: dict, **params) -> dict:
        image = data_dict.get("image")
        if image is not None:
            if image.shape[0] != 4:
                raise ValueError("prior clamp expects four channels")
            image[1:] = image[1:].clamp_(0, 1)
        return data_dict


def get_marker_guided_training_transforms(
    patch_size: Union[np.ndarray, Tuple[int]],
    rotation_for_DA: RandomScalar,
    deep_supervision_scales: Union[List, Tuple, None],
    mirror_axes: Tuple[int, ...],
    do_dummy_2d_data_aug: bool,
    use_mask_for_norm: List[bool] = None,
    is_cascaded: bool = False,
    foreground_labels: Union[Tuple[int, ...], List[int]] = None,
    regions: Union[List, None] = None,
    ignore_label: int = None,
) -> BasicTransform:
    if len(patch_size) != 2 or do_dummy_2d_data_aug or is_cascaded:
        raise RuntimeError("Marker-guided augmentation currently supports non-cascaded 2D plans only")
    transforms = [SpatialTransform(
        patch_size, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
        p_rotation=0.2, rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4),
        p_synchronize_scaling_across_axes=1, bg_style_seg_sampling=False,
        mode_seg="nearest", border_mode_seg="constant", padding_value_seg=-1,
    )]
    transforms.append(ClampPriorChannelsTransform())
    raw_transforms = [
        (GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True), 0.1),
        (GaussianBlurTransform(blur_sigma=(0.5, 1.), synchronize_channels=False,
                               synchronize_axes=False, p_per_channel=1, benchmark=True), 0.2),
        (MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)),
                                           synchronize_channels=False, p_per_channel=1), 0.15),
        (ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True,
                           synchronize_channels=False, p_per_channel=1), 0.15),
        (SimulateLowResolutionTransform(scale=(0.5, 1), synchronize_channels=False,
                                        synchronize_axes=True, ignore_axes=None, allowed_channels=(0,),
                                        p_per_channel=1), 0.25),
        (GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=1,
                        synchronize_channels=False, p_per_channel=1, p_retain_stats=1), 0.1),
        (GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=0,
                        synchronize_channels=False, p_per_channel=1, p_retain_stats=1), 0.3),
    ]
    transforms.extend(RandomTransform(RawOnlyIntensityTransform(transform), apply_probability=p)
                      for transform, p in raw_transforms)
    if mirror_axes:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(MaskImageTransform(
            apply_to_channels=[i for i, apply in enumerate(use_mask_for_norm) if apply],
            channel_idx_in_seg=0, set_outside_to=0))
    transforms.append(RemoveLabelTansform(-1, 0))
    if regions is not None:
        from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
        transforms.append(ConvertSegmentationToRegionsTransform(
            regions=list(regions) + [ignore_label] if ignore_label is not None else regions, channel_in_seg=0))
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    return ComposeTransforms(transforms)
