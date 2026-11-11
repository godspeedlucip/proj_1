"""Synchronous FDK mirror, ignore and nearest deep-supervision transforms."""

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform


def build_transforms(*, training: bool, deep_supervision: bool) -> BasicTransform:
    if type(training) is not bool or type(deep_supervision) is not bool:
        raise ValueError("Training and deep supervision switches must be boolean")
    transforms = []
    if training:
        transforms.append(MirrorTransform(allowed_axes=(0, 1, 2)))
    transforms.append(RemoveLabelTansform(-1, 2))
    if deep_supervision:
        transforms.append(DownsampleSegForDSTransform(ds_scales=(1.0, 0.5, 0.25, 0.125)))
    return ComposeTransforms(transforms)
