from __future__ import annotations

from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_network import build_marker_guided_network
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_transforms import get_marker_guided_training_transforms
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerMarkerGuidedFeatureFusion(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        if num_input_channels != 4:
            raise RuntimeError(f"Marker-guided E5-U expects exactly 4 input channels; received {num_input_channels}")
        if len(configuration_manager.patch_size) != 2:
            raise RuntimeError("Marker-guided E5-U supports only 2D plans")
        if configuration_manager.network_arch_class_name.split(".")[-1] != "PlainConvUNet":
            raise RuntimeError("Marker-guided E5-U supports only PlainConvUNet architecture plans")
        return build_marker_guided_network(configuration_manager, num_input_channels,
                                           num_output_channels, enable_deep_supervision)

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                               do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                               foreground_labels=None, regions=None, ignore_label=None):
        return get_marker_guided_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label)


class nnUNetTrainerMarkerGuidedSmoke(nnUNetTrainerMarkerGuidedFeatureFusion):
    """Short-budget Trainer for software smoke tests; architecture and pipeline stay identical."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
