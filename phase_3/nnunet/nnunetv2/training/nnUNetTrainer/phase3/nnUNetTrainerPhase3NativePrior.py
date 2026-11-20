"""Original nnU-Net training with only the Phase 3 frozen prior network added."""

from nnunetv2.phase3.network import build_phase3_network
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPhase3NativePrior(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return build_phase3_network(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision)
