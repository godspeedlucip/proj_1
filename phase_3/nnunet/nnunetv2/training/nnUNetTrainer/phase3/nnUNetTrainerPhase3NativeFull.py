"""Original nnU-Net training loop with Phase 3 network and loss only."""

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.losses import Phase3CompositeLoss
from nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3NativePrior import (
    nnUNetTrainerPhase3NativePrior,
)


class nnUNetTrainerPhase3NativeFull(nnUNetTrainerPhase3NativePrior):
    def _build_loss(self):
        config = resolve_config(self.plans_manager.plans["phase3"]["config"])
        return Phase3CompositeLoss(config)
