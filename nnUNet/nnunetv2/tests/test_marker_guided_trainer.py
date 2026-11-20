import os
import tempfile
import unittest
from unittest import mock

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_trainer import (
    nnUNetTrainerMarkerGuidedFeatureFusion,
    nnUNetTrainerMarkerGuidedSmoke,
)


class TestMarkerGuidedSmokeTrainer(unittest.TestCase):
    def test_initializes_and_keeps_short_budget(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
            "nnUNet_preprocessed": os.path.join(tmp, "preprocessed"),
            "nnUNet_results": os.path.join(tmp, "results"),
        }):
            plans = {
                "continue_training": False,
                "dataset_name": "Dataset999_MarkerGuidedTest",
                "plans_name": "nnUNetPlans",
                "configurations": {
                    "2d": {
                        "data_identifier": "nnUNetPlans_2d",
                        "architecture": {"network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet"},
                    }
                },
            }
            trainer = nnUNetTrainerMarkerGuidedSmoke(
                plans=plans,
                configuration="2d",
                fold=0,
                dataset_json={"labels": {"background": 0, "foreground": 1}},
                device=torch.device("cpu"),
            )

            self.assertEqual((trainer.num_epochs, trainer.num_iterations_per_epoch,
                              trainer.num_val_iterations_per_epoch), (2, 2, 1))
            self.assertEqual(trainer.my_init_kwargs["configuration"], "2d")

    def test_epoch_end_writes_training_curves_and_runs_real_monitor(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
            "nnUNet_preprocessed": os.path.join(tmp, "preprocessed"),
            "nnUNet_results": os.path.join(tmp, "results"),
        }):
            plans = {
                "continue_training": False,
                "dataset_name": "Dataset999_MarkerGuidedTest",
                "plans_name": "nnUNetPlans",
                "configurations": {"2d": {"data_identifier": "nnUNetPlans_2d", "architecture": {
                    "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet"}}},
            }
            trainer = nnUNetTrainerMarkerGuidedFeatureFusion(
                plans, "2d", 0, {"labels": {"background": 0, "foreground": 1}}, torch.device("cpu"))
            trainer.logger.log("train_losses", 0.2, 0)
            trainer.logger.log("val_losses", 0.1, 0)
            trainer.logger.log("mean_fg_dice", 0.4, 0)
            trainer.logger.log("lrs", 0.001, 0)
            trainer._real_monitor = object()
            with SummaryWriter(os.path.join(tmp, "tensorboard")) as writer:
                trainer._tb_writer = writer
                def end_epoch(instance):
                    instance.current_epoch += 1
                with mock.patch.object(nnUNetTrainer, "on_epoch_end", end_epoch), \
                     mock.patch.object(nnUNetTrainerMarkerGuidedFeatureFusion, "_run_real_inference",
                                       create=True) as real_run:
                    trainer.on_epoch_end()
                    real_run.assert_called_once_with(1)
            events = EventAccumulator(os.path.join(tmp, "tensorboard")).Reload()
            self.assertAlmostEqual(events.Scalars("train/loss")[0].value, 0.2, places=5)
            self.assertAlmostEqual(events.Scalars("valid/loss")[0].value, 0.1, places=5)
            self.assertAlmostEqual(events.Scalars("valid/pseudo_dice")[0].value, 0.4, places=5)
            self.assertEqual(events.Scalars("train/lr")[0].step, 1)
