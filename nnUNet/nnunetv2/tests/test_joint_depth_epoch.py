import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.nnUNetTrainerMarkerGuidedDepth import (
    depth_case_counts, nnUNetTrainerMarkerGuidedDepthSmoke, summarize_depth,
)
from nnunetv2.utilities.checkpoints import load_checkpoint_file


class TinyTrainer(nnUNetTrainerMarkerGuidedDepthSmoke):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        net = torch.nn.Module()
        net.backbone = torch.nn.Conv2d(4, 2, 1)
        net.depth_refiner = torch.nn.Conv2d(3, 1, 1)
        return net

    def _forward_loss(self, batch):
        logits = self.network.backbone(batch["data"])
        depth = self.network.depth_refiner(batch["data"][:, :3])
        return logits.square().mean() + depth.square().mean(), {}, None, None, None, None


def tensor(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


class JointDepthEpochTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text("{}")
        env = patch.dict(os.environ, {"nnUNet_preprocessed": directory.name, "nnUNet_results": directory.name})
        env.start()
        self.addCleanup(env.stop)

    def trainer(self):
        plans = {
            "continue_training": False, "dataset_name": "Dataset991_Tiny", "plans_name": "DepthCascadePlans",
            "joint_depth": {"manifest": str(self.manifest), "seed": 31, "target_scale": 2.0,
                            "init_seg_checkpoint": str(self.root / "not_needed.safetensors")},
            "configurations": {"2d": {"data_identifier": "2d", "batch_size": 1,
                                      "architecture": {"network_class_name": "TinyNet"}, "patch_size": [8, 8]}},
        }
        dataset = {"labels": {"background": 0, "stent": 1}, "channel_names": {str(i): str(i) for i in range(4)}}
        return TinyTrainer(plans, "2d", 0, dataset, torch.device("cpu"))

    def test_full_misses_count_in_macro_mae_and_invalid_canvas_is_excluded(self):
        first = depth_case_counts(tensor([0, 1, 1, 0]), tensor([0, 3, 4, 50]), tensor([1, 1, 0, 1]),
                                  tensor([2, 3, 0, 100]), tensor([1, 1, 1, 0]))
        missed = depth_case_counts(tensor([0, 0]), tensor([0, 0]), tensor([1, 0]), tensor([4, 0]), tensor([1, 1]))
        metrics = summarize_depth([first, missed], target_scale=2)
        self.assertEqual(metrics["fg_mae_macro"], 2.5)
        self.assertEqual(metrics["fg_mae_pixel"], 2.0)
        self.assertEqual(metrics["fg_mae_raw_macro"], 5.0)
        self.assertEqual(metrics["empty_prediction_fraction"], 0.5)
        self.assertEqual(metrics["outside_prediction_nonzero"], 0)
        self.assertEqual(metrics["background_sum"], 4)

    def test_latest_resumes_rng_selection_and_optimizer_after_tail_update(self):
        src = self.trainer()
        src.initialize()
        src.num_iterations_per_epoch = 3
        batch = {"data": torch.ones(1, 4, 8, 8), "keys": ["sample"]}
        initial = src.network.depth_refiner.weight.detach().clone()
        src.train_step(batch)
        torch.testing.assert_close(src.network.depth_refiner.weight, initial)
        src.train_step(batch)
        after_pair = src.network.depth_refiner.weight.detach().clone()
        self.assertFalse(torch.equal(after_pair, initial))
        src.train_step(batch)
        self.assertFalse(torch.equal(src.network.depth_refiner.weight, after_pair))
        self.assertEqual(src.optimizer_updates, 2)
        src.best_fg_mae = 0.125
        dest = str(self.root / "checkpoint_latest.safetensors")
        src.save_checkpoint(dest)
        expected_random = torch.rand(3)
        restored = self.trainer()
        restored.load_checkpoint(dest)
        self.assertEqual(restored.current_epoch, 1)
        self.assertEqual(restored.best_fg_mae, 0.125)
        self.assertEqual(restored.optimizer_updates, 2)
        self.assertTrue(restored.optimizer.state)
        torch.testing.assert_close(torch.rand(3), expected_random)
        torch.testing.assert_close(restored.network.depth_refiner.weight, src.network.depth_refiner.weight)

    def test_best_weights_follow_depth_mae_when_dice_improves(self):
        trainer = self.trainer()
        trainer.initialize()
        trainer.num_epochs = 3
        def end_epoch(pred, depth):
            trainer.on_epoch_start()
            trainer.logger.log("lrs", 1e-5, trainer.current_epoch)
            trainer.logger.log("train_losses", 0.5, trainer.current_epoch)
            stats = depth_case_counts(tensor(pred), tensor(depth), tensor([1, 0]), tensor([1, 0]), tensor([1, 1]))
            trainer.on_validation_epoch_end([{"loss": 0.5, "case": stats}])
            trainer.on_epoch_end()
        # First model has imperfect Dice and exact foreground depth. Second has perfect Dice and worse depth.
        end_epoch([1, 1], [1, 1])
        best_weight = trainer.network.backbone.weight.detach().clone()
        with torch.no_grad():
            trainer.network.backbone.weight.add_(2)
        end_epoch([1, 0], [3, 0])
        def checkpoint(name):
            return load_checkpoint_file(str(Path(trainer.output_folder) / f"checkpoint_{name}.safetensors"))
        best, latest = checkpoint("best"), checkpoint("latest")
        self.assertEqual(best["current_epoch"], 1)
        self.assertEqual(latest["current_epoch"], 2)
        self.assertEqual(latest["joint_depth"]["best_fg_mae"], 0)
        torch.testing.assert_close(best["network_weights"]["backbone.weight"], best_weight)
        torch.testing.assert_close(latest["network_weights"]["backbone.weight"], best_weight + 2)

    def test_changed_manifest_after_preparation_is_rejected(self):
        trainer = self.trainer()
        trainer.joint["manifest_sha256"] = "definitely-not-the-hash-of-the-current-manifest"
        with self.assertRaisesRegex(ValueError, "manifest"):
            trainer._contract()
