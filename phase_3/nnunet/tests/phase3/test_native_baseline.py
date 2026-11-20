"""The plain baseline must use the original nnU-Net trainer."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.planning import prepare_training_fold
from nnunetv2.phase3.losses import Phase3CompositeLoss
from nnunetv2.phase3.prior_weights import prepare_prior_artifact
from nnunetv2.phase3.swin_prior import SwinPriorEncoder
from .fixtures import make_config, synthetic_dataset


def test_prior_variant_keeps_stock_optimizer_step_and_augmentation():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3NativePrior import (
        nnUNetTrainerPhase3NativePrior,
    )

    assert nnUNetTrainerPhase3NativePrior.train_step is nnUNetTrainer.train_step
    assert nnUNetTrainerPhase3NativePrior.validation_step is nnUNetTrainer.validation_step
    assert nnUNetTrainerPhase3NativePrior.get_dataloaders is nnUNetTrainer.get_dataloaders
    assert nnUNetTrainerPhase3NativePrior._build_loss is nnUNetTrainer._build_loss


def test_full_variant_adds_only_network_and_loss():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3NativeFull import (
        nnUNetTrainerPhase3NativeFull,
    )

    assert nnUNetTrainerPhase3NativeFull.train_step is nnUNetTrainer.train_step
    assert nnUNetTrainerPhase3NativeFull.validation_step is nnUNetTrainer.validation_step
    assert nnUNetTrainerPhase3NativeFull.get_dataloaders is nnUNetTrainer.get_dataloaders
    assert nnUNetTrainerPhase3NativeFull.configure_optimizers is nnUNetTrainer.configure_optimizers
    fake = SimpleNamespace(plans_manager=SimpleNamespace(plans={"phase3": {"config": make_config()}}))
    assert isinstance(nnUNetTrainerPhase3NativeFull._build_loss(fake), Phase3CompositeLoss)


def test_short_native_baseline_uses_stock_trainer_and_both_input_channels(tmp_path):
    manifest, splits = synthetic_dataset(tmp_path / "input")
    prepared = prepare_raw_fold(load_manifest(manifest, require_labels=True), load_splits(splits),
                                source_fold=0, dataset_id=916, output=tmp_path / "prepared")
    config = resolve_config(make_config(
        run_kind="smoke", cnn__patch_size=[32, 32, 32], cnn__batch_size=1,
        training__max_optimizer_steps=2, training__updates_per_epoch=1,
        swin__enabled=False, fusion__enabled=False, fusion__bottleneck_enabled=False,
        tubular__enabled=False, projection__enabled=False, projection__weight=0.0))
    completed = prepare_training_fold(prepared, config)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["nnUNet_n_proc_DA"] = "0"
    command = [sys.executable, "-m", "nnunetv2.phase3.native_baseline",
               "--prepared", str(tmp_path / "prepared/prepared.json"),
               "--mode", "plain",
               "--device", "cpu", "--epochs", "1", "--steps-per-epoch", "1",
               "--validation-steps", "1"]
    result = subprocess.run(command, env=env, text=True, capture_output=True,
                            timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    folder = (completed.results_dir
              / f"nnUNetTrainer__{completed.plans_name}__3d_fullres/fold_0")
    checkpoint = torch.load(folder / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    assert checkpoint["trainer_name"] == "nnUNetTrainer"
    assert checkpoint["current_epoch"] == 1
    assert any(weight.ndim == 5 and weight.shape[1] == 2
               for name, weight in checkpoint["network_weights"].items()
               if "encoder" in name and "weight" in name)
    assert len(checkpoint["logging"]["train_losses"]) == 1
    assert len(checkpoint["logging"]["val_losses"]) == 1


def test_full_variant_runs_stock_step_with_added_modules(tmp_path):
    manifest, splits = synthetic_dataset(tmp_path / "input")
    prepared = prepare_raw_fold(load_manifest(manifest, require_labels=True), load_splits(splits),
                                source_fold=0, dataset_id=917, output=tmp_path / "prepared")
    config = resolve_config(make_config(
        run_kind="smoke", cnn__patch_size=[32, 32, 32], cnn__batch_size=1,
        training__max_optimizer_steps=2, training__updates_per_epoch=1))
    completed = prepare_training_fold(prepared, config)
    encoder = SwinPriorEncoder(execute_through="layers3")
    weights = tmp_path / "weights.pt"
    torch.save({"state_dict": {f"module.swin_vit.{key}": value
                                for key, value in encoder.swinViT.state_dict().items()}}, weights)
    artifact = prepare_prior_artifact(weights, tmp_path / "artifact.json")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["nnUNet_n_proc_DA"] = "0"
    command = [sys.executable, "-m", "nnunetv2.phase3.native_baseline",
               "--prepared", str(tmp_path / "prepared/prepared.json"),
               "--mode", "full", "--prior-artifact", str(artifact),
               "--device", "cpu", "--epochs", "1", "--steps-per-epoch", "1",
               "--validation-steps", "1"]
    result = subprocess.run(command, env=env, text=True, capture_output=True,
                            timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    folder = (completed.results_dir
              / f"nnUNetTrainerPhase3NativeFull__{completed.plans_name}__3d_fullres/fold_0")
    checkpoint = torch.load(folder / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    assert checkpoint["trainer_name"] == "nnUNetTrainerPhase3NativeFull"
    assert checkpoint["current_epoch"] == 1
    resume_command = [arg for arg in command if arg not in ("--prior-artifact", str(artifact))]
    resume_command[resume_command.index("--epochs") + 1] = "2"
    resume_command.append("--resume")
    resumed = subprocess.run(resume_command, env=env, text=True, capture_output=True,
                             timeout=180, check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    resumed_checkpoint = torch.load(folder / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    assert resumed_checkpoint["current_epoch"] == 2
