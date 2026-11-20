import copy
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.planning import make_phase3_plans, prepare_training_fold
from nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3 import nnUNetTrainerPhase3
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from .fixtures import make_config, synthetic_dataset


def _trainer(tmp_path, monkeypatch, *, projection=False, deep_supervision=True,
             foreground_probability=0.33):
    torch.set_num_threads(2)
    mp, sp = synthetic_dataset(tmp_path / "input")
    prepared = prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                                source_fold=0, dataset_id=912, output=tmp_path / "fold")
    config = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                        cnn__batch_size=1, training__max_optimizer_steps=4,
                                        training__updates_per_epoch=2, swin__enabled=False,
                                        swin__execute_through="layers2", fusion__enabled=False,
                                        fusion__bottleneck_enabled=False, tubular__enabled=False,
                                        projection__enabled=projection,
                                        projection__weight=.1 if projection else 0.0,
                                        training__foreground_probability=foreground_probability,
                                        deep_supervision__enabled=deep_supervision))
    completed = prepare_training_fold(prepared, config)
    monkeypatch.setenv("nnUNet_raw", str(completed.raw_root))
    monkeypatch.setenv("nnUNet_preprocessed", str(completed.preprocessed_root))
    monkeypatch.setenv("nnUNet_results", str(completed.results_root))
    plans = make_phase3_plans(completed, config)
    plans["continue_training"] = False
    dataset_json = json.loads((completed.raw_dir / "dataset.json").read_text(encoding="utf-8"))
    return nnUNetTrainerPhase3(plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu"))


def test_trainer_is_discoverable_and_rejects_wrong_fold(tmp_path, monkeypatch):
    assert recursive_find_trainer_class_by_name("nnUNetTrainerPhase3") is nnUNetTrainerPhase3
    trainer = _trainer(tmp_path, monkeypatch)
    assert trainer.num_epochs == 2 and trainer.num_iterations_per_epoch == 2
    assert trainer.num_val_iterations_per_epoch == 1
    assert trainer._do_i_compile() is False
    assert set(trainer.do_split()[0]) == {"case_0", "case_1"}
    bad = copy.deepcopy(trainer.plans_manager.plans)
    bad["continue_training"] = False
    with pytest.raises(ValueError, match="fold"):
        nnUNetTrainerPhase3(bad, "3d_fullres", "all", trainer.dataset_json,
                           device=torch.device("cpu"))
    (tmp_path / "fold" / "preprocessed" / trainer.plans_manager.dataset_name / "splits_final.json").unlink()
    with pytest.raises(FileNotFoundError):
        trainer.do_split()


def test_configured_foreground_probability_reaches_both_loaders(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, foreground_probability=0.9)
    trainer.initialize()
    train, validation = trainer.get_dataloaders()
    assert train.oversample_foreground_percent == pytest.approx(0.9)
    assert validation.oversample_foreground_percent == pytest.approx(0.9)


def test_cpu_loader_batch_performs_real_optimizer_step(tmp_path, monkeypatch, capsys):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.initialize()
    train, validation = trainer.get_dataloaders()
    batch = next(train)
    assert batch["data"].shape == (1, 2, 32, 32, 32)
    assert len(batch["target"]) == 4
    first_conv = next(layer for layer in trainer.network.encoder.modules() if isinstance(layer, torch.nn.Conv3d))
    before = first_conv.weight.detach().clone()
    capsys.readouterr()
    result = trainer.train_step(batch)
    assert "phase3_step" not in capsys.readouterr().out
    assert "phase3_step" in Path(trainer.log_file).read_text(encoding="utf-8")
    after = first_conv.weight.detach()
    assert not torch.equal(before, after)
    assert result["loss"] >= 0 and result["global_step"] == 1
    assert trainer.global_step == 1
    assert all(not isinstance(v, torch.Tensor) or not v.requires_grad for v in result.values())
    assert trainer.validation_step(next(validation))["loss"] >= 0
    bad = dict(batch, data=batch["data"][:, :1])
    with pytest.raises(ValueError, match="two"):
        trainer.train_step(bad)


def test_projection_with_no_deep_supervision_performs_one_step(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, projection=True, deep_supervision=False)
    trainer.initialize()
    train, _ = trainer.get_dataloaders()
    batch = next(train)
    assert isinstance(batch["target"], torch.Tensor)
    result = trainer.train_step(batch)
    assert result["projection"] >= 0 and result["global_step"] == 1
    assert trainer.loss.projection.last_valid_ray_counts.shape == (1, 3)


def test_empty_validation_patch_dice_is_undefined_without_runtime_warning(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.current_epoch = 0
    val_outputs = [{"loss": 0.5, "tp_hard": np.array([0]),
                    "fp_hard": np.array([0]), "fn_hard": np.array([0])}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        trainer.on_validation_epoch_end(val_outputs)
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert np.isnan(trainer.logger.get_value("mean_fg_dice", step=-1))
    assert trainer.logger.get_value("val_losses", step=-1) == 0.5


def test_nonempty_validation_patch_dice_keeps_overlap_value(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.current_epoch = 0
    trainer.on_validation_epoch_end([{"loss": 0.25, "tp_hard": np.array([2]),
                                      "fp_hard": np.array([1]), "fn_hard": np.array([1])}])
    assert trainer.logger.get_value("mean_fg_dice", step=-1) == pytest.approx(2 / 3)


def test_amp_overflow_retries_same_batch_before_counting_optimizer_step(tmp_path, monkeypatch, capsys):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.initialize()
    batch = next(trainer.get_dataloaders()[0])

    class OverflowOnceScaler:
        def __init__(self):
            self.attempts = 0
            self.overflow = False
            self.current_scale = 65536.0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            self.attempts += 1
            self.overflow = self.attempts == 1
            if self.overflow:
                next(part for group in optimizer.param_groups for part in group["params"]
                     if part.grad is not None).grad.fill_(float("inf"))

        def step(self, optimizer):
            if not self.overflow:
                optimizer.step()

        def update(self):
            if self.overflow:
                self.current_scale /= 2

        def get_scale(self):
            return self.current_scale

    trainer.grad_scaler = OverflowOnceScaler()
    first_weight = next(part for part in trainer.network.parameters() if part.requires_grad)
    before = first_weight.detach().clone()

    capsys.readouterr()
    result = trainer.train_step(batch)
    output = capsys.readouterr().out
    log = Path(trainer.log_file).read_text(encoding="utf-8")
    assert "phase3_amp_overflow" not in output
    assert "phase3_step" not in output
    assert "phase3_amp_overflow" in log and "phase3_step" in log

    assert trainer.grad_scaler.attempts == 2
    assert trainer.grad_scaler.get_scale() == 32768.0
    assert trainer.global_step == result["global_step"] == 1
    assert result["amp_overflow_retries"] == 1
    assert not torch.equal(first_weight, before)
