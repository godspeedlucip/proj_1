import copy
from pathlib import Path
import shutil

import pytest
import torch

from nnunetv2.phase3.checkpointing import build_phase3_checkpoint, evaluate_validation_group_dice
from nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3 import nnUNetTrainerPhase3
from .test_trainer import _trainer


def test_group_dice_selects_best_independent_of_patch_pseudo_dice(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.initialize()
    trainer.get_dataloaders()
    module = __import__("nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3", fromlist=["evaluate_validation_group_dice"])
    scores = iter((.8, .4))
    monkeypatch.setattr(module, "evaluate_validation_group_dice",
                        lambda _: {"mean": {"dice": next(scores)}, "group_count": 1})
    trainer.global_step = 2
    trainer.logger.log("mean_fg_dice", .1, 0)
    trainer.on_epoch_end()
    best = Path(trainer.output_folder) / "checkpoint_best.pth"
    assert best.is_file() and trainer.best_group_dice == .8
    trainer.global_step = 4
    trainer.logger.log("mean_fg_dice", .9, 1)
    trainer.on_epoch_end()
    best_payload = torch.load(best, map_location="cpu", weights_only=False)
    latest = torch.load(Path(trainer.output_folder) / "checkpoint_latest.pth", map_location="cpu", weights_only=False)
    assert best_payload["phase3_state"]["best_step"] == 2
    assert best_payload["current_epoch"] == 1
    assert latest["current_epoch"] == 2
    assert latest["phase3_state"]["best_group_dice"] == .8


def test_checkpoint_resume_replays_next_three_batches_and_optimizer(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.initialize()
    trainer.get_dataloaders()
    trainer.train_step(next(trainer.dataloader_train))
    trainer.train_step(next(trainer.dataloader_train))
    trainer.current_epoch = 1
    checkpoint = Path(trainer.output_folder) / "checkpoint_latest.pth"
    trainer.save_checkpoint(str(checkpoint))
    uninterrupted = [(batch["keys"], batch["data"].clone()) for batch in
                     (next(trainer.dataloader_train) for _ in range(3))]
    plans = copy.deepcopy(trainer.plans_manager.plans)
    plans["continue_training"] = True
    resumed = nnUNetTrainerPhase3(plans, "3d_fullres", 0, trainer.dataset_json,
                                  device=torch.device("cpu"))
    resumed.load_checkpoint(str(checkpoint))
    resumed.on_train_start()
    replayed = [(batch["keys"], batch["data"].clone()) for batch in
                (next(resumed.dataloader_train) for _ in range(3))]
    assert resumed.current_epoch == trainer.current_epoch
    assert resumed.global_step == trainer.global_step
    assert all(a[0] == b[0] and torch.equal(a[1], b[1]) for a, b in zip(uninterrupted, replayed))
    assert resumed.optimizer.state_dict()["param_groups"] == trainer.optimizer.state_dict()["param_groups"]
    with pytest.raises(FileNotFoundError):
        resumed.load_checkpoint(str(tmp_path / "missing.pth"))


def test_rejects_in_epoch_checkpoint_and_changed_training_sources_or_cache(tmp_path, monkeypatch):
    for kind in ("epoch", "source", "cache"):
        trainer = _trainer(tmp_path / kind, monkeypatch)
        trainer.on_train_start()
        checkpoint = Path(trainer.output_folder) / "checkpoint_latest.pth"
        trainer.save_checkpoint(str(checkpoint))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if kind == "epoch":
            payload["phase3_state"]["global_step"] = 1
        elif kind == "source":
            prepared = Path(trainer.preprocessed_dataset_folder_base).parents[1]
            source = prepared / ".." / "input" / "case_0.nii.gz"
            source = source.resolve()
            source.write_bytes(source.read_bytes() + b"changed")
        else:
            case_file = next(Path(trainer.preprocessed_dataset_folder).glob("*.b2nd"))
            case_file.write_bytes(case_file.read_bytes() + b"changed")
        plans = copy.deepcopy(trainer.plans_manager.plans)
        plans["continue_training"] = True
        resumed = nnUNetTrainerPhase3(plans, "3d_fullres", 0, trainer.dataset_json,
                                      device=torch.device("cpu"))
        with pytest.raises(ValueError, match="boundary|hash"):
            resumed.load_checkpoint(payload if kind == "epoch" else str(checkpoint))


def test_full_volume_validation_uses_original_source_grid(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.initialize()
    summary = evaluate_validation_group_dice(trainer)
    assert summary["case_count"] == 1 and summary["group_count"] == 1
    assert 0 <= summary["mean"]["dice"] <= 1


def test_copied_model_folder_rebuilds_with_native_predictor(tmp_path, monkeypatch):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.on_train_start()
    trainer.save_checkpoint(str(Path(trainer.output_folder) / "checkpoint_best.pth"))
    copied = tmp_path / "portable model"
    shutil.copytree(trainer.output_folder_base, copied)
    predictor = nnUNetPredictor(device=torch.device("cpu"), use_mirroring=False,
                                allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(copied), use_folds=(0,),
                                                   checkpoint_name="checkpoint_best.pth")
    assert predictor.network.__class__.__name__ == "Phase3Network"
    assert predictor.network.decoder.deep_supervision is False
    assert predictor.network.prior is None


def test_native_two_epoch_smoke_keeps_best_and_latest(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.run_training()
    folder = Path(trainer.output_folder)
    assert trainer.global_step == 4 and trainer.current_epoch == 2
    assert (folder / "checkpoint_best.pth").is_file()
    assert (folder / "checkpoint_latest.pth").is_file()
    final = torch.load(folder / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    assert final["current_epoch"] == 2
    assert final["phase3_state"]["global_step"] == 4
