"""Group-selected validation and strict, portable epoch-boundary checkpoint state."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import os
import tempfile

import numpy as np
import torch

from .config import config_digest
from .evaluation import aggregate_group_metrics, compute_case_metrics
from .inference import predict_volume
from .initialization import capture_rng_state, restore_rng_state
from .nifti_io import VolumeCase, read_nifti, same_grid
from .provenance import file_sha256
from .training_support import load_prepared_for_trainer


CHECKPOINT_SCHEMA = "phase3_checkpoint_v1"


def resolve_saved_log_path(saved_path: Path, output_folder: Path, minimum_bytes: int) -> Path:
    """Find an epoch log after its model folder has been moved intact."""
    saved_path = Path(saved_path)
    output_folder = Path(output_folder)
    if saved_path.is_file() and saved_path.stat().st_size >= minimum_bytes:
        return saved_path
    relocated = output_folder / saved_path.name
    if (saved_path.parent.name == output_folder.name
            and saved_path.parent.parent.name == output_folder.parent.name
            and relocated.is_file() and relocated.stat().st_size >= minimum_bytes):
        return relocated
    raise FileNotFoundError(f"Checkpoint log is missing or shorter than saved boundary: {saved_path}")


def evaluate_validation_group_dice(trainer, *, on_case=None) -> dict:
    """Predict every validation source at its original physical grid."""
    prepared = load_prepared_for_trainer(trainer)
    state = capture_rng_state()
    benchmark = torch.backends.cudnn.benchmark
    rows = []
    evaluated = {}
    try:
        for case_id in prepared.validation_ids:
            source = prepared.source_cases.get(case_id)
            if source is None:
                raise ValueError(f"Original validation source missing: {case_id}")
            image_stat = Path(source["image"]).stat()
            label_stat = Path(source["label"]).stat()
            identity = (image_stat.st_dev, image_stat.st_ino,
                        label_stat.st_dev, label_stat.st_ino)
            if identity in evaluated:
                original_row, image_hash, label_hash = evaluated[identity]
                if (source["image_sha256"] != image_hash
                        or source["label_sha256"] != label_hash):
                    raise ValueError(f"Validation duplicate source hash differs: {case_id}")
                rows.append(replace(original_row, case_id=case_id,
                                    group_id=prepared.case_groups[case_id]))
                continue
            if (file_sha256(source["image"]) != source["image_sha256"]
                    or file_sha256(source["label"]) != source["label_sha256"]):
                raise ValueError(f"Original validation source changed: {case_id}")
            image, meta = read_nifti(source["image"])
            target, target_meta = read_nifti(source["label"])
            if not same_grid(meta, target_meta):
                raise ValueError(f"Validation image and label grids differ: {case_id}")
            volume = VolumeCase(case_id, prepared.case_groups[case_id], image, target, meta)
            prediction = predict_volume(trainer.network, volume, plans=trainer.plans_manager.plans,
                                        config=trainer.config, stats=prepared.stats, device=trainer.device)
            row = compute_case_metrics(prediction, target, case_id=case_id,
                                       group_id=prepared.case_groups[case_id])
            if on_case is not None:
                on_case(case_id, volume, prediction)
            evaluated[identity] = row, source["image_sha256"], source["label_sha256"]
            rows.append(row)
    finally:
        restore_rng_state(state)
        torch.backends.cudnn.benchmark = benchmark
    summary = aggregate_group_metrics(rows)
    summary["per_case"] = [asdict(row) for row in rows]
    return summary


def build_phase3_checkpoint(trainer) -> dict:
    if not trainer.was_initialized:
        raise ValueError("Cannot checkpoint uninitialized Phase 3 trainer")
    if trainer.global_step != trainer.current_epoch * trainer.num_iterations_per_epoch:
        raise ValueError("Phase 3 checkpoint requires an optimizer-step epoch boundary")
    plans = trainer.plans_manager.plans
    phase3 = plans["phase3"]
    log_path = Path(trainer.log_file)
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    state = {
        "schema_version": CHECKPOINT_SCHEMA,
        "next_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "scheduler_state": trainer.lr_scheduler.state_dict(),
        "best_group_dice": trainer.best_group_dice,
        "best_step": trainer.best_step,
        "rng": capture_rng_state(),
        "train_loader_state": (trainer.dataloader_train.state_dict()
                               if trainer.dataloader_train is not None else None),
        "val_loader_state": (trainer.dataloader_val.state_dict()
                             if trainer.dataloader_val is not None else None),
        "config_digest": phase3["config_digest"],
        "plans_name": plans["plans_name"],
        "dataset_name": plans["dataset_name"],
        "source_fold": phase3["source_fold"],
        "data_digest": phase3["data_digest"],
        "split_digest": phase3["split_digest"],
        "stats_digest": phase3["stats_digest"],
        "stats": phase3["stats"],
        "manifest_digest": phase3["manifest_digest"],
        "seed": trainer.config.seed,
        "max_optimizer_steps": trainer.config.training.max_optimizer_steps,
        "updates_per_epoch": trainer.config.training.updates_per_epoch,
        "device_precision": "amp_fp16" if trainer.device.type == "cuda" else "fp32",
        "prior_load_report": (trainer.prior_load_report.to_dict()
                              if hasattr(trainer.prior_load_report, "to_dict")
                              else trainer.prior_load_report),
        "log_file": str(log_path), "log_bytes": log_path.stat().st_size,
    }
    return {
        "network_weights": trainer.network.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "grad_scaler_state": trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None,
        "logging": trainer.logger.get_checkpoint(),
        "_best_ema": trainer._best_ema,
        "current_epoch": trainer.current_epoch,
        "init_args": trainer.my_init_kwargs,
        "trainer_name": trainer.__class__.__name__,
        "inference_allowed_mirroring_axes": trainer.inference_allowed_mirroring_axes,
        "phase3_state": state,
    }


def restore_phase3_state(trainer, payload: dict) -> None:
    state = payload.get("phase3_state")
    if not isinstance(state, dict) or state.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Missing or unsupported Phase 3 checkpoint state")
    phase3 = trainer.plans_manager.plans["phase3"]
    expected = {
        "config_digest": phase3["config_digest"],
        "plans_name": trainer.plans_manager.plans["plans_name"],
        "dataset_name": trainer.plans_manager.dataset_name,
        "source_fold": phase3["source_fold"],
        "data_digest": phase3["data_digest"],
        "split_digest": phase3["split_digest"],
        "stats_digest": phase3["stats_digest"],
        "stats": phase3["stats"],
        "manifest_digest": phase3["manifest_digest"],
        "seed": trainer.config.seed,
        "max_optimizer_steps": trainer.config.training.max_optimizer_steps,
        "updates_per_epoch": trainer.config.training.updates_per_epoch,
        "device_precision": "amp_fp16" if trainer.device.type == "cuda" else "fp32",
    }
    differences = [key for key, value in expected.items() if state.get(key) != value]
    if differences:
        raise ValueError(f"Phase 3 checkpoint differs from current experiment: {differences}")
    if (payload.get("current_epoch") != state.get("next_epoch")
            or not 0 <= state["next_epoch"] <= trainer.num_epochs
            or not 0 <= state["global_step"] <= trainer.config.training.max_optimizer_steps):
        raise ValueError("Phase 3 checkpoint epoch or step is inconsistent")
    if state["global_step"] != state["next_epoch"] * trainer.num_iterations_per_epoch:
        raise ValueError("Phase 3 resume requires an optimizer-step epoch boundary")
    load_prepared_for_trainer(trainer, verify_files=True)
    if state["global_step"] and (state["train_loader_state"] is None or state["val_loader_state"] is None):
        raise ValueError("Phase 3 checkpoint lacks sampler state")
    log_path = resolve_saved_log_path(state["log_file"], trainer.output_folder,
                                      state["log_bytes"])
    with log_path.open("r+b") as stream:
        stream.truncate(state["log_bytes"])
    trainer.current_epoch = state["next_epoch"]
    trainer.global_step = state["global_step"]
    trainer.best_group_dice = state["best_group_dice"]
    trainer.best_step = state["best_step"]
    trainer.lr_scheduler.load_state_dict(state["scheduler_state"])
    trainer.prior_load_report = state["prior_load_report"]
    trainer._pending_phase3_state = state


def atomic_save_checkpoint(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".phase3-checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
