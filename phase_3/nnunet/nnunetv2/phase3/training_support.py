"""Strict split, prepared-fold and loader wiring for the native trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .data_conversion import PreparedFold
from .storage import FDKDatasetBlosc2
from .dataloading import Phase3DataLoader
from .transforms import build_transforms
from .provenance import file_sha256


def load_prepared_for_trainer(trainer, *, verify_files: bool = False) -> PreparedFold:
    folder = Path(trainer.preprocessed_dataset_folder_base)
    path = folder.parents[1] / "prepared.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared fold record missing: {path}")
    prepared = PreparedFold.from_dict(json.loads(path.read_text(encoding="utf-8")))
    phase3 = trainer.plans_manager.plans["phase3"]
    if (prepared.dataset_name != trainer.plans_manager.dataset_name
            or prepared.source_fold != phase3["source_fold"] or prepared.nnunet_fold != 0
            or prepared.config_digest != phase3["config_digest"]
            or prepared.data_digest != phase3["data_digest"]
            or prepared.split_digest != phase3["split_digest"]
            or prepared.stats_digest != phase3["stats_digest"]
            or not prepared.preprocessed_complete):
        raise ValueError("Prepared fold record differs from Phase 3 plans")
    if verify_files:
        expected_ids = set(prepared.train_ids) | set(prepared.validation_ids)
        if set(prepared.source_cases) != expected_ids:
            raise ValueError("Prepared source hash inventory differs from train/validation split")
        for case_id, row in prepared.source_cases.items():
            for kind in ("image", "label"):
                if file_sha256(Path(row[kind])) != row[f"{kind}_sha256"]:
                    raise ValueError(f"Prepared source {kind} hash changed: {case_id}")
        data_dir = Path(trainer.preprocessed_dataset_folder)
        actual = {path.relative_to(data_dir).as_posix(): file_sha256(path)
                  for path in sorted(data_dir.rglob("*")) if path.is_file()}
        if not prepared.preprocessed_file_hashes or actual != prepared.preprocessed_file_hashes:
            raise ValueError("Preprocessed Phase 3 cache hash changed")
    return prepared


def get_strict_split(trainer) -> tuple[list[str], list[str]]:
    if trainer.fold != 0:
        raise ValueError("Phase 3 maps each source fold to nnU-Net fold 0")
    split_path = Path(trainer.preprocessed_dataset_folder_base) / "splits_final.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Explicit Phase 3 split missing: {split_path}")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(splits, list) or len(splits) != 1 or set(splits[0]) != {"train", "val"}:
        raise ValueError("Phase 3 requires one explicit nnU-Net fold 0 split")
    train, val = splits[0]["train"], splits[0]["val"]
    prepared = load_prepared_for_trainer(trainer, verify_files=True)
    if (set(train) != set(prepared.train_ids) or set(val) != set(prepared.validation_ids)
            or set(train) & set(val) or not train or not val):
        raise ValueError("Phase 3 split differs from prepared group partition")
    available = set(FDKDatasetBlosc2.get_identifiers(trainer.preprocessed_dataset_folder))
    if available != set(train) | set(val):
        raise ValueError("Preprocessed case inventory differs from explicit split")
    return train, val


def build_phase3_loaders(trainer):
    prepared = load_prepared_for_trainer(trainer)
    train_keys, val_keys = get_strict_split(trainer)
    dataset_train = FDKDatasetBlosc2(trainer.preprocessed_dataset_folder, train_keys)
    dataset_val = FDKDatasetBlosc2(trainer.preprocessed_dataset_folder, val_keys)
    patch = trainer.configuration_manager.patch_size
    common = dict(batch_size=trainer.batch_size, patch_size=patch, final_patch_size=patch,
                  label_manager=trainer.label_manager, oversample_foreground_percent=.33,
                  sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
    train_loader = Phase3DataLoader(dataset_train, transforms=build_transforms(
        training=True, deep_supervision=trainer.enable_deep_supervision),
        case_groups=prepared.case_groups, seed=trainer.config.seed, **common)
    val_loader = Phase3DataLoader(dataset_val, transforms=build_transforms(
        training=False, deep_supervision=trainer.enable_deep_supervision),
        case_groups=prepared.case_groups, seed=trainer.config.seed + 1, **common)
    return train_loader, val_loader


def prior_artifact_from_context() -> Path:
    context_path = os.getenv("PHASE3_RUN_CONTEXT")
    if not context_path:
        raise ValueError("SSL training requires PHASE3_RUN_CONTEXT with prior_artifact")
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    artifact = context.get("prior_artifact")
    if not artifact:
        raise ValueError("SSL training context lacks prior_artifact")
    return Path(artifact)
