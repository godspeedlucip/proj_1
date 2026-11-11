"""Fixed 3D full-resolution nnU-Net plans for grouped FDK experiments."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np

from .config import Phase3Config, config_digest, resolve_config
from .data_conversion import PreparedFold
from .nifti_io import read_nifti
from .provenance import file_sha256


def _source_geometry(prepared: PreparedFold) -> tuple[list[int], list[float], dict]:
    case_ids = (*prepared.train_ids, *prepared.validation_ids)
    shapes, spacings = [], []
    for case_id in case_ids:
        image = prepared.raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        data, meta = read_nifti(image)
        shapes.append(tuple(int(v) for v in data.shape))
        spacings.append(tuple(float(v) for v in np.linalg.norm(meta.affine_mm[:3, :3], axis=0)[::-1]))
    median_shape = [int(round(v)) for v in np.median(np.asarray(shapes), axis=0)]
    median_spacing = [float(v) for v in np.median(np.asarray(spacings), axis=0)]
    fingerprint = {
        "case_ids": list(case_ids),
        "shapes_after_crop": [list(v) for v in shapes],
        "spacings": [list(v) for v in spacings],
        "median_relative_size_after_cropping": 1.0,
        "foreground_intensity_properties_per_channel": {},
        "phase3_data_digest": prepared.data_digest,
    }
    return median_shape, median_spacing, fingerprint


def make_phase3_plans(prepared: PreparedFold, config: Phase3Config) -> dict:
    if prepared.schema_version != "phase3_prepared_fold_v1" or prepared.nnunet_fold != 0:
        raise ValueError("Expected a Phase 3 prepared fold mapped to nnU-Net fold 0")
    median_shape, median_spacing, _ = _source_geometry(prepared)
    digest = config_digest(config)
    plans_name = "FDKPhase3_" + digest[:12]
    channels = list(config.cnn.channels)
    arch_kwargs = {
        "n_stages": 5,
        "features_per_stage": channels,
        "conv_op": "torch.nn.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * 5,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * 4,
        "n_conv_per_stage": [2] * 5,
        "n_conv_per_stage_decoder": [2] * 4,
        "conv_bias": True,
        "norm_op": "torch.nn.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"negative_slope": 0.01, "inplace": True},
    }
    configuration = {
        "data_identifier": plans_name + "_3d_fullres",
        "preprocessor_name": "FDKPreprocessor",
        "batch_size": config.cnn.batch_size,
        "patch_size": list(config.cnn.patch_size),
        "median_image_size_in_voxels": median_shape,
        "spacing": [0.2, 0.2, 0.2],
        "normalization_schemes": ["NoNormalization", "NoNormalization"],
        "use_mask_for_norm": [False, False],
        "resampling_fn_data": "identity_resample",
        "resampling_fn_data_kwargs": {},
        "resampling_fn_seg": "identity_resample",
        "resampling_fn_seg_kwargs": {},
        "resampling_fn_probabilities": "identity_resample",
        "resampling_fn_probabilities_kwargs": {},
        "batch_dice": False,
        "architecture": {
            "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            "arch_kwargs": arch_kwargs,
            "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
        },
    }
    plans = {
        "dataset_name": prepared.dataset_name,
        "plans_name": plans_name,
        "original_median_spacing_after_transp": median_spacing,
        "original_median_shape_after_transp": median_shape,
        "image_reader_writer": "FDKNiftiIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {"3d_fullres": configuration},
        "experiment_planner_used": "FDKnnUNetPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {},
        "phase3": {
            "schema_version": config.schema_version,
            "config": config.to_dict(),
            "config_digest": digest,
            "prepared_schema": prepared.schema_version,
            "manifest_digest": prepared.manifest_digest,
            "split_digest": prepared.split_digest,
            "data_digest": prepared.data_digest,
            "stats_digest": prepared.stats_digest,
            "stats": prepared.stats.to_dict(),
            "source_fold": prepared.source_fold,
            "nnunet_fold": 0,
        },
    }
    validate_phase3_plans(plans)
    return plans


def validate_phase3_plans(plans: dict) -> None:
    try:
        if set(plans["configurations"]) != {"3d_fullres"}:
            raise ValueError("Only 3d_fullres is supported")
        phase3 = plans["phase3"]
        cfg = resolve_config(phase3["config"])
        if phase3["config_digest"] != config_digest(cfg):
            raise ValueError("Phase 3 configuration digest mismatch")
        if phase3["nnunet_fold"] != 0 or phase3["schema_version"] != cfg.schema_version:
            raise ValueError("Phase 3 fold/schema mismatch")
        if plans["image_reader_writer"] != "FDKNiftiIO" or plans["label_manager"] != "LabelManager":
            raise ValueError("Unsupported reader or labels")
        if plans["transpose_forward"] != [0, 1, 2] or plans["transpose_backward"] != [0, 1, 2]:
            raise ValueError("FDK transpose must be identity")
        c = plans["configurations"]["3d_fullres"]
        if c["patch_size"] != list(cfg.cnn.patch_size) or c["batch_size"] != cfg.cnn.batch_size:
            raise ValueError("Patch or batch conflicts with Phase 3 config")
        if c["spacing"] != [0.2, 0.2, 0.2] or c["normalization_schemes"] != ["NoNormalization"] * 2:
            raise ValueError("FDK spacing/normalization contract mismatch")
        if c["use_mask_for_norm"] != [False, False] or c["preprocessor_name"] != "FDKPreprocessor":
            raise ValueError("Unsupported preprocessing")
        for key in ("data", "seg", "probabilities"):
            if c[f"resampling_fn_{key}"] != "identity_resample":
                raise ValueError("FDK export must use identity resampling")
        arch = c["architecture"]
        if arch["network_class_name"] != "dynamic_network_architectures.architectures.unet.PlainConvUNet":
            raise ValueError("Phase 3 requires PlainConvUNet")
        kwargs = arch["arch_kwargs"]
        if (kwargs["features_per_stage"] != list(cfg.cnn.channels)
                or kwargs["strides"] != [[1, 1, 1]] + [[2, 2, 2]] * 4
                or kwargs["n_conv_per_stage"] != [2] * 5
                or kwargs["n_conv_per_stage_decoder"] != [2] * 4):
            raise ValueError("Five-stage CNN topology mismatch")
        if (kwargs["conv_op"] != "torch.nn.Conv3d" or kwargs["norm_op"] != "torch.nn.InstanceNorm3d"
                or kwargs["nonlin"] != "torch.nn.LeakyReLU"):
            raise ValueError("CNN operator mismatch")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed Phase 3 plans: {exc}") from exc


def prepare_training_fold(prepared: PreparedFold, config: Phase3Config) -> PreparedFold:
    """Save plans and run the native Blosc2 case writer for train+validation only."""
    from nnunetv2.preprocessing.preprocessors.fdk_preprocessor import FDKPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = make_phase3_plans(prepared, config)
    manager = PlansManager(plans)
    configuration = manager.get_configuration("3d_fullres")
    plans_path = prepared.preprocessed_dir / f"{plans['plans_name']}.json"
    data_dir = prepared.preprocessed_dir / configuration.data_identifier
    if plans_path.exists() or data_dir.exists():
        raise FileExistsError("Existing Phase 3 plans or preprocessed data will not be overwritten")
    _, _, fingerprint = _source_geometry(prepared)
    fingerprint_path = prepared.preprocessed_dir / "dataset_fingerprint.json"
    if fingerprint_path.exists():
        raise FileExistsError(fingerprint_path)
    shutil.copy2(prepared.raw_dir / "dataset.json", prepared.preprocessed_dir / "dataset.json")
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    plans_path.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    data_dir.mkdir(parents=True, exist_ok=False)
    dataset_json = json.loads((prepared.raw_dir / "dataset.json").read_text(encoding="utf-8"))
    preprocessor = FDKPreprocessor(verbose=False)
    for case_id in (*prepared.train_ids, *prepared.validation_ids):
        images = [str(prepared.raw_dir / "imagesTr" / f"{case_id}_{channel:04d}.nii.gz") for channel in (0, 1)]
        label = str(prepared.raw_dir / "labelsTr" / f"{case_id}.nii.gz")
        preprocessor.run_case_save(str(data_dir / case_id), images, label, manager, configuration, dataset_json)
    cache_hashes = {path.relative_to(data_dir).as_posix(): file_sha256(path)
                    for path in sorted(data_dir.rglob("*")) if path.is_file()}
    if not cache_hashes:
        raise ValueError("Preprocessed Phase 3 cache is empty")
    completed = replace(prepared, plans_name=plans["plans_name"],
                        config_digest=plans["phase3"]["config_digest"], preprocessed_complete=True,
                        preprocessed_file_hashes=cache_hashes)
    (prepared.raw_dir.parent.parent / "prepared.json").write_text(json.dumps(completed.to_dict(), indent=2), encoding="utf-8")
    return completed
