"""Select a Phase 3 patch with nnU-Net's native 3D full-resolution planner."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from .config import Phase3Config, resolve_config
from .data_conversion import PreparedFold


def apply_nnunet_patch_plan(raw: dict, plan: dict,
                            source_spacing: tuple[float, float, float],
                            *, foreground_probability: float | None = None) -> Phase3Config:
    """Resolve a fresh config; the source template is never modified."""
    config = deepcopy(raw)
    patch = [int(value) for value in plan["patch_size"]]
    if len(patch) != 3 or any(value < 32 or value % 32 for value in patch):
        raise ValueError("nnU-Net patch edges must be divisible by 32 for the Swin prior")
    planned_batch = int(plan["batch_size"])
    if planned_batch < 1:
        raise ValueError("nnU-Net planned an invalid batch size")
    config["cnn"]["patch_size"] = patch
    # Native estimates omit the frozen Swin, tubular, and projection branches.
    config["cnn"]["batch_size"] = min(planned_batch, int(config["cnn"]["batch_size"]))
    config["normalization"]["spacing_mm"] = [float(value) for value in source_spacing]
    if foreground_probability is not None:
        config["training"]["foreground_probability"] = foreground_probability
    return resolve_config(config)


def plan_for_prepared(prepared: PreparedFold, gpu_memory_target_gb: float) -> tuple[dict, list[int], list[float]]:
    """Run the installed nnU-Net planner on the prepared dataset fingerprint."""
    if not 8 <= gpu_memory_target_gb <= 100:
        raise ValueError("GPU memory target must be between 8 and 100 GB")
    from .planning import _source_geometry
    from nnunetv2.experiment_planning.experiment_planners import default_experiment_planner as native

    shape, spacing, fingerprint = _source_geometry(prepared)
    fingerprint_path = prepared.preprocessed_dir / "dataset_fingerprint.json"
    created_fingerprint = not fingerprint_path.exists()
    if created_fingerprint:
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    old_raw, old_preprocessed = native.nnUNet_raw, native.nnUNet_preprocessed
    # The custom Phase 3 trainer permits zero nnU-Net augmentation workers,
    # but the native planner passes this value to torch.set_num_threads.
    old_n_proc = os.environ.get("nnUNet_n_proc_DA")
    try:
        if old_n_proc == "0":
            os.environ["nnUNet_n_proc_DA"] = "1"
        native.nnUNet_raw = str(prepared.raw_root)
        native.nnUNet_preprocessed = str(prepared.preprocessed_root)
        planner = native.ExperimentPlanner(prepared.dataset_name,
                                           gpu_memory_target_in_gb=gpu_memory_target_gb,
                                           preprocessor_name="FDKPreprocessor",
                                           overwrite_target_spacing=list(spacing),
                                           suppress_transpose=True)
        approximate_voxels = float(np.prod(shape, dtype=np.float64) *
                                   (len(prepared.train_ids) + len(prepared.validation_ids)))
        plan = planner.get_plans_for_configuration(spacing, shape, "phase3_auto", approximate_voxels, {})
        return plan, shape, spacing
    finally:
        native.nnUNet_raw, native.nnUNet_preprocessed = old_raw, old_preprocessed
        if old_n_proc is None:
            os.environ.pop("nnUNet_n_proc_DA", None)
        else:
            os.environ["nnUNet_n_proc_DA"] = old_n_proc
        if created_fingerprint:
            fingerprint_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-memory-target-gb", type=float, required=True)
    parser.add_argument("--foreground-probability", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = PreparedFold.from_dict(json.loads(args.prepared.read_text(encoding="utf-8")))
    source = json.loads(args.config.read_text(encoding="utf-8"))
    plan, shape, spacing = plan_for_prepared(prepared, args.gpu_memory_target_gb)
    config = apply_nnunet_patch_plan(source, plan, tuple(spacing),
                                     foreground_probability=args.foreground_probability)
    report = {"planner": "nnUNetv2.ExperimentPlanner.get_plans_for_configuration",
              "gpu_memory_target_gb": args.gpu_memory_target_gb,
              "source_shape_dhw": [int(value) for value in shape],
              "source_spacing_dhw_mm": [float(value) for value in spacing],
              "native_patch_size_dhw": [int(value) for value in plan["patch_size"]],
              "native_batch_size": int(plan["batch_size"]),
              "phase3_patch_size_dhw": list(config.cnn.patch_size),
              "phase3_batch_size": config.cnn.batch_size,
              "foreground_sampling_probability": config.training.foreground_probability,
              "note": "Phase 3 retains its five-stage fusion network; nnU-Net selects the training patch."}
    report_path = args.output.with_suffix(".plan.json")
    serialized_report = json.dumps(report, indent=2)
    serialized_config = json.dumps(config.to_dict(), indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(serialized_report, encoding="utf-8")
    args.output.write_text(serialized_config, encoding="utf-8")
    print(json.dumps({"config": str(args.output), "report": str(report_path), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
