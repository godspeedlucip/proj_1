"""Isolated native nnU-Net training subprocess for a prepared Phase 3 fold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .config import Phase3Config, config_digest
from .data_conversion import PreparedFold


def model_folder(prepared: PreparedFold) -> Path:
    if not prepared.plans_name:
        raise ValueError("Prepared fold has no resolved plans name")
    return (prepared.results_dir / f"nnUNetTrainerPhase3__{prepared.plans_name}__3d_fullres"
            / "fold_0")


def launch_training(prepared: PreparedFold, config: Phase3Config, *, seed: int,
                    device: str, prior_artifact: Path | None, resume: bool) -> int:
    if device not in ("cpu", "cuda"):
        raise ValueError("Phase 3 training supports CPU or CUDA")
    if seed != config.seed or not prepared.preprocessed_complete or prepared.nnunet_fold != 0:
        raise ValueError("Training seed or prepared fold state differs")
    if prepared.config_digest != config_digest(config):
        raise ValueError("Prepared plans do not match resolved training configuration")
    plans_file = prepared.preprocessed_dir / f"{prepared.plans_name}.json"
    if not plans_file.is_file():
        raise FileNotFoundError(plans_file)
    output = model_folder(prepared)
    if resume:
        if prior_artifact is not None:
            raise ValueError("Resume must use checkpoint prior weights, not an external artifact")
        if not any((output / name).is_file() for name in ("checkpoint_latest.pth", "checkpoint_final.pth")):
            raise FileNotFoundError("Resume requires an existing latest or final checkpoint")
    elif output.exists():
        raise FileExistsError(f"Existing training result will not be overwritten: {output}")
    if config.swin.enabled and config.swin.init == "ssl" and not resume:
        if prior_artifact is None or not Path(prior_artifact).is_file():
            raise FileNotFoundError("New SSL training requires a prior artifact")
    elif prior_artifact is not None:
        raise ValueError("Prior artifact is only valid for a new SSL training run")
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update({
        "nnUNet_raw": str(prepared.raw_root),
        "nnUNet_preprocessed": str(prepared.preprocessed_root),
        "nnUNet_results": str(prepared.results_root),
        "nnUNet_compile": "false",
        "nnUNet_n_proc_DA": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": str(root),
    })
    environment.pop("nnUNet_extTrainer", None)
    context = {"schema_version": "phase3_run_context_v1",
               "prepared": str((prepared.raw_dir.parent.parent / "prepared.json").resolve()),
               "prior_artifact": str(Path(prior_artifact).resolve()) if prior_artifact else None}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json",
                                     prefix="phase3-run-context-", dir=prepared.raw_dir.parent.parent,
                                     delete=False) as stream:
        json.dump(context, stream, indent=2)
        context_path = Path(stream.name)
    environment["PHASE3_RUN_CONTEXT"] = str(context_path)
    args = [sys.executable, "-m", "nnunetv2.phase3.launch", "--child",
            "--dataset", prepared.dataset_name, "--plans", prepared.plans_name,
            "--device", device]
    if resume:
        args.append("--resume")
    result = subprocess.run(args, cwd=str(root), env=environment, check=False)
    return int(result.returncode)


def _run_child(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from nnunetv2.phase3.runtime import verify_runtime
    from nnunetv2.run.run_training import get_trainer_from_args

    verify_runtime(Path(__file__).resolve().parents[2])
    torch.set_num_threads(2 if args.device == "cpu" else 1)
    trainer = get_trainer_from_args(args.dataset, "3d_fullres", 0,
                                    "nnUNetTrainerPhase3", args.plans, args.resume,
                                    device=torch.device(args.device))
    if args.resume:
        folder = Path(trainer.output_folder)
        checkpoint = next((folder / name for name in ("checkpoint_latest.pth", "checkpoint_final.pth")
                           if (folder / name).is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError("Resume checkpoint disappeared before child startup")
        trainer.load_checkpoint(str(checkpoint))
    trainer.run_training()
    trainer.perform_actual_validation()
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_child())
