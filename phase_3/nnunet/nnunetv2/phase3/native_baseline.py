"""Run the original nnUNetTrainer loop on a prepared Phase 3 fold.

Prior mode changes network construction. Full mode also adds the Phase 3 loss.
Optimizer steps, augmentation, and validation remain those of the trainer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--mode", choices=("plain", "prior", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--prior-artifact", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if any(value is not None and value < 1 for value in
           (args.epochs, args.steps_per_epoch, args.validation_steps)):
        raise ValueError("Epoch and step counts must be positive")
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    if not prepared.get("preprocessed_complete") or prepared.get("nnunet_fold") != 0:
        raise ValueError("Expected a completed Phase 3 fold 0")
    preprocessed_dir = Path(prepared["preprocessed_dir"])
    plans = json.loads((preprocessed_dir / f"{prepared['plans_name']}.json").read_text(encoding="utf-8"))
    if (plans["dataset_name"] != prepared["dataset_name"]
            or plans["phase3"]["config_digest"] != prepared["config_digest"]):
        raise ValueError("Prepared record and plans differ")
    plans["continue_training"] = args.resume
    config = plans["phase3"]["config"]
    prior_enabled = bool(config["swin"]["enabled"])
    extras_enabled = bool(config["tubular"]["enabled"] or config["projection"]["enabled"])
    if args.mode == "plain":
        if prior_enabled or extras_enabled or args.prior_artifact is not None:
            raise ValueError("Plain mode requires a plain plan and no prior artifact")
    elif args.mode == "prior":
        if not prior_enabled or extras_enabled or (args.prior_artifact is None and not args.resume):
            raise ValueError("Prior mode requires only the frozen encoder and initial weights")
    elif not prior_enabled or (args.prior_artifact is None and not args.resume):
        raise ValueError("Full Phase 3 mode requires the frozen encoder and initial weights")
    if args.prior_artifact is not None and not args.prior_artifact.is_file():
        raise FileNotFoundError(args.prior_artifact)

    trainer_name = {"plain": "nnUNetTrainer",
                    "prior": "nnUNetTrainerPhase3NativePrior",
                    "full": "nnUNetTrainerPhase3NativeFull"}[args.mode]
    output = (Path(prepared["results_dir"])
              / f"{trainer_name}__{prepared['plans_name']}__3d_fullres/fold_0")
    if args.resume:
        checkpoint = next((output / name for name in ("checkpoint_latest.pth", "checkpoint_final.pth")
                           if (output / name).is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError("Resume requires a latest or final checkpoint")
    elif output.exists():
        raise FileExistsError(f"Existing native baseline will not be overwritten: {output}")

    root = args.prepared.resolve().parent
    os.environ["nnUNet_raw"] = str(root / "raw")
    os.environ["nnUNet_preprocessed"] = str(root / "preprocessed")
    os.environ["nnUNet_results"] = str(root / "results")
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    trainer_class = nnUNetTrainer
    if prior_enabled:
        from nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3NativePrior import (
            nnUNetTrainerPhase3NativePrior,
        )
        trainer_class = nnUNetTrainerPhase3NativePrior
    if args.mode == "full":
        from nnunetv2.training.nnUNetTrainer.phase3.nnUNetTrainerPhase3NativeFull import (
            nnUNetTrainerPhase3NativeFull,
        )
        trainer_class = nnUNetTrainerPhase3NativeFull
    dataset_json = json.loads((preprocessed_dir / "dataset.json").read_text(encoding="utf-8"))
    trainer = trainer_class(plans, "3d_fullres", 0, dataset_json, torch.device(args.device))
    if args.epochs is not None:
        trainer.num_epochs = args.epochs
    if args.steps_per_epoch is not None:
        trainer.num_iterations_per_epoch = args.steps_per_epoch
    if args.validation_steps is not None:
        trainer.num_val_iterations_per_epoch = args.validation_steps
    if args.save_every_epoch:
        trainer.save_every = 1
    if args.resume:
        trainer.load_checkpoint(str(checkpoint))
    elif prior_enabled:
        from nnunetv2.phase3.prior_weights import initialize_prior
        trainer.initialize()
        initialize_prior(trainer.network, args.prior_artifact)
    trainer.run_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
