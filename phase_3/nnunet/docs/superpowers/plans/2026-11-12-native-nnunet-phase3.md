# Native nnU-Net Phase 3 Implementation Plan

> **For agentic workers:** Use the original `nnUNetTrainer` methods for the training loop. Do not copy their bodies into Phase 3 code.

## Goal and boundary

The plain baseline uses `nnUNetTrainer` itself. The prior variant adds the frozen Swin/UNETR encoder and its fusion network. The full Phase 3 variant additionally adds its configured segmentation/projection loss. All variants retain the original trainer's optimizer step, data loader, augmentation, sampling policy, learning-rate schedule, validation step, and checkpoint format. Existing `nnUNetTrainerPhase3` checkpoints remain tied to the old class and are not reinterpreted.

The `PlainConvUNet` named in the nnU-Net plan is a network architecture; `nnUNetTrainer` is the class that trains it. The current `nnUNetTrainerPhase3` inherits from `nnUNetTrainer` but overrides `train_step`, `get_dataloaders`, `configure_optimizers`, validation, and checkpoint methods, so its earlier “baseline” was not the original trainer.

## Task 1: Original plain baseline

- Keep `nnunetv2/phase3/native_baseline.py` as a launcher that directly instantiates `nnUNetTrainer` for a prepared fold without a prior.
- Default to stock trainer values (1000 epochs, 250 training steps, 50 validation steps, foreground oversampling 0.33, stock checkpoint cadence). Permit explicit short-run budget/checkpoint options only for diagnostics, and label those runs accordingly.
- Test in `tests/phase3/test_native_baseline.py`: one CPU epoch writes a checkpoint whose `trainer_name` is exactly `nnUNetTrainer`, whose first convolution uses both input channels, and whose stock Dice+CE loss is active.

## Task 2: Prior and full Phase 3 with stock training methods

- Keep `nnUNetTrainerPhase3NativePrior` as a thin subclass overriding only `build_network_architecture`. Load official prior weights after `initialize()` and before the first optimizer step; keep the prior frozen.
- Add `nnUNetTrainerPhase3NativeFull`, overriding only `build_network_architecture` (inherited from prior class) and `_build_loss` to use `Phase3CompositeLoss`. No `train_step`, loader, optimizer, scheduler, validation, or checkpoint overrides.
- Test that those methods are inherited by identity from `nnUNetTrainer`, that the prior is frozen and loaded before training, and that the full variant selects the configured loss.

## Task 3: Entry point and compatibility

- Extend the native launcher to select plain, prior-only, or full mode from explicit options and validate that the prepared plan matches the choice. Do not silently replace the existing `train` command, because prior checkpoints use the old trainer class and state format.
- Document commands and the distinction between old Phase 3 runs and new native-trainer runs in `PHASE3_README.md`.
- Test wrong mode/artifact combinations and resume behavior. A resumed native run must restore a native checkpoint, never an old Phase 3 checkpoint.

## Task 4: Verify the requested comparison

- Run a tiny CPU training test first. Check GPU occupancy before any GPU run.
- If the user still wants a three-epoch comparison, start new plain and prior-only runs on the same prepared cases with the same explicit short-run budget. Record that the short budget alters only experiment duration, not the trainer methods.
- Compare per-epoch patch Dice from the original trainer and full-volume Dice from an external evaluator. Report train-case and held-out `stent2` predictions separately, because the validation 75 IDs represent only three distinct volumes.

## Review focus

- The original trainer must consume both input channels in the plain baseline; the prior model intentionally routes the first through CNN and the second through the frozen encoder.
- The stock loss must remain Dice+CE in the plain and prior-only variants.
- The full variant's additional loss must not require a custom optimizer step; verify finite loss and gradients on a small batch.
- Existing custom checkpoints must not load into the new trainer classes.
- Any short-run budget override must be explicit and must not change production defaults.
