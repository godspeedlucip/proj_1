# Projection Regression U-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trainable 2D U-Net that predicts continuous implant projections, including markers, on a fixed training-derived 0–1 scale using the saved Phase 2 nnU-Net topology.

**Architecture:** Load the existing 2D plans and instantiate the Phase 2 `MarkerGuidedPlainConvUNet` with one image channel, two prior channels, a validity gate, and one linear output. Read paired source TIFFs lazily from the physical manifest, and train with a sparse regression loss. Preserve legacy plain U-Net checkpoint loading for the prior baseline.

**Tech Stack:** Python, PyTorch 2.3.1, tifffile, dynamic-network-architectures, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-projection-regression-unet-design.md`

## Global Constraints

- Source data is `output/data/phase1_200`; never overwrite it or the Dataset501 results.
- Use the saved `nnUNetPlans.json` architecture and default patch/batch settings.
- Target is the original float32 projection including markers, divided by the maximum intensity observed in training targets and clipped to [0, 1].

## Review Focus

- Missing target TIFF: dataset indexing raises a useful error.
- Mismatched input and target dimensions: loading raises before training.
- Duplicate stent group in train and validation: indexing rejects the fold.
- Background-dominated optimization: loss explicitly weights target support.
- Model output is one floating channel, not a two-class probability.

---

### Task 1: Model and plans

**Files:** Create `regression_unet/model.py`; test `regression_unet/test_regression_unet.py`.

**Interfaces:** `load_plan(path) -> dict`, `build_unet(plan, input_channels=4) -> torch.nn.Module`.

- [x] Write tests for saved-plan topology, one output channel, and a small-plan forward pass.
- [x] Run them and confirm failure due to the missing module.
- [x] Implement the two interfaces with the plans-configured E5-U fusion network and a legacy plain U-Net loading path.
- [x] Run tests to green.

### Task 2: Paired dataset and regression loss

**Files:** Create `regression_unet/data.py`, `regression_unet/loss.py`; extend test file.

**Interfaces:** `load_samples(manifest, split) -> list[ProjectionSample]`; `ProjectionDataset(samples, patch_size, training)`; `weighted_smooth_l1(pred, target, foreground_weight=100.0)`.

- [x] Write tests for source pairing, split isolation, channels, float target, dimension failures and foreground weighting.
- [x] Run them and confirm failure.
- [x] Implement the interfaces.
- [x] Run tests to green.

### Task 3: Training entry point and usage

**Files:** Create `regression_unet/train.py`, `regression_unet/README.md`; extend test file.

**Interfaces:** `python -m regression_unet.train --help`; `train(...)` saves latest and best checkpoints.

- [x] Write a minimal training-loop test using a tiny plan and TIFF fixture.
- [x] Run it and confirm failure.
- [x] Implement CLI, nnU-Net SGD/poly defaults, validation metrics and checkpointing.
- [x] Run tests and a short CPU smoke.
- [x] Inspect diffs and report what was verified; do not launch a long GPU job.
