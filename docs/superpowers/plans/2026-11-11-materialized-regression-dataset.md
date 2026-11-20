# Materialized Regression Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a self-contained Phase 2 marker-fusion regression dataset under `/data1/liuyang/stent/output/data`.

**Architecture:** Copy 6000 stacked TIFF inputs; write each of the 240 unique implant projections once as a float32 [0,1] TIFF using the training maximum; write shared point/band/validity prior TIFFs and a relative-path manifest. Update the training loader to consume the exported package.

**Tech Stack:** Python, tifffile, NumPy, PyTorch.

**Spec:** The user's request to create owned data in `./output/data`, with the previously agreed continuous implant target including markers.

## Global Constraints

- Destination is `/data1/liuyang/stent/output/data/phase2_marker_fusion_regression_v1`.
- Never modify source TIFFs or prior experiment outputs.
- Keep train/validation stent-series split and the fixed training-derived target scale.
- Do not publish the package until all referenced files are present and verified.

## Review Focus

- Repeated CT views of one implant share the same target and priors; different marker coordinates must fail.
- Manifest paths remain within the package and refer only to existing files.
- Saved labels are continuous, finite float32 in [0,1], including markers.
- Training on the package does not normalize the label twice.
- Partial exports remain in a temporary directory and cannot masquerade as complete data.

---

### Task 1: Exporter

**Files:** Create `regression_unet/export_data.py`; extend `regression_unet/test_regression_unet.py`.

- [x] Write a failing fixture test for copied image, shared normalized target/priors, relative manifest and source preservation.
- [x] Implement atomic package export and package verification.
- [x] Run the targeted test to green.

### Task 2: Package loader

**Files:** Modify `regression_unet/data.py`, `regression_unet/train.py`; extend tests.

- [x] Write a failing test that loads the package and reads its precomputed channels without scaling the label twice.
- [x] Add `--dataset-dir` to training and retain legacy source-manifest support.
- [x] Run the targeted test to green.

### Task 3: Full export and audit

**Files:** Create package under `output/data`; update `regression_unet/README.md`.

- [x] Run the exporter against the 6000-view physical manifest.
- [x] Verify counts, file ranges, split isolation, absolute containment, and a sample through the training loader.
- [x] Run relevant tests and report package path and size.
