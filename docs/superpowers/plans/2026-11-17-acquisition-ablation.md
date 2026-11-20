# Acquisition Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Generate five complete paired acquisition datasets and run controlled Phase 3 training and held-out evaluation.

**Architecture:** Add an acquisition builder and a resumable orchestration entry point to the existing nnU-Net fork. Extend its manifest contract to describe variable-view FDK honestly, retaining legacy three-view compatibility. Keep data on a dedicated spacious volume and preserve the existing training/evaluation implementations.

**Tech Stack:** Python 3.10, TIGRE in nnunet environment; PyTorch/MONAI/nnU-Net in phase3 environment; NIfTI and JSON provenance.

**Spec:** ../specs/2026-11-17-acquisition-ablation-design.md

## Global Constraints

- Five configurations: 60/3, 45/3, 75/3, 60/2, 60/5; center directions paired.
- Exact current Phase 2 split and complete 1680/240/20 case inventory.
- Full source grid; no oracle crop; exact simulated projections.
- Identical GDL full-network training configuration, seed and 17500-update budget.
- Never overwrite existing experiments or use held-out data for training statistics.

## Review Focus

- Wrong split or missing Phase 2 views must fail before writing data.
- 2/5-view manifests must validate without false three-view metadata.
- Fractional/negative angles must remain exact, including endpoint span.
- Interrupted stages and duplicate supervisor launches must not overwrite work.
- Formal metrics must only come from completed held-out predictions and correct checkpoint provenance.

### Task 1: Acquisition metadata and paired dataset builder

**Files:** Create nnunetv2/phase3/acquisition.py and tests/phase3/test_acquisition.py; modify nnunetv2/phase3/manifest.py.

**Interfaces:** acquisition_angles(center, span, views); collect_acquisitions(phase1_manifest, phase2_mapping, span, views); build_dataset(..., output, gpu).

- [ ] Test exact 45/75/60 angles, Phase 2 split coverage and mismatch rejection.
- [ ] Test variable-view manifests and legacy validation; confirm failures first.
- [ ] Implement strict metadata validation, exact TIGRE projection/FDK and immutable repeated-volume links.
- [ ] Verify focused tests and existing data contract tests.

### Task 2: Preparation, smoke and formal orchestration

**Files:** Create nnunetv2/phase3/acquisition_run.py, acquisition_launch.py and runner tests.

**Interfaces:** Absolute-path CLI with init, build, prepare, smoke, run, status and compare; stage state JSON and lock files.

- [ ] Test state/lock and matrix factor isolation failures first.
- [ ] Implement fresh run directories, interpreter-specific subprocesses, logs/PIDs, free-space checks and supported resume.
- [ ] Prepare each full fold using existing preprocessing; create separate smoke folds from a per-group representative.
- [ ] Formal completion triggers existing predict/evaluate and paired JSON/CSV comparison.
- [ ] Run relevant CPU suite, then CUDA projection equivalence and full smoke train/reload/predict.

### Task 3: Execute and verify

- [ ] Request dedicated /data2 output access and preserve run pointer in project.
- [ ] Build all five datasets; assert exact split and unique-volume counts and baseline equivalence.
- [ ] Preprocess and plan actual data; verify smoke checks.
- [ ] Recheck GPUs, launch monitored long training once per arm and verify worker PIDs/GPU/log progress.
- [ ] Complete held-out testing when training completes; report actual state without claiming pending results.
- [ ] Fresh final review of implementation and scientific controls; address material findings.
