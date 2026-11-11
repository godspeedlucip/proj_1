# Phase 3 nnU-Net migration verification

Checked 2026-11-11 (Asia/Shanghai). The pinned upstream source is nnU-Net v2.8.1,
commit `468cf803df9b267150ae2b6c0c59b8ac84f16227`; the 281 archive files
matched the recorded source hashes at import. The Phase 3 additions are local
extensions, not part of that upstream commit. `UPSTREAM_SOURCE.json` records the
local Python source digest and its calculation method.

## Environment

The isolated `tmp/phase3_nnunet/venv` used Python 3.10.19, torch
2.3.1+cu118, NumPy 1.26.4, MONAI 1.3.0, and locally installed nnU-Net 2.8.1.
`nnunetv2` resolved to this source tree. `pip check` reported no broken
requirements. The venv inherits pinned torch and NumPy from
`D:/python/miniconda/env/env/nnunet`; rebuilding against a changed base
requires rerunning the bootstrap and lock check.

## Results

| Check | Command / scope | Result |
| --- | --- | --- |
| Phase 3 software suite | `python -m pytest -c phase_3/nnunet/pytest-phase3.ini phase_3/nnunet/tests/phase3 -q` | 87 passed, 2 opt-in GPU tests skipped, 148 third-party warnings |
| Public CLI pipeline | `test_pipeline.py` on four independent synthetic source groups | Passed: validate-data → prepare → 2 epochs × 2 updates → resume → experimental predict → evaluate → external predict with input/checkpoint/output hashes; one held-out volume exceeds the 32³ smoke patch |
| Official SSL artifact | `test_prior_artifact.py` with the existing 411 MB `model_swinvit.pt` | 100% MONAI backbone key/shape match; SHA-256 `1cf19eca3ed0472daf33dedd2f095b118db16edac0fc53d9e878f328ca68c988` |
| Formal STP CUDA capacity | `PHASE3_GPU_TESTS=1 python -m pytest -c phase_3/nnunet/pytest-phase3.ini phase_3/nnunet/tests/phase3/test_gpu_capacity.py -q` | 1 passed; full 128³, batch 2, float16 AMP optimizer step and native multi-tile sliding prediction |
| Cross-stage regression | Old fusion Python, `--basetemp=tmp/phase3_nnunet/pytest-crossstage`, phase_1/2/3 tests excluding monai, prior_artifact, gpu, slow | 122 passed, 17 deselected |
| Dependencies | `tmp/phase3_nnunet/venv/Scripts/python.exe -m pip check` | No broken requirements |

The first cross-stage attempt used pytest's default system temporary directory;
67 tests failed during fixture setup with `PermissionError` at
`C:/Users/19599/AppData/Local/Temp/pytest-of-19599`. Rerunning the same
selection with a workspace-local `--basetemp` passed 122/122. No Phase 1/2
source files were changed for that environment issue.

## Measured GPU capacity

NVIDIA GeForce RTX 4070 Ti SUPER, 17,170,956,288 bytes reported total VRAM;
15,819,866,112 bytes free before the test. The full preset had 24,956,053
parameters, 16,894,051 trainable. The tested optimizer step included the
frozen official SSL Swin prior, bottleneck and decoder-guided fusion,
TubularEnhancement, four-scale segmentation loss, and three-angle projection
loss. It completed in 2.24 s with peak 8,555,659,264 bytes allocated and
10,435,428,352 bytes reserved. Native sliding prediction on an original
130×97×65 DHW volume took 1.44 s, with peak 4,304,359,424 bytes allocated
and 6,134,169,600 bytes reserved. These are one-run measurements, not training
throughput estimates. Raw measurement JSON is in the uncommitted
`tmp/phase3_nnunet/gpu_capacity.json` runtime directory.

## Acceptance boundaries

- Synthetic software and GPU capacity checks passed. No real paired stent
  labels or verified independent clinical source groups were supplied, so
  the eight formal 10,000-update experiments and performance comparisons
  were not run.
- The Windows Blosc2 adapter opens lazy chunks using an ASCII basename under
  a process-wide working-directory lock. Unrelated concurrent file I/O in
  the same process is outside this checked path.
- GPU `grid_sample` backward is not claimed to be bitwise deterministic.
  Checkpoint resume was verified for CPU next-batch and optimizer state.
- Checkpoint restore rehashes original training and validation sources and
  every prepared cache file. Checkpoints require an optimizer-step epoch boundary.
- Experimental evaluation checks the image, label and output physical grids;
  external predictions retain input, checkpoint and output hashes and cannot be
  evaluated as a held-out experimental split.
- The preparation CLI rejects an existing output directory; reuse of a
  previously prepared cache is not implemented. New preparations remain
  digest-bound and do not silently accept an existing directory.

Run instructions, preset names, native command equivalents, model directory
layout, and external versus experimental prediction rules are in
`PHASE3_README.md`.
