# FDK fusion v3 implementation acceptance — 2026-11-10

This is the historical acceptance record for source commit `b156557`, copied
into Phase 3. Branch/path names below describe that original verification.
The new pipeline now lives only in `phase_3.fdk_unet`; see the
[Phase 3 synchronization record](../README.md) for current location and checks.

Branch: `codex/fdk-unet-swin-fusion-v3`, from `b9d0edb`.
Plan: `docs/superpowers/plans/2026-11-10-fdk-unet-swin-fusion-v3-implementation.md`.
Spec SHA-256: `eafd2f05e83e3ddad1e89309da7d1cef2d08385fcfc9668262e11942ba738f80`.

Tasks 1–11 are implemented. Task 12 software/device acceptance is complete;
formal real-data experiments remain pending user-supplied paired labels and
source evidence. Optional Tasks 13–15 were not selected. There are **zero**
formal runs and no claimed scientific benefit.

## Evidence

| Check | Observed result |
|---|---|
| Final phase 1/2 baseline regression | 120 passed, 17 deselected; Python 3.14.3 / CPU torch 2.11.0 |
| Final pinned MONAI/SSL regression | 6 passed, 89 deselected; Python 3.10.19 / torch 2.3.1+cu118 / MONAI 1.3.0 |
| Slow checks | 2 passed: actual plain 128³ shapes; two asymmetric rod phantoms reach minimum Dice 0.9801802 at step 50 |
| GPU core capacity | 6 passed: A, B, H00, H10, H01, C; actual 128³, batch 2, float16 |
| GPU prefix equivalence | 2 passed: both execution depths, actual 128³ |
| GPU continuous/resume | 1 passed; observed maximum tensor difference 0.0 in one four-step comparison |
| Final review regressions | 7 expected failures before fixes, then 21 targeted tests passed; all included in final regression |
| Full repository | Collection fails in 4 unchanged nnUNet test modules due to missing `SEG_TIEBREAKS` in installed batchgenerators |

The full-repository collection failures affect `test_checkpoints.py`,
`test_do_split.py`, `test_foreground_locations.py`, and `test_resampling.py`.
No unrelated nnUNet implementation or installed dependency was modified.
Deselected or skipped tests are not counted as successful acceptance.

RTX 4070 Ti SUPER peak allocated memory was 2.450 GiB for A and
7.415–7.416 GiB for fusion configurations. No patch/batch reduction was needed.
Timings measure one update and one synchronized batch-1 forward, not a warmed
benchmark. Generated resource records remain in
`tmp/fdk_v3/gpu_resources_128_v2.json` and `tmp/fdk_v3/gpu_resume.json`.

Real official SSL weights loaded 100% of backbone parameters, with SHA-256
`1cf19eca3ed0472daf33dedd2f095b118db16edac0fc53d9e878f328ca68c988`
and size 411,162,269 bytes. Download metadata and load audit remain under
`tmp/fdk_v3/official_ssl/`. Offline checkpoint restore succeeds without external
SSL files. The isolated local interpreter is
`tmp/fdk_v3/fusion310/Scripts/python.exe`; the portable lock is in `env/`.

The user-supplied example resolves to
`D:/desktop/ICBM_010_sample_00_0-60.nii.gz` (the message had an extra final `z`).
It is 128³ with 0.2 mm spacing, mm units, uint8 storage and NIfTI scaling.
The decoded finite range is approximately -1.73622 to 2.67435. Original-grid
reading passed. This is an image-only check, not supervised data acceptance.
Variable shapes are covered by mixed short/long axes and original-grid export
tests. User correction: `vessel=1` is blood vessel; use verified stent-strut
labels (`stent=1` where applicable).

## Independent review

One fresh-context whole-branch reviewer examined `b9d0edb..02377e4`. Three
important findings were accepted and fixed in one test-first pass:

1. Reserved experiment IDs must agree with the actual core architecture.
   Four parameterized identity regressions failed before the fix and now pass.
2. Experimental predictions and evaluation must be bound to checkpoint fold,
   split, case identity, image/output hashes and label hashes. Wrong-fold
   prediction and altered evaluation provenance regressions failed, then passed.
   Unlabeled external inference remains supported without splits.
3. A formal comparison must not pool different projection sources. A mixed
   source manifest regression failed, then passed.

The final 120-test baseline suite and 6-test MONAI/SSL suite passed after these
fixes. The fixes do not alter model arithmetic; existing slow/GPU numerical
acceptance was not rerun unnecessarily. No second review was requested.

## Decisions and limits

These are the complete rulings from implementation and the reviewer's declined
judgments, in decision order:

1. Serialize independent-source evidence as `group_provenance`: the spec requires
   evidence but does not prescribe its field. Cost if wrong: externally prepared
   formal manifests need conversion to this schema.
2. Use strict CPU deterministic algorithms and CUDA `warn_only`: this installed
   PyTorch lacks deterministic CUDA MaxPool3d backward. Cost: bitwise CUDA
   continuation is not guaranteed, despite the observed zero difference.
3. Defer real source independence, stent-label validity and scientific benefit
   to supplied evidence and formal runs, as the user requested. Software guards
   cannot prove those facts. Cost: current software acceptance is insufficient
   to support research conclusions.
4. Leave optional Tasks 13–15 unimplemented: they were not selected and do not
   block the core. Cost: staged fine-tuning and spatial adapters require later
   implementation and acceptance.
5. Accept the executor's actual GPU/SSL evidence without a second expensive
   run by the reviewer. Cost: numerical/device evidence has not been independently
   reproduced on another machine or by another executor.
6. Leave the unrelated nnUNet dependency/collection failure unchanged. Cost:
   the entire repository cannot currently receive a green pytest result, although
   the affected phase 1/2 suite passes.
7. Retain the disclosed CUDA reproducibility limit rather than extending the
   one observed exact comparison to a general promise. Cost: repeat GPU runs
   may differ; record environment and evaluate repeated seeds.

Deferred minor: smoke inference uses half-patch stride even when
`inference_stride` requests a different value. Actual stride is recorded in
telemetry. Formal core 128³/64 stride is unaffected. This minor was left out of
the review fix pass as required by the execution workflow.

The requested branch and pre-existing user changes are preserved. The branch
has not been merged or pushed. Runtime weights, environments, checkpoints and
large scratch data are not committed.
