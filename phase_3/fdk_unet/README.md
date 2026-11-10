# FDK U-Net × Swin fusion v3

This is the Phase 3 copy of source commit `b156557`. Use the commands below with
`phase_3.fdk_unet`. This is the sole implementation of the new pipeline.
The Phase 2 FDK package retains its pre-existing six-level U-Net and tests;
see `../README.md`.

Independent single-volume FDK → **stent strut** binary segmentation. The original
`model.UNet3D` API remains available. New experiments use the five-level
`PlainUNet3D` or `SwinPriorUNet3D`, with one shared decoder and one logit channel.

## Installation

Ordinary baseline (no MONAI, nnUNet or external paper imports):

```powershell
python -m pip install -r phase_3/fdk_unet/requirements-base.txt
```

Fusion requires MONAI **1.3.0**, einops **0.8.2**, and a compatible Python.
The tested isolated environment is Python 3.10.19 / PyTorch 2.3.1+cu118 /
NumPy 1.26.4. See `env/requirements-windows-cu118.lock.txt` and `env/README.md`.
Install CUDA PyTorch from its matching official wheel index before the lock.
MONAI 1.3.0 fails on this host's default Python 3.14.3; do not silently upgrade
MONAI or change its prefix. The local validation interpreter is
`tmp/fdk_v3/fusion310/Scripts/python.exe` (generated, not portable or committed).

## Data

Use `fdk_stent_manifest_v1` with explicit case/group/source IDs, image/label paths,
`input_kind=fdk_3view`, actual three angles, projection_source, upstream_run_id,
and `target_semantics=stent_struts`. Paths are relative to the manifest directory.
Formal manifests also require `group_provenance`, mapping each group ID to a
traceable source description or evidence path. These declarations require human
verification; software cannot establish physical independence from filenames.
Each formal manifest must use one fixed `projection_source`; different source
conditions require separate runs and reports.
For predicted masks, `upstream_registry[run_id]` must list `fit_group_ids`,
`selection_group_ids` and `calibration_group_ids`.

**User-confirmed correction:** `vessel=1` means blood vessel, not stent.
Use verified stent-strut labels (`stent=1` where applicable). Do not relabel vessel
data merely by changing the manifest. Real paired data and source mappings are
still pending; no formal independent-entity experiment has been run.

NIfTI volumes may vary in size. Each axis must have 0.2 mm spacing (tolerance
1e-4 mm); mm/meter/micron units are converted for checks and distance metrics.
Original shape, affine and header are retained. Unknown units, conflicting
qform/sform, nonfinite values, nonbinary labels or mismatched grids fail.
Storage `(i,j,k)` becomes tensor `(k,j,i)` and is reversed on export.
NIfTI scaling is applied on read and cleared on floating probability/uint8 mask
write. There is no implicit resampling. Short axes are padded symmetrically;
long axes use terminal-inclusive overlapping windows. Probability is averaged.

Splits use `fdk_stent_splits_v1`:

```json
{"schema_version":"fdk_stent_splits_v1","folds":{"0":{"train":["a","b"],"validation":["c"],"test":["d"]}}}
```

This is an example, not evidence that four real groups exist. Shared sources,
patients, decoded images or paths cannot cross groups. Upstream fitting,
selection and calibration may not use the downstream validation/test groups.
`--smoke` relaxes provenance evidence only; geometry and binary targets remain
checked. Import requires a JSON `--source-groups` file with target_semantics,
`cases` keyed by full case ID, optional group_provenance/upstream_registry, and
the same per-case provenance fields as the manifest. It strips only `_0000`.

## Core configs

All paths below are relative to `phase_3/fdk_unet/configs/`.

| ID | Config | Prior | Bottleneck | Decoder-guided gate |
|---|---|---|---|---|
| A | `plain_fdk_unet3d_v1.json` | none | no | no |
| B | `ablations/random_prior.json` | random | yes | yes |
| H00 | `ablations/h00.json` | SSL | no | no |
| H10 | `ablations/h10.json` | SSL | yes | no |
| H01 | `ablations/h01.json` | SSL | no | yes |
| C/H11 | `swin_prior_fdk_unet3d_v2.json` | SSL | yes | yes |

C and H11 are aliases for one experiment. All skip gates have `3*C` input width.
Reserved experiment IDs must match `core_experiment_id`; a baseline cannot be
reported as C by changing only its display ID.
Layers4 weights are loaded and frozen but never executed. No-bottleneck groups
stop at layers2. All core residual alpha values begin at zero; alignment weights
are nonzero and gate final layers start at zero. This preserves the baseline's
initial logits while allowing staged gradient activation.

Formal defaults: patch 128³, batch 2, Adam 1e-3, 10,000 optimizer updates,
validation every 250 and at the last update. Loss is per-case masked Dice+BCE.
The best checkpoint maximizes equal-group validation Dice; earlier ties win.
Smoke configs must explicitly set a distinct experiment_id, run_kind=smoke and
a different budget. Optional S2/S3/spatial-adapter Tasks 13–15 are not selected
or implemented; unsupported stages/configuration fields are rejected.

## Commands

Run from the repository root with the validated interpreter for fusion:

```powershell
python -m phase_3.fdk_unet.cli import-nnunet --source data/Dataset --source-groups data/source_groups.json --output data/fdk_unet
python -m phase_3.fdk_unet.cli validate-data --manifest data/fdk_unet/manifest.json --splits data/fdk_unet/splits.json
python -m phase_3.fdk_unet.cli prepare-prior --init ssl --weights tmp/fdk_v3/official_ssl/model_swinvit.pt --manifest data/fdk_unet/manifest.json --splits data/fdk_unet/splits.json --fold 0 --output artifacts/prior/fold_0_ssl
python -m phase_3.fdk_unet.cli prepare-prior --init random --manifest data/fdk_unet/manifest.json --splits data/fdk_unet/splits.json --fold 0 --output artifacts/prior/fold_0_random
python -m phase_3.fdk_unet.cli train --config phase_3/fdk_unet/configs/plain_fdk_unet3d_v1.json --manifest data/fdk_unet/manifest.json --splits data/fdk_unet/splits.json --fold 0 --seed 2026 --initialization-artifact artifacts/initialization/fold_0_seed_2026.pt --device cuda --output runs/fdk_unet_v3/A/fold_0_seed_2026
python -m phase_3.fdk_unet.cli train --config phase_3/fdk_unet/configs/swin_prior_fdk_unet3d_v2.json --manifest data/fdk_unet/manifest.json --splits data/fdk_unet/splits.json --fold 0 --seed 2026 --prior-artifact artifacts/prior/fold_0_ssl/prior.json --initialization-artifact artifacts/initialization/fold_0_seed_2026.pt --device cuda --output runs/fdk_unet_v3/C/fold_0_seed_2026
python -m phase_3.fdk_unet.cli train --resume runs/fdk_unet_v3/C/fold_0_seed_2026/last.pt --device cuda
python -m phase_3.fdk_unet.cli predict --checkpoint runs/fdk_unet_v3/C/fold_0_seed_2026/best.pt --manifest data/fdk_unet/manifest.json --splits data/fdk_unet/splits.json --fold 0 --split test --device cuda --output runs/fdk_unet_v3/C/fold_0_seed_2026/predictions
python -m phase_3.fdk_unet.cli evaluate --manifest data/fdk_unet/manifest.json --splits data/fdk_unet/splits.json --fold 0 --split test --predictions runs/fdk_unet_v3/C/fold_0_seed_2026/predictions --output runs/fdk_unet_v3/C/fold_0_seed_2026/evaluation
```

Only `prepare-prior --download-official` accesses the network. SSL requires
either explicit local weights or that download flag; random prior preparation
reads no SSL file. Statistics use only train groups (65,536 samples per group),
never labels or held-out intensities. Prepare separate statistics per fold.
New runs use one shared initialization artifact per fold/seed. Named RNGs keep
CNN/fusion initialization, sampling and augmentation independent.

Existing output directories are rejected. Resume accepts only checkpoint/device,
restores the saved experiment and snapshots, and verifies current file hashes.
No new seed, budget, config, prior artifact or initialization override is accepted.
CPU/CUDA precision changes cannot promise exact training continuation and are
rejected. Inference can restore onto either supported device and requires no
original weight file or training dataset. Checkpoints contain trusted Python
training state; load only your own run artifacts. Unlabeled prediction manifests
are supported; omit splits to predict all listed cases. Labels never select ROI.
Experimental prediction with `--splits` must match the checkpoint's fold, split,
case identities and image hashes. It writes `prediction_run.json`, binding output
hashes and the saved label hashes. Evaluation requires this experimental record
and checks it against the requested split and labels. External inference without
splits remains available, but its outputs cannot be reported through this command
as independent experimental test results.

Every run records resolved configuration, absolute data snapshots, data hashes,
source commit/dirty state, environment, metrics.jsonl, last.pt and best.pt.
Predictions preserve the original grid; telemetry contains checkpoint SHA-256,
window settings, latency, peak CUDA allocation and input clipping fractions.
Evaluation exports case CSV, group JSON and summary JSON. Skeleton distances use
Lee skeletonization and affine-mapped millimeter coordinates; missing distances
have explicit statuses and denominators rather than silently becoming zero.

## Verification status (2026-11-10)

- Source-branch regression: 120 tests passed across `phase_1/tests` and `phase_2/tests`
  with dependency/device/slow markers excluded; another 6 MONAI/SSL integration
  tests passed in the pinned environment. Two slow and nine GPU acceptance tests
  also passed. See `VERIFICATION.md` for scope and review decisions.
- New/old networks, zero-alpha equivalence, activated gradient paths, provenance,
  NIfTI units/directions, synchronized data, physical metrics and CLI: tested.
- CPU continuation versus interrupted/resumed training: exact tensors, BN,
  optimizer, RNG and next batch match.
- Two fixed asymmetric synthetic 32³ rod phantoms: minimum training Dice 0.98018
  at step 50 (maximum smoke budget 500). This tests optimization, not FDK accuracy.
- MONAI 1.3.0 CPU 32³ prefix: matches official outputs at atol=1e-6, rtol=1e-5;
  hooks confirm unused stages do not execute. Six-structure synthetic CLI tested.
- Real official SSL: 100% backbone parameters loaded; artifact size 411,162,269
  bytes, SHA-256 `1cf19eca3ed0472daf33dedd2f095b118db16edac0fc53d9e878f328ca68c988`.
  Download metadata and detailed audit are in generated `tmp/fdk_v3/official_ssl/`.
- Offline MONAI restore: tested after removing external weights/artifact.
- GPU 128³/batch 2/float16: all six groups passed forward/backward, feature-shape,
  finite-gradient and frozen-prior checks on RTX 4070 Ti SUPER. Peak allocation
  was 2.450 GiB for A and 7.415–7.416 GiB for fusion groups; no patch/batch reduction.
  Generated per-model counts, hashes and timings: `tmp/fdk_v3/gpu_resources_128_v2.json`.
- Official prefix equivalence also passed on CUDA at 128³. A four-step CUDA
  continuous/resumed comparison observed maximum tensor difference 0.0 in this
  run. PyTorch 2.3.1 CUDA MaxPool3d backward lacks a deterministic implementation:
  CUDA uses deterministic-algorithm warnings and records that limitation; CPU
  uses strict determinism. GPU bitwise reproducibility is not promised.
- Formal real-data runs: **0**; paired stent labels and independent source mappings
  are pending. A sample image passes read/grid validation, not training validation.
- Full-repository collection is blocked in four pre-existing vendored nnUNet
  modules by missing `batchgenerators...SEG_TIEBREAKS`; these are outside this
  implementation. The phase 1/2 regression suite above is green.
- Deferred minor: smoke inference always uses half-patch stride, even if its
  `inference_stride` field specifies another value. Telemetry records the actual
  stride; the fixed formal core setting (128/64) is unaffected.

```powershell
python -m pytest -q -p no:cacheprovider phase_3/tests -m "not monai and not prior_artifact and not gpu and not slow"
python -m pytest -q -p no:cacheprovider phase_3/tests -m slow
$env:FDK_SSL_WEIGHTS="<absolute model_swinvit.pt path>"
tmp/fdk_v3/fusion310/Scripts/python.exe -m pytest -q -p no:cacheprovider phase_3/tests -m "(monai or prior_artifact) and not gpu"
tmp/fdk_v3/fusion310/Scripts/python.exe -m pytest -q -p no:cacheprovider phase_3/tests -m gpu
```

On this host the shared pytest temp/cache directories have permission problems;
use a fresh explicit `--basetemp <scratch>/<unique-test-run>` and create its
parent first. The D: drive is low on space; checkpoint tests need several GiB
and the final regressions used the authorized C: scratch directory.
Skipped dependency/device tests do not count as passed acceptance.

The resource test measures one training update and one synchronized batch-1
forward per group (no TTA or postprocessing); timings are capacity diagnostics,
not a warmed benchmark or a prediction of 72-run wall-clock time. The core real
matrix remains six configurations × confirmed folds × seeds 2026/2027/2028,
10,000 updates each. C/H11 counts once. Report per-group paired differences
H10−H00, C−H01, H01−H00, C−H10, their interaction, C−B and C−A only after
those runs exist. Test groups never select architecture, budgets or checkpoints.
