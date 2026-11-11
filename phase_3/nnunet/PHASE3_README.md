# Phase 3 nnU-Net migration

This directory contains the pinned nnU-Net v2.8.1 source at commit
`468cf803df9b267150ae2b6c0c59b8ac84f16227`. The migration is specified
in `PHASE3_MIGRATION_SPEC.md` and executed by the plan under
`docs/superpowers/plans/2026-11-11-phase3-nnunet-migration.md` at the repository
root. Implementation and verification status are recorded as tasks complete.

## Runtime setup

From `D:/desktop/课题1/code` in PowerShell, provide an existing Python 3.10
executable to the bootstrap script. It creates a separate environment under
`tmp/phase3_nnunet/venv`, inherits the pinned CUDA PyTorch and NumPy from
the specified Python 3.10 base, installs this source tree in editable mode,
checks dependencies, and verifies that
`nnunetv2` was imported from this directory.

```powershell
& 'phase_3/nnunet/env/bootstrap-phase3.ps1' -Python310 'D:/python/miniconda/env/env/nnunet/python.exe'
& 'tmp/phase3_nnunet/venv/Scripts/python.exe' -m pytest -c 'phase_3/nnunet/pytest-phase3.ini' 'phase_3/nnunet/tests/phase3/test_runtime_config.py' -q
```

The source tree contains multiple local `nnunetv2` copies. Always use this
environment and run `verify_runtime` before training or prediction.
The public entry point is `python -m nnunetv2.phase3.cli`. Use the dedicated
environment for all commands below. The 15 self-contained JSON configurations
are under `configs/phase3`; `smoke` is for software checks only. The eight
formal combinations are `baseline`, `swin`, `tubular`, `projection`,
`swin_tubular`, `swin_projection`, `tubular_projection`, and `full`.

## End-to-end commands

Run from `D:/desktop/课题1/code` in PowerShell. Replace the manifest, split,
image, and SSL paths with reviewed inputs. Source fold IDs belong to the
research split; each prepared dataset maps to nnU-Net fold `0` internally.

```powershell
$py = 'tmp/phase3_nnunet/venv/Scripts/python.exe'
$manifest = 'D:/data/fdk/manifest.json'
$splits = 'D:/data/fdk/splits.json'
$run = 'D:/desktop/课题1/code/tmp/phase3_nnunet/formal/fold_0_full'

& $py -m nnunetv2.phase3.cli validate-data --manifest $manifest --splits $splits --run-kind formal
& $py -m nnunetv2.phase3.cli prepare-prior --weights 'D:/desktop/课题1/code/tmp/fdk_v3/official_ssl/model_swinvit.pt' --output "$run/prior_artifact.json"
& $py -m nnunetv2.phase3.cli prepare --manifest $manifest --splits $splits --source-fold 0 --dataset-id 901 --config full --seed 2026 --output "$run/prepared"
& $py -m nnunetv2.phase3.cli train --prepared "$run/prepared/prepared.json" --config full --seed 2026 --device cuda --prior-artifact "$run/prior_artifact.json"
```

The model directory is
`$run/prepared/results/Dataset901_Phase3F00/nnUNetTrainerPhase3__FDKPhase3_<config-digest-prefix>__3d_fullres`.
Use the actual plans name written in `prepared.json`. New runs refuse an
existing result directory. To continue the same experiment from the latest
checkpoint, use `train` with the same `--prepared`, `--config`, `--seed`,
and `--device`, plus `--resume`. Resume reads prior weights from the checkpoint;
omit `--prior-artifact`.

```powershell
$model = 'D:/path/to/the/model-directory'
$prediction = 'D:/path/to/test-predictions'
& $py -m nnunetv2.phase3.cli predict --model-dir $model --mode experimental --manifest $manifest --splits $splits --source-fold 0 --split-name test --device cuda --output $prediction
& $py -m nnunetv2.phase3.cli evaluate --predictions $prediction --manifest $manifest --splits $splits --source-fold 0 --split-name test --output 'D:/path/to/test-metrics'
& $py -m nnunetv2.phase3.cli predict --model-dir $model --mode external --image 'D:/path/to/new-FDK.nii.gz' --case-id external_001 --device cuda --output 'D:/path/to/external-prediction'
```

`experimental` predicts the complete selected split and writes a provenance
record that `evaluate` verifies against image, label, checkpoint, output, split,
  and physical-grid checks. `external` accepts one unlabeled raw FDK volume and
  writes an input, checkpoint and output provenance record; it is ineligible for
  experimental evaluation. Both use the stored train-only intensity
statistics, two aligned views, nnU-Net sliding windows with 0.5 tile step and
Gaussian blending, no mirroring, and original-grid NIfTI export. The mask uses
`p_foreground > 0.5`, assigning exact ties to background. The model folder is
self-contained for prediction; external SSL files and training images are not
required. Native `nnUNetv2_predict` can read the internal `_0000` and `_0001`
two-view NIfTI files; a single raw FDK image needs the wrapper above.

The equivalent native training command, after setting `nnUNet_raw`,
`nnUNet_preprocessed`, `nnUNet_results`, `PHASE3_RUN_CONTEXT`, and
`nnUNet_compile=false`, is:

```powershell
& $py -m nnunetv2.run.run_training Dataset901_Phase3F00 3d_fullres 0 -tr nnUNetTrainerPhase3 -p FDKPhase3_<config-digest-prefix> -device cuda
```

The public `train` command creates those environment values and checks that a
resume checkpoint exists. New SSL training requires a `prepare-prior` artifact;
missing or mismatched weights fail before optimizer updates. `smoke` uses a
32³ patch, batch 1, and four updates. Formal presets retain 128³, batch 2,
and 10,000 updates each; no formal effect estimate is implied by smoke tests.

### Task 1 environment check (2026-11-11)

The checked environment uses Python 3.10.19, torch 2.3.1+cu118, NumPy 1.26.4,
MONAI 1.3.0 and nnU-Net 2.8.1 from this source tree. `pip check` found no
broken requirements. The configuration contract tests passed 11/11 and the
pinned upstream smoke tests passed 7/7. CUDA was visible on an NVIDIA GeForce
RTX 4070 Ti SUPER; model capacity remains untested. The venv inherits torch
and NumPy from `D:/python/miniconda/env/env/nnunet`; changes to that base
environment would require this check and the lock file to be regenerated.
