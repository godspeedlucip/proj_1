# Marker-guided nnU-Net E5-U runbook

This runbook covers the software path from a Phase 1 v1 prediction package to
four-channel E5-U data and nnU-Net training/inference. A synthetic run checks
software wiring only; it does not establish segmentation performance.

## Input package

The converter consumes two JSON documents. The dataset manifest contains
`phase2_fold_id`, group-wise `splits` (`train`, `val`, `test`) and one sample
row per `(case_id, view_id)`. Each row has `nnunet_case_id`, `group_id`,
`split`, `projection_path`, `mask_path` for train/val, and `geometry_id`.
Each sample also carries either `angle_deg` or a non-empty
`geometry_manifest_key` to identify its acquisition geometry.
Paths are relative to the manifest file or absolute. The Phase 1 package
contains the same `phase2_fold_id`, a `source_registry` with each model's
`model_id`, `fit_group_ids`, `selection_group_ids`, `model_artifact_id`, and
`pipeline_config_id`, and `cases` in
`phase1_marker_case_v1` format. The validator joins only by the exact pair
`(case_id, view_id)` and checks that each case has exactly three views.

Training predictions must use `group_oof`; validation/test predictions must
use `held_out`. Source fitting and parameter-selection groups must be drawn
from this fold's training groups. The training group's own case cannot be in
the fitting or selection groups of the prediction used for that case.

## Install and paths

Use Python 3.10 or newer in an environment that satisfies the dependencies in
`phase_2/nnUNet/pyproject.toml`:

```powershell
python -m pip install -e .\phase_2\nnUNet
$env:nnUNet_raw = 'D:\nnUNet\nnUNet_raw'
$env:nnUNet_preprocessed = 'D:\nnUNet\nnUNet_preprocessed'
$env:nnUNet_results = 'D:\nnUNet\nnUNet_results'
$env:nnUNet_compile = 'false'
```

Allocate a new Dataset ID for each Phase 2 fold and experiment. The examples
below use `501` and `fold_0`; change both consistently to avoid overwriting an
existing dataset or model.

## Validate and convert

```powershell
python -m phase_2.tools.marker_guided.cli validate `
  --manifest D:\data\phase2_manifest.json `
  --package D:\data\phase1_package `
  --phase2-fold-id fold_0

python -m phase_2.tools.marker_guided.cli build-e5u `
  --manifest D:\data\phase2_manifest.json `
  --package D:\data\phase1_package `
  --phase2-fold-id fold_0 `
  --dataset-id 501 `
  --output-dir "$env:nnUNet_raw\Dataset501_MarkerGuided_fold_0"
```

The default prior parameters are saved into `dataset.json` and checked into
`prior_config` metadata: `sigma_0=2`, `k_sigma=1`, `w_0=3`, `k_w=1`,
`endpoint_taper_px=4`, and `d_min=1` pixels. They are engineering defaults;
for a real study, select them using only the current fold's training groups.
The converter writes channels 0–3 as projection, point, band, and validity.
Train and validation samples go into `imagesTr`/`labelsTr`; held-out test
projections go into `imagesTs`. `splits_final.json` contains case IDs and keeps
all three views of each case together. The planner copies it into the
preprocessed dataset folder for the Trainer.

## Plan, preprocess, train and predict

```powershell
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
nnUNetv2_train 501 2d 0 -tr nnUNetTrainerMarkerGuidedSmoke -device cpu
nnUNetv2_predict `
  -i "$env:nnUNet_raw\Dataset501_MarkerGuided_fold_0\imagesTs" `
  -o D:\nnUNet\predictions\Dataset501_fold_0 `
  -d 501 -c 2d -f 0 -tr nnUNetTrainerMarkerGuidedSmoke -device cpu
```

The smoke Trainer shortens the run to two epochs with two training and one
validation iteration per epoch. For real training, use
`nnUNetTrainerMarkerGuidedFeatureFusion` and the intended training budget.
CPU is useful for verifying the pipeline but is slow; use `-device cuda` when
a compatible GPU is available. Checkpoint inference must use the same Trainer
name that created the checkpoint.

Before the first training run, verify that the converted
`splits_final.json` reached
`$env:nnUNet_preprocessed\Dataset501_MarkerGuided_fold_0\splits_final.json`.
After prediction, use `case_mapping.json` to map each output back to its
original case and view. Checkpoint and prediction paths are under
`nnUNet_results` and the requested output directory, respectively.

The Phase 1 exporter is present, but its current output container differs from
the Phase 2 converter input: Phase 1 writes separate case prediction files and
a dictionary source registry, while Phase 2 currently expects inline cases
and a list source registry. Build the small manifest/package adapter described
in [`phase_2/README.md`](../README.md) before passing a Phase 1 export to the
commands above. Contract-valid synthetic fixtures verify software wiring only;
they do not establish segmentation performance.
