# Phase 3 acquisition ablation

## Intent and selected approach

Run a paired acquisition experiment with the existing Phase 3 full nnU-Net model.
The requested workflow includes dataset generation, training and held-out testing.
The initial matrix is 60°/3 (reference), 45°/3, 75°/3, 60°/2 and 60°/5.
This five-arm design estimates separate span and view-count effects economically.
A nine-arm factorial design would additionally estimate interactions, at greater
compute and storage cost. Defer it until these main effects have been measured.
The assistant selects the five-arm design under the user's authorization to
evaluate and run the most necessary configurations.

## Data and acquisition

Use all 1940 cases in the CURRENT Phase 1 manifest and cross-check every original
view and split with Phase 2 Dataset503 case_mapping.json (5820 rows). This is
1680 train, 240 validation and 20 test acquisition cases, representing 168/24/2
physical head/stent pairs. Never reuse the old wall_stent test split: wall_stent
is now training; stent2 is validation; stent5 and w_stent are test.

Keep each original triplet's center direction. New angles are
center + linspace(-span/2, span/2, views), including endpoints. At center 30°:
45/3 = [7.5,30,52.5], 60/3 = [0,30,60], 75/3 = [-7.5,30,67.5],
60/2 = [0,60], 60/5 = [0,15,30,45,60]. Use continuous unwrapped angle arrays.
Reproject the same full 3D stent and marker volume with TIGRE at exact angles;
do not round or interpolate existing projections. Preserve stored attenuation
coefficients, geometry, full source voxel grid and target semantics.

Default to ideal simulated implant line integrals, as Dataset601 did. An optional
question about real Phase 2 predictions was presented to the user. Until steered
otherwise, this is a pure acquisition experiment, not end-to-end Phase 2 accuracy.
Validate new baseline projections numerically against existing angle TIFFs.

Preserve all case IDs and write explicit per-case provenance. Repeated head
backgrounds produce identical ideal implant inputs; use immutable hard links only
after matching model, geometry and exact angles. There should be 100 distinct
volumes per arm (70/10/20 by split), not 1940 independent observations. Never
create a biological sample confidence interval from angle/head duplicates.

## Training and evaluation

Use the phase3 environment and pinned local nnU-Net fork. Reuse the established
full model: Swin SSL frozen prior, fusion, tubular refinement, GDL+BCE,
canonical-patch projection auxiliary loss weight 0.1. Keep its auxiliary loss
angles [0,30,60] fixed across arms; they are not acquisition metadata.
Use seed 2026, 192³ patch, batch 2, foreground probability 0.9,
250 updates/epoch, 17500 updates (70 epochs), SGD and existing schedule.
Compute each arm's intensity statistics from training only. Check native planner
on the actual geometry before training; hold model capacity and patch budget fixed.

All arms train from fresh initialization, with validation selecting checkpoint_best.
Before any long training, validate datasets, preprocess, complete short CUDA
training, reload its checkpoint and predict. Keep smoke results separate.
After successful formal training, predict and evaluate all 20 test cases using
the selected best checkpoint. Save per-case and per-stent Dice, IoU, precision,
recall, false-positive volume, skeleton distance and inference time, plus paired
differences against 60/3. No test selection, threshold tuning or oracle ROI.

## Execution and storage

/data1 had only 27 GB free; /data2 had 2.3 TB free at preflight. Request elevated
filesystem access for a dedicated /data2/liuyang/stent/proj_1 acquisition run.
Keep source, design and a run pointer in the requested /data1 project. Do not
delete, move or overwrite existing experiments. Estimate space and check free
space before generating/preprocessing and starting each long job.

Check current GPU occupancy before choosing a physical GPU; set
CUDA_VISIBLE_DEVICES for each subprocess and use cuda internally. Keep log files,
wrapper/worker PIDs, state files and an exclusive lock to prevent duplicate runs.
Resume only with matching metadata and a supported existing latest checkpoint.
An automatic stage failure must stop that arm and preserve the diagnostic logs.
Record actual training/test completion separately from launch status.
