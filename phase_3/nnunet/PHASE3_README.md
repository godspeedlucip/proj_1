# Phase 3 nnU-Net migration

2026-11-14 数据整理后，早期 `compare_plain_vs_unetr` 与
`gdl_full_fov_3epoch` 的数据缓存已删除；下文相关命令为历史记录。
当前 Dataset603 的 raw、共用 prepared 缓存及结果集中位于
`data/training/nnunet/phase3/Dataset603_Phase3FDKThree/`；旧绝对路径的
checkpoint 摘要通过该目录 `raw/adapter/relocation.json` 核验。

## 可切换的极度不平衡损失

`segmentation.loss` 支持 `foreground_dice_bce`（原行为）和
`generalized_dice_bce`。后者在每个病例、每个非零权重的深监督尺度上，
对背景和前景计算按目标体积平方倒数加权的 Generalized Dice Loss；
BCE 保持原先按有效体素均值的定义。标签 `2` 不参与体积、
Dice 或 BCE；无前景 patch 只计算背景，辅助尺度全为 ignore 时贡献零。
类别权重按病例归一化，以免大 patch 的小权重让平滑项主导结果。

可以用 `full_gdl` 预设，或者在现有 JSON 配置中只把
`segmentation.loss` 改为 `generalized_dice_bce`。投影辅助项仍使用原投影
均方误差。切换损失会改变配置摘要，
因此应为 GDL 建立新的 prepared
fold 和结果目录；不能在原损失的 checkpoint 上使用 `--resume`。
比较两种损失时保持数据划分、采样率和其他配置相同，在有标签的完整体积
验证集上比较 Dice 与召回率。GDL 是待验证的训练策略，不保证单独解决
全背景预测。

### 实际前景占比与训练曲线（2026-11-12）

Dataset903 的 192³ 全视野训练在 8 个已完成 epoch 中，完整体积验证 Dice
和召回率均为 0，预测前景为空。Dataset907 的 160×64×64 紧 ROI 训练继续
使用原 `foreground_dice_bce` 和相同的 0.9 前景定位概率；ROI 完整体积验证 Dice 从
epoch 1 的 0.765 上升到 epoch 17 的 0.923，此后在约 0.91～0.92 间波动。
Dice 不会逐 epoch 单调上升，模型选择仍以最佳完整体验证 Dice 为准。

用各自正式配置的真实训练 loader、同一采样种子各抽取 20 个 patch：
全视野 patch 有 90% 含前景，中位前景体素数 2446、中位前景占比 0.0346%；
紧 ROI patch 全部含前景，中位前景体素数 2582.5、中位前景占比 0.978%。
全视野四级深监督标签的前景存活率为 90%、90%、85%、85%；紧 ROI 四级
均为 100%。结合验证曲线，应优先优化有效视野及 patch 内前景占比，
目前没有证据支持把前景定位概率从 0.9 降到 0.5，或把 GDL 当作已验证的
主训练损失。紧 ROI 同时改变了重建视野，单次比较不能隔离各因素的贡献。

紧 ROI 用真实三维标签包围盒生成，是 oracle 可行性实验；无标签推理需要
独立的 ROI 定位器，并在原始全视野网格验证。当前验证组的 75 个病例 ID
只有 3 个不同的角度体积，指标不能当作 75 个独立验证样本。

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
environment for all commands below. The 16 self-contained JSON configurations
are under `configs/phase3`; `smoke` is for software checks only, and `full_gdl`
selects the imbalance-aware loss. The eight
formal combinations are `baseline`, `swin`, `tubular`, `projection`,
`swin_tubular`, `swin_projection`, `tubular_projection`, and `full`.

## Current Linux stent dataset

The source at `/data1/liuyang/stent/proj_1/output/data/phase3_from_phase2`
contains 25 heads, eight stent categories per head, and ten available
three-view angle sets per head/stent. The current selection exports three
angle sets per head/stent: starts at 0°, 100°, and 230°, with 30° between
views within each set. This gives 600 case IDs: 450 training, 75 validation,
and 75 test. Labels contain both stent struts and markers. The exported data
keep the source voxel size of about 0.188946 mm without resampling.

The original 25 heads reuse the same eight stent meshes, and matching
head/stent/angle cases are hard links to identical FDK and label volumes.
There are 24 distinct FDK volumes among the 600 case IDs. The prepared
dataset keeps every case ID and shares identical channel and Blosc2 files
through hard links to avoid wasting disk space. The effective number of
independent volumes is 24, which matters when interpreting validation scores.

The export is at
`/data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/raw`.
Its adapter manifest and group split are at
`/data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/raw/adapter`.
To reproduce the export and adapter on Linux:

```bash
export PYTHONPATH=/data1/liuyang/stent/proj_1/output/phase2_nnunet/deps:/data1/liuyang/stent/proj_1/segment/phase3
/data1/liuyang/conda/envs/nnunet/bin/python -m phase_3.export_nnunet \
  --source /data1/liuyang/stent/proj_1/output/data/phase3_from_phase2 \
  --output /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/raw \
  --triplets-per-head-stent 3
export PYTHONPATH=/data1/liuyang/stent/proj_1/segment/phase3/phase3-nnunet/phase_3/nnunet:/data1/liuyang/stent/proj_1/output/phase2_nnunet/deps
/data1/liuyang/conda/envs/nnunet/bin/python -m nnunetv2.phase3.cli import-fdk-dataset \
  --source /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/raw \
  --config full \
  --output /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/raw/adapter
```

These export commands require new output directories. The exported data and
prepared fold already exist for the current run; use the nnU-Net-planned
configuration below for training.

The 600-case dataset is prepared at
`/data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/prepared/prepared.json`.
The old generated `phase3_config.json` files and 128³ preprocessing cache were
removed. A fresh config was generated from the installed nnU-Net planner with
a 24 GB VRAM target on the 48 GB RTX A6000. It selects a 192³ patch and batch
size 2 at the original 0.188946 mm spacing. Sparse-target foreground sampling
uses probability 0.9. The report is
`raw/adapter/phase3_config_nnunet_24gb_fg09.plan.json`. Phase 3 keeps its
five-stage fusion network; it takes the patch size from the native planner.
This plan starts a new model folder and cannot resume the old 128³ checkpoint.

To regenerate a config from an existing prepared Dataset903 fold:

```bash
conda activate phase3
cd /data1/liuyang/stent/proj_1/segment/phase3/phase3-nnunet/phase_3/nnunet
python -m nnunetv2.phase3.autoplan \
  --prepared /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/prepared/prepared.json \
  --config /data1/liuyang/stent/proj_1/segment/phase3/phase3-nnunet/phase_3/nnunet/configs/phase3/full.json \
  --gpu-memory-target-gb 24 \
  --foreground-probability 0.9 \
  --output /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/raw/adapter/phase3_config_nnunet_24gb_fg09.json
```

The output path must be new; the command will not overwrite an existing config.
nnU-Net selects patch size from image shape, spacing, topology, and VRAM
budget. Foreground occupancy does not directly determine patch size. The
loader's foreground oversampling probability for this run is 0.9.

Inference accepts the full 533×512×512 image and returns a full-size mask.
The predictor computes it with overlapping sliding windows. A single
full-resolution forward pass would require much more GPU memory than the
48 GB card, especially during training with gradients.

The Linux training environment is `conda activate phase3`. It uses Python
3.10, PyTorch 2.3.1+cu118, NumPy 1.26.4, MONAI 1.3.0, and this nnU-Net fork
installed in editable mode. To install or refresh it from this directory:

```bash
cd /data1/liuyang/stent/proj_1/segment/phase3/phase3-nnunet/phase_3/nnunet
conda activate phase3
python -m pip install --no-build-isolation 'numpy==1.26.4' 'monai==1.3.0' -e .
python -m pip install pytest
python -m pip check
```

For the `full` preset, check GPU availability, prepare the SSL prior artifact,
then train with the nnU-Net-planned original-spacing configuration for Dataset603.
The example selects GPU 5; inspect the GPU process list and choose an available
GPU for the actual run.

```bash
conda activate phase3
cd /data1/liuyang/stent/proj_1/segment/phase3/phase3-nnunet/phase_3/nnunet
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
python -m nnunetv2.phase3.cli prepare-prior \
  --download-official \
  --output /data1/liuyang/stent/proj_1/output/phase3_nnunet/phase3_full_prior.json
CUDA_VISIBLE_DEVICES=5 python -m nnunetv2.phase3.cli train \
  --prepared /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/prepared/prepared.json \
  --config /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/raw/adapter/phase3_config_nnunet_24gb_fg09.json \
  --device cuda \
  --prior-artifact /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/prepared/prior/phase3_full_prior.json
```

If the official download is unavailable, pass `--weights /absolute/path/model_swinvit.pt`
instead of `--download-official`. To continue from the latest checkpoint,
repeat the `train` command with the same prepared and config paths, replace
`--prior-artifact ...` with `--resume`, and keep the same GPU selection.

The full-resolution tubular refinement runs in FP32 inside CUDA autocast. This
prevents the final-scale Dice gradient from overflowing its FP16 normalization
and convolution backward path on sparse, empty-foreground patches. If another
AMP overflow occurs, the trainer logs `phase3_amp_overflow`, lets `GradScaler`
reduce its scale, and retries the same batch before counting an optimizer step.
Persistent nonfinite gradients still stop training after 16 scale reductions.
Detailed `phase3_step` and `phase3_amp_overflow` records go to
`fold_0/training_log_*.txt` without printing their JSON payloads to the console.
After every completed epoch, the trainer saves a full resumable checkpoint as
`fold_0/checkpoint_epoch_0001.pth`, `checkpoint_epoch_0002.pth`, and so on.
`checkpoint_latest.pth` and `checkpoint_best.pth` remain available for their
usual purposes. Epoch numbers count completed epochs, starting at 1.
The old 128³ checkpoint belongs to a different plan. Start this planned run
without `--resume`.

### TensorBoard and validation reports

Each completed epoch records training loss components, gradient norm and
learning rate, plus full-volume validation Dice, IoU, foreground-probability
MSE/MAE, precision, recall, false-positive volume, component counts and
skeleton distance. Metrics are available as overall means/standard deviations
and per stent group. The full per-case results are saved to
`fold_0/validation_metrics/epoch_XXXX.json`.

The validation pass selects one informative slice from each of the three
distinct validation angle volumes. TensorBoard displays each as four panels:
FDK input, marker-and-stent label, predicted mask, and overlay. PNG copies are
saved under `fold_0/validation_previews/epoch_XXXX/`. The preview reuses the
full-volume predictions already computed for metrics, so it does not run a
second inference pass. MSE is the mean squared error between foreground
probability and the binary target over the full original-grid volume.
When a sampled validation patch has neither foreground label nor foreground
prediction, patch Dice is undefined and is omitted from TensorBoard for that
epoch. Validation loss and full-volume metrics remain available; checkpoint
selection uses the full-volume group Dice.

```bash
conda activate phase3
python -m tensorboard.main \
  --logdir /data1/liuyang/stent/proj_1/data/training/nnunet/phase3/Dataset603_Phase3FDKThree/results/prepared603_initial/results/Dataset903_Phase3F00/nnUNetTrainerPhase3__FDKPhase3_b8a3562f0f7b__3d_fullres/fold_0/tensorboard
```

### Original nnU-Net trainer comparison

`PlainConvUNet` is the CNN architecture selected by the nnU-Net plan;
`nnUNetTrainer` is the original trainer that controls sampling, augmentation,
Dice+CE loss, optimizer steps, validation, and checkpointing. The older
`nnUNetTrainerPhase3` inherits from that trainer but overrides those training
methods. Its earlier `baseline` experiment was therefore a Phase 3 baseline,
not an original nnU-Net training run.

The native comparison entry point is `nnunetv2.phase3.native_baseline`:
`--mode plain` instantiates `nnUNetTrainer` itself and uses both input channels.
`--mode prior` instantiates `nnUNetTrainerPhase3NativePrior`, which only changes
network construction to add the frozen SSL encoder and fusion modules; it keeps
the original Dice+CE loss and trainer methods. `--mode full` adds only
`Phase3CompositeLoss` to that network. The prior weights are loaded before the
first optimizer step. Existing `nnUNetTrainerPhase3` checkpoints are separate
from these native trainer classes.

Omitting `--epochs`, `--steps-per-epoch`, `--validation-steps`, and
`--save-every-epoch` keeps the original trainer defaults of 1000 epochs, 250
training steps and 50 validation steps per epoch, and the original checkpoint
interval. Short comparison runs may override the budget explicitly; this does
not replace the trainer's optimizer step or augmentation. The prepared native
comparison folders use the original 0.33 foreground oversampling setting.

```bash
conda activate phase3
cd /data1/liuyang/stent/proj_1/segment/phase3/phase3-nnunet/phase_3/nnunet
CUDA_VISIBLE_DEVICES=0 nnUNet_compile=false nnUNet_n_proc_DA=0 python -m nnunetv2.phase3.native_baseline \
  --prepared /data1/liuyang/stent/proj_1/output/phase3_nnunet/compare_plain_vs_unetr/prepared_native_plain/prepared.json \
  --mode plain --device cuda --epochs 3 --steps-per-epoch 200 --save-every-epoch
CUDA_VISIBLE_DEVICES=1 nnUNet_compile=false nnUNet_n_proc_DA=0 python -m nnunetv2.phase3.native_baseline \
  --prepared /data1/liuyang/stent/proj_1/output/phase3_nnunet/compare_plain_vs_unetr/prepared_native_prior/prepared.json \
  --mode prior --prior-artifact /data1/liuyang/stent/proj_1/output/phase3_nnunet/phase3_full_prior.json \
  --device cuda --epochs 3 --steps-per-epoch 200 --save-every-epoch
```

The original trainer logs patch validation Dice. Full-volume Dice can be
measured separately from a saved checkpoint without changing its training
loop. With `--save-every-epoch`, nnU-Net writes `checkpoint_latest.pth` after
each nonfinal epoch and replaces it at the next epoch; the final epoch writes
`checkpoint_final.pth`. The native comparison results live in separate model folders
named `nnUNetTrainer__...` and `nnUNetTrainerPhase3NativePrior__...`.
For `--mode prior` or `--mode full`, `--prior-artifact` is needed at the start;
`--resume` loads the prior weights from the checkpoint and can omit it.

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
32³ patch, batch 1, and four updates. Formal training has 10,000 updates;
the Dataset603 patch is planned from its geometry and a VRAM target. No formal
effect estimate is implied by smoke tests.

### Task 1 environment check (2026-11-11)

The checked environment uses Python 3.10.19, torch 2.3.1+cu118, NumPy 1.26.4,
MONAI 1.3.0 and nnU-Net 2.8.1 from this source tree. `pip check` found no
broken requirements. The configuration contract tests passed 11/11 and the
pinned upstream smoke tests passed 7/7. CUDA was visible on an NVIDIA GeForce
RTX 4070 Ti SUPER; model capacity remains untested. The venv inherits torch
and NumPy from `D:/python/miniconda/env/env/nnunet`; changes to that base
environment would require this check and the lock file to be regenerated.
