# Phase 2 直接回归 U-Net（历史实验）

当前的 mask→连续投影两阶段实验见 [MASK_THEN_DEPTH.md](MASK_THEN_DEPTH.md)。本文件记录此前直接回归实验。

2026-11-14 核对时，本文件所述外部 `phase2_stent_model_regression_v2`
数据目录已不存在。以下训练命令仅是历史记录，不能原样运行。

当前实验从带头部背景的 `phase1_200/stacked/.../projections/image/*.tif` 预测
`phase1_200/stent_models/.../stent_real_grid.nii.gz` 在同角度的二维射线积分。
该三维模型是 0/1 支架占据体，包含两端 marker 的体素；投影时所有支架体素使用
相同的 1.0 值，不再把 marker 单独赋予更高衰减系数。目标是连续 `float32` 图，
没有转换成二值 mask。使用**训练集投影最大值**统一除数并截断到 `[0,1]`，
验证集不参与尺度计算。输入为原图 z-score、marker 点图、marker band 和有效性图
四通道。当前训练先验来自几何真值。旧的高衰减 marker 标签保留在 v1 数据包中。

## 来自上次 nnU-Net 运行的配置

源文件：`training_data/training/nnunet/phase2/preprocessed/Dataset501_MarkerGuided_fold_0/nnUNetPlans.json`，
已用训练日志核对。

| 项目 | 上次配置／本实现默认值 |
|---|---|
| 网络 | 2D `PlainConvUNet` 图像主干，9 层；通道数 32、64、128、256、512、512、512、512、512 |
| 卷积 | 每层 2 个 3×3 卷积；InstanceNorm2d、LeakyReLU；逐层 2 倍下采样 |
| 输入／输出 | 原图 1 通道进入主 encoder；点图、band 进入独立 prior encoder；validity 控制融合；输出 1 通道浮点图，无深监督 |
| patch／batch | 1280×1024／2 |
| 原图归一化 | 全图 z-score；其余先验通道不归一化 |
| 优化 | SGD，初始学习率 0.01、momentum 0.99、Nesterov、weight decay 3e-5、poly 指数 0.9 |
| 训练预算 | 1000 epoch、每轮 250 训练步、50 验证步，33% 前景中心裁剪 |

上表记录原 nnU-Net 配置。当前整图训练使用 `--full-image`，读取完整的
1056×1056 输入并在右侧和下侧补零到 1280×1280；不使用前景中心裁剪。
batch 仍为 2，其余网络和优化参数沿用上表。

回归任务改用前景加权 Smooth L1（默认权重 100，`beta=0.05`），验证记录前景强度 MAE、
背景平均预测值和加权验证损失。输出头为线性层；导出 TIFF 时截断到 `[0,1]`。
prior encoder 使用两层卷积，在前两级高分辨率 skip 上通过 E5-U 的残差 attention adapter
调制图像特征；validity 为 0 时该调制恒等。网络直接复用 Phase 2 nnU-Net 的
`MarkerGuidedPlainConvUNet` 实现，将原二类分割头改为单通道回归头。
保存的 checkpoint 是本脚本的 PyTorch 格式，不能传给 `nnUNetv2_train --c`。
`run_config.json` 的 `model_family=marker_fusion` 标识新结构；没有该字段的旧版
普通 U-Net checkpoint 仍可通过推理入口加载，但不能与新结构互相续训。

## 使用

GPU 检测、选卡及避免重复长训练的步骤遵循仓库根目录 `AGENTS.md`。
下面的命令使用绝对路径和 `--device cuda:0`，可从任意工作目录直接运行；
启动前先确认 GPU 0 空闲。`launch.py` 在程序内部定位额外 Python 依赖。
历史数据包曾位于 `/data1/liuyang/stent/proj_1/training_data/training/generated/phase2_stent_model_regression_v2`，
含 6000 张输入、240 张由三维支架模型生成的共享连续投影标签及对应先验。
输入和先验文件与 v1 数据包使用硬链接，目标文件为 v2 独立生成；请勿原地修改
硬链接文件。删除 v1 不会使 v2 的硬链接失效。

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/model_networks/phase2/regression_unet/launch.py train \
  --dataset-dir /data1/liuyang/stent/proj_1/training_data/training/generated/phase2_stent_model_regression_v2 \
  --full-image \
  --output-dir /data1/liuyang/stent/proj_1/model_outputs/models/phase2_unet_regression/stent_model_full_realmonitor_v1 \
  --device cuda:0 --epochs 1000 --schedule-epochs 1000 \
  --steps-per-epoch 250 --val-steps 50 --num-workers 4 \
  --real-frames-dir /data1/liuyang/stent/proj_1/training_data/real_data/head_with_stent/head_2_over_frames \
  --real-markers-json /data1/liuyang/stent/proj_1/test_results/phase1/single_view_200_run1/real_all_epoch1/head_2_over/predictions.json \
  --real-samples frame_0001 frame_0300 frame_0600
```

从同一结果目录的 `checkpoint_latest.pt` 续训，执行以下完整命令：

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/model_networks/phase2/regression_unet/launch.py train \
  --dataset-dir /data1/liuyang/stent/proj_1/training_data/training/generated/phase2_stent_model_regression_v2 \
  --full-image \
  --output-dir /data1/liuyang/stent/proj_1/model_outputs/models/phase2_unet_regression/stent_model_full_realmonitor_v1 \
  --device cuda:0 --epochs 1000 --schedule-epochs 1000 \
  --steps-per-epoch 250 --val-steps 50 --num-workers 4 \
  --real-frames-dir /data1/liuyang/stent/proj_1/training_data/real_data/head_with_stent/head_2_over_frames \
  --real-markers-json /data1/liuyang/stent/proj_1/test_results/phase1/single_view_200_run1/real_all_epoch1/head_2_over/predictions.json \
  --real-samples frame_0001 frame_0300 frame_0600 --resume
```

可将 `--epochs` 增大到固定的
`--schedule-epochs` 上限；调度总轮数、数据清单哈希、marker JSON 与所选真实帧哈希、模型配置、
batch 和步数会在续训时核对。checkpoint 原子写入，包含模型、优化器、AMP 和随机
数状态；`checkpoint_best.pt` 只用于评估。对训练主 PID 发 `SIGUSR1` 可请求在
当前 epoch 结束、保存 checkpoint 和真实帧输出后安全退出，之后加 `--resume`。

每轮会把三张完整真实帧的连续预测保存到
结果目录下的 `real_predictions/epoch_XXXX/frame_*.tif`，并保存 `summary.json`。
TensorBoard 位于结果目录的 `tensorboard` 子目录，记录训练/验证损失、验证前景 MAE、
背景预测均值、学习率，以及真实帧的预测图、与原图和 marker 点的叠加图。
终端逐轮显示训练、验证和真实帧推理的进度条；训练条显示当前平均损失和学习率，
验证条显示当前平均损失与前景 MAE。
叠加图中红色强度对应预测值，青色圆点是 Phase 1 预测的 marker。
真实帧没有投影真值，图像与输出强度仅供定性观察。

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/model_networks/phase2/regression_unet/launch.py tensorboard \
  --logdir /data1/liuyang/stent/proj_1/model_outputs/models/phase2_unet_regression/stent_model_full_realmonitor_v1/tensorboard \
  --host 127.0.0.1 --port 6006
```

默认使用 `phase1_200/phase1/physical_manifest.json` 的 `fold_0`：7 个支架系列
5250 张训练视角，留出 `stent2` 的 750 张视角验证。checkpoint、配置和逐轮指标分别在
`checkpoint_latest.pt`、`checkpoint_best.pt`、`run_config.json`、`metrics.jsonl`。
`run_config.json` 和 checkpoint 都记录训练集目标强度除数 `target_scale`；
如需还原原始射线积分，可将输出乘以此值。
重复运行同一目录而未加 `--resume` 会报错，以免覆盖训练结果。

单张 TIFF 推理使用同一入口的 `predict` 子命令；用下面的命令查看所需参数：

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/model_networks/phase2/regression_unet/launch.py predict --help
```

## 验证及局限

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/model_networks/phase2/regression_unet/launch.py test -q
```

训练图像里叠加的是经过 `sigma=1.2 px` 平滑的植入物投影，而目标是从三维支架体
重新生成、没有单独增强 marker 的原始投影。
因此本任务包括恢复细节；高 MAE 不一定只来自定位误差。验证集留出支架系列，仍共享
训练集使用的 25 个 CT 背景，不能代表完全独立的真实影像性能。真实帧没有投影真值，
其结果只能定性检查。

## 2026-11-11 一轮验证

新 v2 真值的整图实验位于
`model_outputs/models/phase2_unet_regression/epoch1_stent_model_full_20261111`。使用 GPU 0、
完整 1056×1056 输入、1280×1280 零填充、batch 2，完成 250 个训练步和 50 个
验证步。训练损失 `0.080593`，验证损失 `0.059072`，验证前景 MAE `0.221337`，
背景平均预测值 `0.018692`。最佳 checkpoint 已重新加载，并对一张完整的留出图像
生成 `validation_sample_prediction.tif`；输出为 1056×1056，样本统计在
`validation_sample_summary.json`。单轮结果仍明显低估支架强度，需要更长训练才可
判断模型效果。

以下为旧 v1 高衰减 marker 标签的历史结果，不可与新 v2 真值的损失直接比较。

普通 U-Net 基线结果目录：`model_outputs/models/phase2_unet_regression/epoch1_normalized_20261111`。GPU 0 上完成
250 个训练步、50 个验证步；训练损失 `0.015653`、验证损失 `0.007226`、验证前景 MAE
`0.037635`、背景平均预测值 `0.006080`，数值均处于归一化尺度。最佳 checkpoint 已重新
加载并在一张留出验证图上生成 `validation_sample_prediction.tif`；对应统计见
`validation_sample_summary.json`。这些是合成验证结果，不代表真实帧精度。

Marker 融合版本使用相同数据划分与 250/50 步预算，结果在
`model_outputs/models/phase2_unet_regression/epoch1_marker_fusion_20261111`：训练损失 `0.031835`、
验证损失 `0.015855`、验证前景 MAE `0.062512`、背景平均预测值 `0.012644`。
在一张留出验证图上，以相同权重关闭 validity 门控后，有 28008 个像素变化超过 `1e-4`，
对应 `marker_gate_ablation_sample.json` 和两张 TIFF。该单样本检查证明输出响应 marker，
不构成整体性能提升证据；一轮预算下融合版本的验证指标低于普通 U-Net 基线。
