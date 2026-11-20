# Phase 2: 每轮真实投影分割与 TensorBoard

> 历史实验记录：Dataset501 的 checkpoint 已移至
> `data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/results/pretrained_from_Dataset501/`
> 并明确标注为 501 预训练权重。下面的旧结果路径不再直接包含 checkpoint；
> 这些命令用于理解当时的运行配置，不能原样续训 Dataset501，也不代表已训练 Dataset503。

当前实验使用 `Dataset501_MarkerGuided_fold_0`。训练和验证是合成数据；真实数据没有分割标签，真实帧输出只用于定性检查，不能当作 Dice。Dataset501 的训练 marker 是投影几何真值，真实帧的 marker 则来自指定的 Phase 1 `predictions.json`。

从 `proj_1` 目录执行。`deps` 是本次实验单独安装的 nnU-Net 依赖目录，不改共享 conda 环境。首次训练去掉 `--c`；从 `checkpoint_latest.safetensors` 继续时保留 `--c`。

```bash
cd /data1/liuyang/stent/proj_1
export PYTHONPATH=/data1/liuyang/stent/proj_1/output/phase2_nnunet/deps:/data1/liuyang/stent/proj_1/segment/phase2:/data1/liuyang/stent/proj_1/segment/phase2/nnUNet
export nnUNet_raw=/data1/liuyang/stent/proj_1/output/data/phase2
export nnUNet_preprocessed=/data1/liuyang/stent/proj_1/output/phase2_nnunet/preprocessed
export nnUNet_results=/data1/liuyang/stent/proj_1/output/phase2_nnunet/results
export nnUNet_compile=false
export nnUNet_n_proc_DA=4
export PHASE2_REAL_FRAMES=/data1/liuyang/stent/proj_1/data/real_data/head_with_stent/head_2_over_frames
export PHASE2_REAL_MARKERS=/data1/liuyang/stent/proj_1/output/models/phase1/single_view_200_run1/real_all_epoch1/head_2_over/predictions.json
export PHASE2_REAL_SAMPLE=frame_0001
CUDA_VISIBLE_DEVICES=1 /data1/liuyang/conda/envs/nnunet/bin/python \
  -m nnunetv2.run.run_training 501 2d 0 \
  -tr nnUNetTrainerMarkerGuidedFeatureFusion -device cuda --c
```

监测器先按 `source` 绝对路径核对 600 张 `frame_0001.tif` 至 `frame_0600.tif` 和 600 条 marker 记录，每轮只分割 `PHASE2_REAL_SAMPLE` 指定的 `frame_0001`。point、band、validity 与训练数据使用同一生成函数和先验参数；`edge_candidate` 记录为 `uncertain`，E5-U 的二值 validity 仍为 1。来源 JSON、缓存配置和计数写入 `real_predictions/inputs/manifest.json`。marker 预测没有校准误差半径，当前使用 `[0,0]` 以保持训练先验宽度；这不表示检测点没有误差。

每轮结果位于 `nnUNet_results/Dataset501_MarkerGuided_fold_0/nnUNetTrainerMarkerGuidedFeatureFusion__nnUNetPlans__2d/fold_0/real_predictions/epoch_XXXX/`，含一张压缩的二值 TIFF 和 `summary.json`。TensorBoard 目录在同一 `fold_0/tensorboard/`。固定帧的叠加图中，红色为预测分割，青色为 Phase 1 marker 点；曲线包括训练和验证 loss、验证 pseudo Dice、学习率、真实帧前景比例和推理耗时。

```bash
PYTHONPATH=/data1/liuyang/stent/proj_1/output/phase2_nnunet/deps \
  /data1/liuyang/conda/envs/nnunet/bin/python -m tensorboard.main \
  --logdir /data1/liuyang/stent/proj_1/output/phase2_nnunet/results/Dataset501_MarkerGuided_fold_0/nnUNetTrainerMarkerGuidedFeatureFusion__nnUNetPlans__2d/fold_0/tensorboard \
  --host 127.0.0.1 --port 16006 --reload_interval 5
```

真实数据无标签，所以 `real/mean_foreground_fraction` 等只描述输出规模。`6006`–`6009` 端口当前由其他实验的 TensorBoard 占用，Phase 2 分割使用上面的 `16006`。

2026-11-11 14:47 新启动的 nnU-Net 训练曾与更早完成的实验复用结果目录，导致其第 4–68 轮真实帧监测误读旧文件。旧实验产物已按用户要求删除；当前训练第 1–3 轮真实帧结果已保留，从第 69 轮起重新写入真实帧分割和 TensorBoard 图像。新训练启动时若目录中已有逐轮真实帧结果，监测器会直接报错，防止再次静默复用。

当前训练第 68 轮 checkpoint 的三张真实帧独立推理位于 `output/phase2_nnunet/real_test_current_epoch68/`，包含 `preview.png`、`summary.json` 和二值 TIFF。关闭 marker 门控后，三张真实帧的二值 mask 分别改变 295、166、326 个像素；真实图像无标签，无法判断这些变化是否改善精度。

相同第 68 轮权重在 150 张有标签合成验证图上的配对消融位于 `output/phase2_nnunet/ablation_current_epoch68/summary.json`：开启 marker 的平均 Dice 为 0.938993，关闭门控为 0.938772，平均差值 +0.000221；按 50 个病例重采样的 95% 区间为 [+0.000106, +0.000335]。这是很小的正向差异，且验证集 marker 为几何真值，不能代表 Phase 1 预测点在真实数据上的精度收益。结果及真实帧预览已写入当前 TensorBoard 的 `ablation_current/*` 标签。

第 300 轮曾是最佳 checkpoint。相同 150 张合成验证图上，`output/phase2_nnunet/ablation_current_best_epoch300/summary.json` 记录 marker 开启 Dice 0.942892、关闭门控 Dice 0.942887，配对差值仅 +0.0000044，按 50 个病例重采样的 95% 区间为 [-0.0000184, +0.0000272]。与第 68 轮相比，总 Dice 提升约 0.003899，但第 300 轮的提升几乎都不依赖 marker 门控。相同权重对三张无标签真实帧关闭门控时，mask 仅分别改变 11、17、3 个像素；见 `output/phase2_nnunet/ablation_current_best_epoch300/real_summary.json` 和 `real_preview.png`。

训练于 2026-11-11 21:49 刷新第 425 轮最佳 EMA pseudo Dice 0.9424 后，固定权重的整图复测见 `output/phase2_nnunet/ablation_current_best_epoch425/summary.json`：marker 开启/关闭平均 Dice 为 0.941754/0.941903，差值 -0.000149（病例重采样 95% 区间 [-0.000191, -0.000109]）。该权重的整图 Dice 比第 300 轮权重低约 0.001138，说明按每轮随机 patch 计算的 EMA pseudo Dice 与固定整图评估排序不完全一致。真实 `frame_0001` 关闭门控仅改变 14 像素，见同目录 `real_summary.json`。两次消融均已写入 TensorBoard 的 `ablation_current/*` 标签。门控开关是同一已训练模型的输入消融，不等同于独立训练的无 marker 基线；真实帧没有标签，不能据此判断真实数据精度。
