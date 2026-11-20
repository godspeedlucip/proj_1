# Phase 2：独立 U-Net 先预测 mask，再预测连续投影

每张训练图像对应两张对齐的标签：`0/1` 支架 mask，以及由 `phase1_200/stent_models` 投影并按训练集统一尺度归一化到 `[0,1]` 的连续值图。mask 定义为连续值大于零的位置。“深度图”在这里指连续投影值，不是物理距离。数据包位于 `/data1/liuyang/stent/proj_1/data/training/unet/phase2/phase2_mask_depth_v1/raw`，含 5250 张训练视角、750 张留出支架系列的验证视角和 240 组共享标签；没有独立测试划分。原图完整输入；1056×1056 可被网络 16 倍步幅整除，因此没有裁剪或额外填充。

新模型见 `cascade_model.py`，完全由 PyTorch 层实现，不调用 nnU-Net 网络或载入 nnU-Net checkpoint。nnU-Net 的 2D plan 只用来读取主干通道配置；mask U-Net 使用 5 层编码器、InstanceNorm、LeakyReLU 和 skip connection。Phase 1 的 marker 点热图经独立小分支注入前两层；不使用人为连接两点的直线 band。连续值 U-Net 也有 5 层，只接收 `原图 × 预测 mask 概率` 与预测 mask 概率；最终连续投影等于其输出乘以 mask 概率。

训练首先只更新 mask U-Net 20 轮，使用加权 BCE 和 Dice；此后联合训练两阶段，对条件连续值使用前景 Smooth L1，并惩罚背景泄漏。联合阶段的 mask 学习率是连续值分支的 0.1 倍。训练和推理都把**预测** mask 输入第二阶段，没有使用真值 mask 代替预测。最终的 0/1 mask TIFF 在概率 0.5 处阈值化，连续投影 TIFF 保留浮点值。前 20 轮的连续值分支仍是随机初始化，期间只根据 mask 指标判断进展。

## 训练

先按仓库根目录 `AGENTS.md` 检查 GPU 占用。下面命令使用物理 GPU 2；如果占用变化，直接修改 `--device`。正式实验结果目录为 `standalone_mask_depth_full_v1`，同一目录只能有一个训练进程。

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/segment/phase2/regression_unet/launch.py train \
  --dataset-dir /data1/liuyang/stent/proj_1/data/training/unet/phase2/phase2_mask_depth_v1/raw \
  --output-dir /data1/liuyang/stent/proj_1/data/training/unet/phase2/phase2_mask_depth_v1/results/standalone_mask_depth_full_v1 \
  --model-family mask_then_depth --mask-warmup-epochs 20 --mask-lr-factor 0.1 \
  --full-image --device cuda:2 --epochs 1000 --schedule-epochs 1000 \
  --steps-per-epoch 250 --val-steps 50 --batch-size 8 --num-workers 4 \
  --real-frames-dir /data1/liuyang/stent/proj_1/data/real_data/head_with_stent/head_2_over_frames \
  --real-markers-json /data1/liuyang/stent/proj_1/output/models/phase1/single_view_200_run1/real_all_epoch1/head_2_over/predictions.json \
  --real-samples frame_0001 frame_0300 frame_0600
```

从 `checkpoint_latest.pt` 恢复时运行完整命令并加 `--resume`：

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/segment/phase2/regression_unet/launch.py train \
  --dataset-dir /data1/liuyang/stent/proj_1/data/training/unet/phase2/phase2_mask_depth_v1/raw \
  --output-dir /data1/liuyang/stent/proj_1/data/training/unet/phase2/phase2_mask_depth_v1/results/standalone_mask_depth_full_v1 \
  --model-family mask_then_depth --mask-warmup-epochs 20 --mask-lr-factor 0.1 \
  --full-image --device cuda:2 --epochs 1000 --schedule-epochs 1000 \
  --steps-per-epoch 250 --val-steps 50 --batch-size 8 --num-workers 4 \
  --real-frames-dir /data1/liuyang/stent/proj_1/data/real_data/head_with_stent/head_2_over_frames \
  --real-markers-json /data1/liuyang/stent/proj_1/output/models/phase1/single_view_200_run1/real_all_epoch1/head_2_over/predictions.json \
  --real-samples frame_0001 frame_0300 frame_0600 --resume
```

checkpoint 包含模型、优化器、AMP 与随机数状态，恢复时核对数据清单、模型配置、真实帧来源等。向训练主 PID 发 `SIGUSR1` 会在当前轮保存后退出。`metrics.jsonl` 和 TensorBoard 记录 mask Dice、precision、recall、连续值前景 MAE、背景输出、各损失项和两阶段学习率。每轮在 `real_predictions/epoch_XXXX` 保存三张真实帧的最终投影、`masks` 中的 0/1 图、`mask_probabilities` 与 `conditional_depth` 中的浮点图。真实帧没有标签，叠加图只供定性检查。

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/segment/phase2/regression_unet/launch.py tensorboard \
  --logdir /data1/liuyang/stent/proj_1/data/training/unet/phase2/phase2_mask_depth_v1/results/standalone_mask_depth_full_v1/tensorboard \
  --host 127.0.0.1 --port 6006
```

单图推理参数及测试命令：

```bash
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/segment/phase2/regression_unet/launch.py predict --help
/data1/liuyang/conda/envs/nnunet/bin/python /data1/liuyang/stent/proj_1/segment/phase2/regression_unet/launch.py test -q
```
