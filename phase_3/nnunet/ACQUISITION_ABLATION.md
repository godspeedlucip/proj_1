# Phase 3 采集角度消融（2026-11-17）

本轮只改变 FDK 输入的采集跨度和投影数，使用当前 Phase 2 的完整病例划分。
数据来自理想仿真支架射线积分，与旧 Dataset601 的输入类型一致。
它评估采集条件对 Phase 3 的影响；不包含实际 Phase 2 预测误差。

## 优先配置

| 标识 | 跨度 | 投影数 | 中心方向为 30° 时的实际角度 | 用途 |
|---|---:|---:|---|---|
| a60_n3 | 60° | 3 | 0°, 30°, 60° | 必须重新训练的同划分基线 |
| a45_n3 | 45° | 3 | 7.5°, 30°, 52.5° | 缩小采集跨度 |
| a75_n3 | 75° | 3 | −7.5°, 30°, 67.5° | 扩大采集跨度 |
| a60_n2 | 60° | 2 | 0°, 60° | 少一张投影的影响 |
| a60_n5 | 60° | 5 | 0°, 15°, 30°, 45°, 60° | 增加投影的收益，包含基线方向 |

先做这五组可以分别观察跨度和投影数的主效应。完整 3×3 组合需要九次训练，
适合后续研究两因素的交互作用。更极端的 30°/3、90°/3 可作为第二优先级。
投影数增加在本轮没有配套改变每张投影噪声，因此不能直接解读成固定剂量下的收益。

每个病例保留旧三视角的中心方向，以精确浮点角度重新进行 TIGRE 正投影和 FDK。
负角度保留为连续角度，不做取整、图像插值或重新排序导致的跨 0° 断点。
保留原始衰减系数（支架与 marker 分别读取源 projection.json）、几何和完整体素网格。
辅助投影损失仍在 canonical patch 的 0°/30°/60° 方向计算，所有组相同。

## 完整数据与划分

来源：

- `training_data/training/generated/phase1_200/phase1/manifest.json`
- `training_data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/raw/Dataset503_MarkerGuided_fold_0/case_mapping.json`

生成前逐条核对 5820 条原始 Phase 2 视角记录与病例、组、角度和划分。

| 划分 | 物理头部/支架组合 | 每组配置的采集记录数 | 不同的理想 FDK 体积 |
|---|---:|---:|---:|
| 训练 | 168 | 1680 | 70 |
| 验证 | 24 | 240 | 10 |
| 测试 | 2 | 20 | 20 |

训练组为 bent_stent、ring_stent、sin_stent、stent1、stent3、stent4、wall_stent；
验证组为 stent2；测试组为 stent5、w_stent（topcow_ct_027）。
全部病例 ID 均保留；相同理想投影产生的重复数据只共享不可变文件。
20 个测试采集记录来自两个支架模型，不能视作 20 个独立生物学样本。
原始 FDK 与标签均保留完整 `512×512×533` 网格；预处理不围绕支架裁剪。
训练时的 `192³` 是从完整体积抽取的更新 patch，验证和测试在完整网格上进行。

旧 dataset601_adapter 有 80 条记录，清单引用的 80 个输入图像路径均已失效；
它还使用旧的 wall_stent 测试划分。新基线不能直接复用旧权重或旧测试结果。

## 训练与测试

五组使用同一份 full GDL 配置：Swin SSL 冻结先验、融合、tubular refinement、
GDL+BCE、权重 0.1 的辅助投影损失；192³ patch、batch 2、前景采样概率 0.9、
seed 2026、每轮 250 更新、70 轮共 17500 更新。各组从头训练。
每组的归一化统计只使用其训练集。原生 nnU-Net planner 对实际数据运行并保存报告，
模型容量与 patch 预算保持一致。

正式训练前，每组分别完成完整数据验证、预处理和独立 smoke：正式 patch/batch 的
两次更新、一轮完整体积验证、checkpoint 重载、完整验证体积推理及指标检查。
Smoke 使用每个支架组的一条代表记录，不代替正式数据；其指标不用于挑选实验配置。

正式训练结束后，自动加载 validation group Dice 最佳 checkpoint，对全部 20 条测试记录
推理并保存 Dice、IoU、precision、recall、假阳性体积、骨架距离、概率 MSE/MAE 和耗时。
五组全部成功后生成 `comparison.json` 与 `paired_differences.csv`，按同一病例 ID
计算相对 a60_n3 的差值。测试结果不参与模型、阈值或后处理选择。

## 路径和执行

项目入口：
`/data1/liuyang/stent/proj_1/training_data/training/nnunet/phase3/acquisition_ablation_20261117`

实际存储：
`/data2/liuyang/stent/proj_1/phase3_acquisition_ablation_20261117`

采用 /data2 是因为启动时 /data1 仅余约 27 GB。`run.json` 保存清单哈希、配置矩阵、
训练预算及代码快照哈希；实际执行的代码快照以 `run.json` 的 `code_root` 为准。
规划器线程数修复及验证图选片修复后的正式训练快照是
`code/nnunet_preview_fix_20261117`；`autoplan_fix_audit.json` 和
`preview_fix_audit.json` 保存两次修复的原因及文件哈希。
每组有 raw、prepared、smoke、predictions、metrics，以及日志、PID、status.json。
当前 `formal_scheduler.pid` 对应的后台调度进程会在每组自己的 smoke 通过后，
立即按实际空闲 GPU 启动该组正式训练；优先使用物理 GPU 4、5，允许多组并行。
`formal_scheduler_status.json` 记录等待、运行、完成或失败。
旧的 `formal_continuation.pid` 所属进程已停止，不能用于判断当前运行状态。
若任一 smoke 失败，正式训练不会启动。`progress_latest.json` 和
`progress_30min.jsonl` 每 30 分钟记录 checkpoint、TensorBoard loss、验证图像及 GPU 占用；
最终 `formal_artifact_audit.json` 核验全部 70 轮的 checkpoint、验证图和 loss 事件。
`formal_scheduler.log` 与各组的 `formal_worker.log`、`train.log`、`predictions/`、`metrics/`
留存实际执行证据。

查看状态：

```bash
/data1/liuyang/conda/envs/phase3/bin/python /data1/liuyang/stent/proj_1/model_networks/phase3/phase3-nnunet/phase_3/nnunet/acquisition_launch.py status --run-dir /data2/liuyang/stent/proj_1/phase3_acquisition_ablation_20261117
```

完整运行或恢复命令（先检查现有 supervisor/worker，运行中无需再次执行）：

```bash
/data1/liuyang/conda/envs/phase3/bin/python /data1/liuyang/stent/proj_1/model_networks/phase3/phase3-nnunet/phase_3/nnunet/acquisition_launch.py run --run-dir /data2/liuyang/stent/proj_1/phase3_acquisition_ablation_20261117 --gpus 0 1 2 4 5
```

GPU 顺序对应表格中的五组。设备可见性由入口为每个子进程单独设置，
子进程内使用 cuda:0。启动训练前等待所选卡空闲，不干预其他任务。
锁防止重复 worker；已有相同配置的 latest checkpoint 使用训练程序支持的恢复入口。
本 Trainer 没有通用 SIGUSR1 退出保证，不发送该信号。

结果未出现 `metrics/summary.json` 前，该组尚未完成正式测试。
后台进程启动、数据生成完成、短训练通过和正式实验完成是不同状态，
以 `status.json`、checkpoint 审计与实际指标文件为准。

## 方法来源

重建使用项目已有的 [TIGRE](https://github.com/CERN/TIGRE) 正投影和 FDK 实现。
本轮保持重建算法不变；五组配置的优先级来自上述受控实验设计，并非外部文献的效果结论。
