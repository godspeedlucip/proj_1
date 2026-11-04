# Phase 2 Marker-Guided nnU-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 Phase 1 预测 Marker 点生成软先验，在现有 nnU-Net v2.8.1 中完成二维支架 Mask 分割的可运行链路。

**Architecture:** 校验按 Phase 2 fold 导出的预测包，离线生成 `[projection, point, band, validity]` 四通道 TIFF。新增 E5-U 自定义网络/Trainer，在高分辨率 skip 上使用轻量残差先验适配器；保留 nnU-Net 原生 decoder、loss、训练和滑窗推理。

**Tech Stack:** Python 3.10+、PyTorch、NumPy、SciPy、tifffile、pytest、`phase_2/nnUNet` 中的 nnU-Net v2.8.1。单进程 CPU/GPU 应能完成小型 smoke test；真实尺寸训练需核对 GPU 内存。

**Spec:** [Phase 2 Spec](marker-guided-nnunet-codegen-spec.md)；输入遵守[交接契约](../../docs/phase1-phase2-handoff-design.md)。上游导出见 [Phase 1 Plan](../../phase_1/docs/marker-detection-implementation-plan.md)。

## Global Constraints

- 首版只覆盖 E5-U；E0/E1-P/E1/E2/E2-U 对照、校准 q、FiLM、E6 与拓扑损失后置。
- 有效点对允许 `q_pair=null`；第四通道只由 `pair_status` 得到 1/0，不能冒充校准概率。
- point/band 是软输入；弯曲支架可能离开直线 band，禁止硬 ROI、裁剪或硬掩膜。
- 只在 `phase_2/nnUNet/` 内增加自定义网络/Trainer，不直接修改默认 Trainer、planner、decoder、loss、predictor。
- Phase 2 fold 的训练先验来自 Phase 1 组折外预测；验证/测试先验来自只在允许训练组拟合的冻结流程。
- 合成数据 smoke 只证明软件链路；没有真实数据实验时不宣称分割性能收益。

## Review Focus

- `q_pair=null` 的有效点对和 `invalid` 零先验都应可消费、数值有限；Task 1、2 测试。
- `(case_id,view_id)`、尺寸、几何或来源错配必须报错；Task 1 测试。
- 曲线支架离开直线 band 时仍保留原始投影与完整标签；Task 1、4 测试。
- 预处理、增强、checkpoint 恢复和推理时第四通道语义一致；Task 3、4 测试。

## Planned Files

| 位置 | 职责 |
|---|---|
| `phase_2/tools/marker_guided/validate_dataset.py`, `build_prior_channels.py`, `cli.py` | 契约校验、四通道 TIFF 与 split 导出 |
| `phase_2/nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided/` | prior encoder、adapter、自定义网络/Trainer、raw-only 增强 |
| `phase_2/nnUNet/nnunetv2/tests/`, `phase_2/docs/marker-guided-runbook.md` | 单元/端到端 smoke 与运行说明 |

---

### Task 1: Phase 2 数据验证与 E5-U 四通道转换

**Files:** Create `phase_2/tools/marker_guided/{__init__,validate_dataset,build_prior_channels,cli}.py`, `phase_2/nnUNet/nnunetv2/tests/test_marker_prior_channels.py`。

**Interfaces:** `validate_prediction_package(manifest, package, phase2_fold_id)` 校验二元键、尺寸、几何、来源与 split；`build_marker_channels(view_record, shape, prior_config)` 返回 point/band/validity 三张 `[0,1]` 图；`build_e5u_dataset(...)` 将投影及先验写成 `{nnunet_case_id}_0000..0003.tif`，标签写 `labelsTr/{id}.tif`，生成 `dataset.json`、映射表和 group-wise `splits_final.json`。原图 `zscore`，三先验通道 `nonorm`；每个 Phase 2 fold 使用独立 Dataset ID。

- [ ] 写测试：亚像素高斯峰、端点外衰减、点交换不变、退化短线段零 band、`invalid` 全零、`uncertain` validity=1；错配 `(case_id,view_id)`、重复视角、尺寸/geometry 错误必须失败。
- [ ] 运行 `python -m pytest phase_2/nnUNet/nnunetv2/tests/test_marker_prior_channels.py -q`，确认失败。
- [ ] 实现严格校验与 TIFF 转换；先验参数及来源写入数据集侧元数据，Mask 只用于标签，不参与单样本先验生成。
- [ ] 复跑测试；读取实际写出的 TIFF/`dataset.json` 检查四通道顺序和 group-wise split。

### Task 2: nnU-Net 自定义网络

**Files:** Create `phase_2/nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided/{__init__,marker_guided_encoder,residual_prior_adapter,marker_guided_network}.py`, `phase_2/nnUNet/nnunetv2/tests/test_residual_prior_adapter.py`。

**Interfaces:** `MarkerGuidedPlainConvUNet` 按 nnU-Net 2D PlainConvUNet plans 构造只读通道 0 的 image encoder、轻量 point/band prior encoder、选定高分辨率 skip 的 `ResidualPriorAdapter` 和原生 decoder；`forward(x)` 消费四通道并返回符合深监督顺序的 logits。`gate_map` 来自第四通道：E5-U 有效=1、无效=0；adapter 依 Spec 的 `F'_l=F_l×[1+q_l β_l(2A_l−1)]`，初始化 `β_l=0.05`、attention 末层为零。

- [ ] 写测试：不同 plans stride/patch 下 skip 尺寸正确、无硬编码 stage、零 prior/零 gate 有限输出、`q_l=0` adapter 逐元素恒等、首次 backward 有梯度、深监督输出尺度正确。
- [ ] 运行 `python -m pytest phase_2/nnUNet/nnunetv2/tests/test_residual_prior_adapter.py -q`，确认失败。
- [ ] 实现网络；仅支持二维 PlainConvUNet plans，其他架构明确报错，所有实际使用的 prior 分支都在 forward 图中。
- [ ] 复跑测试并核对 nnU-Net v2.8.1 的真实 plans 对象构建路径。

### Task 3: 自定义 Trainer 与 raw-only 增强

**Files:** Create `phase_2/nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided/{marker_guided_trainer,marker_guided_transforms}.py`, `phase_2/nnUNet/nnunetv2/tests/test_marker_guided_transforms.py`。

**Interfaces:** `nnUNetTrainerMarkerGuidedFeatureFusion` 以静态方法覆盖与 v2.8.1 兼容的 `build_network_architecture(plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision)`，因为训练从实例调用、推理从 Trainer 类调用；该方法检查四通道及二维 plans，不依赖 `self.dataset_json`。训练增强入口保持本地源码签名，保留原生 loss/decoder/checkpoint/inference。另提供只缩短 epoch 和迭代数的 `nnUNetTrainerMarkerGuidedSmoke`。强度变换只作用通道 0，空间变换同步作用全部通道与分割标签。

- [ ] 写测试：原图强度变化时 point/band/validity 原值保持，空间翻转各通道与 Mask 同步，零先验产生有限 logits，checkpoint 构建时能通过 Trainer 名称找到同一网络。
- [ ] 运行 `python -m pytest phase_2/nnUNet/nnunetv2/tests/test_marker_guided_transforms.py -q`，确认失败。
- [ ] 根据本地 `nnUNetTrainer.py` 与 `predict_from_raw_data.py` 的静态构建签名及 `get_training_transforms` 签名实现子类；保持默认训练/验证流程，设置 `nnUNet_compile=false` 用于首版 smoke。
- [ ] 复跑测试；检查未改动 nnU-Net 默认 Trainer、planner、decoder、loss、predictor。

### Task 4: 真实 nnU-Net 命令端到端 smoke 与运行说明

**Files:** Create `phase_2/nnUNet/nnunetv2/tests/integration_tests/test_marker_guided_smoke.py`, `phase_2/docs/marker-guided-runbook.md`；必要时补充前述 CLI 文件的小型参数。

**Interfaces:** 从 Phase 1 计划 Task 1 的合成三视角病例和其 Task 4 的 fold 预测包，调用本计划 Task 1 转换器生成 E5-U Dataset；使用本地 nnU-Net 控制台入口完成 `plan_and_preprocess → train 2d fold 0 → checkpoint reload → predict`。运行文档写明 `pip install -e phase_2/nnUNet`、`nnUNet_raw/nnUNet_preprocessed/nnUNet_results`、数据集 ID、CPU/GPU 选择和真实数据替换路径。

- [ ] 写集成测试：检查四通道 raw/preprocessed 数据、训练至少一步、恢复 checkpoint、滑窗预测并导出对应二维 Mask；同一病例三视角保持同组，`invalid` 样本仍保留原投影。
- [ ] 运行 `python -m pytest phase_2/nnUNet/nnunetv2/tests/integration_tests/test_marker_guided_smoke.py -q`，确认缺失实现导致失败。
- [ ] 补齐 smoke Trainer、CLI 参数及 runbook，执行仓库实际 `nnUNetv2_plan_and_preprocess`、`nnUNetv2_train ... -tr nnUNetTrainerMarkerGuidedSmoke -device cpu`、`nnUNetv2_predict ... -tr nnUNetTrainerMarkerGuidedSmoke -device cpu`；验证 `splits_final.json` 在预处理后的正确位置并被训练读取。
- [ ] 复跑上述集成测试与 Phase 1/Phase 2 关键单测，保存命令输出和 checkpoint/Mask 路径；若环境缺少 nnU-Net 依赖或真实数据，明确区分软件 smoke、真实数据运行与性能评价的完成状态。

## Completion Check

- 可消费 Phase 1 预测包，生成正确四通道二维 TIFF 和组划分，并由新增 nnU-Net Trainer 完成 plan、preprocess、短训练、checkpoint 恢复与滑窗推理。
- 合成病例涵盖 `invalid` 和弯曲 Mask 超出直线 band；关键测试通过。
- 真实数据的文件格式、投影几何、二维 Mask 和运行环境由接入时验证；消融和性能提升不属于本计划完成条件。
