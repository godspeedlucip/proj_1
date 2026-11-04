# Phase 1 Marker Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 60° 窗口内的三张支架投影定位每张图的两个 Marker，并按 Phase 2 fold 导出可追溯预测包。

**Architecture:** 单通道简单 2D U-Net 输出 Gaussian 热图；Top-K 峰值经三视角几何搜索得到最终二维点对，低视差时按冻结规则单视角回退。训练组按原始 3D 支架模型折外预测，验证/测试组使用冻结检测流程。

**Tech Stack:** Python 3.10+、PyTorch、NumPy、SciPy、tifffile、pytest。

**Spec:** [Phase 1 Spec](marker-detection-codegen-spec.md)；跨阶段输出遵守[交接契约](../../docs/phase1-phase2-handoff-design.md)。下游实施见 [Phase 2 Plan](../../phase_2/docs/marker-guided-nnunet-implementation-plan.md)。

## Global Constraints

- 首版只覆盖 S0+V1：简单 U-Net、完整图热图训练、峰值解码、几何选择与回退；响应图、残差、Patch 预训练、可学习热图融合与消融后置。
- 三视角属于同一静止状态，角度落在不超过 60° 的窗口；两个真实 Marker 中心均在视野内，预测 `invalid` 只表示算法失败。
- `group_id` 是原始 3D 支架模型；三视角不可拆组，检测器和候选几何参数只用当前 Phase 2 fold 允许的训练组确定。
- 原图坐标 `x=列、y=行`，保留亚像素；输入变换与几何一致，正式 JSON 严格遵守交接契约。
- 首版有效点对允许 `q_pair=null, calibration_id=null`；不把峰值分数冒充校准概率。
- 合成数据用于软件 smoke；真实定位性能待实际数据验证。

## Review Focus

- 两点接近或峰值合并：不能虚构第二个点；Task 3 测试。
- 角度跨越 0°（如 350°、0°、10°）：仍识别为 20° 窗口；Task 1 测试。
- 有效视差不足但每图有双候选：输出 `single_view_fallback`；Task 3 测试。
- 折外记录的模型与参数选择来源不得包含被预测的支架组；Task 4 测试。

## Planned Files

| 位置 | 职责 |
|---|---|
| `phase_1/marker_pipeline/schema.py`, `data.py`, `synthetic.py` | 病例和组划分校验、投影读取、合成 smoke 数据 |
| `phase_1/marker_pipeline/models/unet2d.py`, `heatmap.py`, `train.py` | 简单 U-Net、热图标签和训练/checkpoint |
| `phase_1/marker_pipeline/peaks.py`, `geometry.py`, `fusion.py` | Top-K、三角化/重投影、候选搜索与回退 |
| `phase_1/marker_pipeline/export.py`, `cli.py` | 按 Phase 2 fold 导出、评价与命令入口 |
| `phase_1/tests/`, `phase_1/docs/marker-runbook.md` | 验证与运行说明 |

---

### Task 1: Phase 1 输入契约与合成病例

**Files:** Create `phase_1/marker_pipeline/{__init__,schema,data,synthetic}.py`; review/modify existing `phase_1/tests/test_data_contract.py`。

**Interfaces:** 病例清单采用 JSON：每例 `case_id,group_id,views`；每视角 `view_id,angle_deg,projection_path,marker_xy,P`，其中 `P` 为 `3×4`，可选 `mask_path` 留给 Phase 2。`load_manifest(path) -> list[Case]` 返回经验证的三视角病例；`validate_group_split(cases, fold_spec)` 检查组互斥；`make_synthetic_dataset(out_dir, seed)` 生成至少 6 个独立支架组的小型 TIFF、真值点、已知几何及 Mask，覆盖正常、双点接近和低视差病例，并给出可供折外训练的 group-wise split。

- [ ] 核对现有测试并补足：恰好三个不同视角、同组不跨 split、坐标/P 有限且在界内、图像尺寸匹配；350°/0°/10° 接受，超过 60° 拒绝。
- [ ] 运行 `python -m pytest phase_1/tests/test_data_contract.py -q`，确认缺失接口导致失败。
- [ ] 实现最小 schema/loader/合成生成器；不读取 Mask 作为 Phase 1 网络输入。
- [ ] 复跑上述测试；保存一份供后续任务复用的合成数据格式说明。

### Task 2: 简单 U-Net 热图训练

**Files:** Create `phase_1/marker_pipeline/models/{__init__,unet2d}.py`, `phase_1/marker_pipeline/{heatmap,train}.py`, `phase_1/tests/test_unet_training.py`。

**Interfaces:** `MarkerUNet2D(in_channels=1, widths=(32,64,128,256))` 输出 `[B,1,H,W]` logits；`make_heatmap(marker_xy, image_size, sigma)` 生成两 Gaussian 的逐像素 max；`train_detector(cases, config, out_dir)` 保存权重、配置、训练组清单和 checkpoint；`predict_heatmap(checkpoint, image)` 返回原图大小热图。三视角展平为单图批次，但共享权重。

- [ ] 写测试：非 8 倍数尺寸仍原样输出、双点热图有两个峰、权重更新且 loss 有限、checkpoint 加载后推理一致。
- [ ] 运行 `python -m pytest phase_1/tests/test_unet_training.py -q`，确认失败。
- [ ] 实现普通卷积 U-Net、`sigmoid` 后可配置正样本加权 MSE（首版 `lambda_fn=0`）、全图训练、可复现 seed 与 checkpoint。
- [ ] 复跑测试；用 Task 1 合成病例完成至少一步 CPU 训练。

### Task 3: 峰值候选、三视角搜索与回退

**Files:** Create `phase_1/marker_pipeline/{peaks,geometry,fusion}.py`, `phase_1/tests/test_fusion.py`。

**Interfaces:** `decode_heatmap(heatmap, config) -> list[Candidate]` 返回原图 `(x,y,score)` Top-K；`fuse_three_views(candidate_sets, projection_matrices, config) -> FusionResult` 返回每视角两个无序点、`pair_status`、`fusion_mode`、`association_status` 和诊断。默认 K=10/L=5 可配置；联合枚举每视角无序点对及另外两视角的 4 种相对交换，采用无权 DLT/SVD 与鲁棒重投影评分；低视差/不稳定时按冻结规则单视角回退。

- [ ] 写测试：交换第二/第三视角点顺序不改最终无序点对、错误投影几何有明确诊断、低视差双峰进入回退、单峰不足时 `invalid`、近距离双点不重复用一个候选。
- [ ] 运行 `python -m pytest phase_1/tests/test_fusion.py -q`，确认失败。
- [ ] 实现 NMS、有限值/深度检查、假设评分与确定性回退；阈值只来自配置，不在推理时用真值选择。
- [ ] 复跑测试，并在合成病例上核对三视角几何能找回已知点。

### Task 4: Phase 1 命令入口与按 fold 导出

**Files:** Create `phase_1/marker_pipeline/{export,cli}.py`, `phase_1/tests/test_oof_export.py`, `phase_1/docs/marker-runbook.md`。

**Interfaces:** `python -m phase_1.marker_pipeline.cli` 提供 `make-synthetic`、`validate`、`train`、`predict`、`evaluate`、`export-fold` 子命令；`evaluate` 对有真值的允许样本报告无序点对定位误差、`PairAcc@τ`、三视角全成功率和回退/失败率。`export_phase2_fold(manifest, split, config, out_dir)` 为外层训练组再按 `group_id` 留一组/内层 K 折训练检测器产生 `group_oof`；外层验证/测试由仅在允许训练组拟合的冻结检测器产生 `held_out`。输出每例三视角 v1 JSON、包级 `phase2_fold_id`、来源登记表及 `case_id/view_id` 对照；`model_id` 绑定 checkpoint 与解码/几何配置摘要。

- [ ] 写测试：被预测组不在对应模型拟合组、验证/测试组未参与拟合或调参、三视角在同一包、`q_pair=null` 有效记录及 `invalid q_pair=0` 通过契约校验；无序点交换不改变 `evaluate` 的误差与 `PairAcc@τ`。
- [ ] 运行 `python -m pytest phase_1/tests/test_oof_export.py -q`，确认失败。
- [ ] 实现导出、保守 `error_radius_px` 规则和可追溯来源；误差半径只用允许训练组的保留病例估计，样本不足使用配置中预先声明的保守半径，不虚构校准概率。
- [ ] 复跑测试；用合成组完成 `train → predict → evaluate → export-fold` 并记录实际命令及产物路径。

## Completion Check

- 训练、推理、评价和折外导出命令能生成符合 v1 契约的三视角预测包及来源登记。
- 合成病例覆盖正常、双点接近、低视差回退与候选不足，关键测试通过。
- 真实投影和几何接入后再验证定位性能；消融不属于首版完成条件。
