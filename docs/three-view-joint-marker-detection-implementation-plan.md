# Phase 1 三视角联合 Marker 检测 Implementation Plan

实施记录（2026-11-05）：软件链路已完成，包含固定锥束极线支持、病例级联合训练、旧模型兼容、Phase 2 OOF 包、候选诊断及无几何搜索消融。相关自动测试 53 项通过；解析锥束 Gaussian 样例仅用于软件验收。真实扫描器标定、可信 DRR/真实数据和正式折外性能比较仍需数据资产。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 Phase 1 manifest、CLI 和 Phase 2 二维点对交接格式的前提下，将固定 30° 间隔锥束投影的三个视角作为一个样本联合训练和预测。

**Architecture:** 复用单图 U-Net 产生三个初始热图；固定锥束极线采样产生跨视角软支持图，可学习的质量权重和小型残差融合头产生三个最终热图；沿用既有候选解码和几何搜索得到每视角无序二维点对。旧 checkpoint 与旧路径保留作基线。

**Tech Stack:** Python 3.10+、PyTorch、NumPy、SciPy、tifffile、pytest；不新增必需依赖。

**Spec:** [固定间隔锥束投影三视角联合检测 spec](three-view-joint-marker-detection-spec.md)。同时阅读[现有 Phase 1 spec](marker-detection-codegen-spec.md)和[Phase 1 → Phase 2 交接契约](../../docs/phase1-phase2-handoff-design.md)。

## Global Constraints

- 外部仍使用每病例恰好三个 `views` 的 manifest 及 `train/predict/evaluate/export-fold` 原有位置参数；不新增逐病例位置输入字段，不读取 `mask_path` 作为 Phase 1 模型输入。
- 联合模型同一训练/预测包内的图像使用相同探测器像素网格 `[H,W]`；若曾重采样，`P` 必须对应当前网格。当前仓库没有真实锥束病例或独立标定资产，软件测试使用解析锥束 fixture；真实几何冻结和性能结论待外部数据到位。
- 联合模型只接受角度 `[n·30°, (n+1)·30°, (n+2)·30°] mod 360`，容差 `1e-3°`；旧单图模型仍接受其原有 `≤60°` 数据。
- 联合模型图像输入 `[B,3,1,H,W]`，初始和最终输出均 `[B,3,1,H,W]`；每视角两个无序浮点坐标，原图 `(x=列,y=行)`。
- 极线支持在 `1/4` 分辨率、每线 32 个有效点上计算，分块采样；固定几何校验容差的工程初值为探测器四角线位移 `0.25 px`，正式阈值在训练组内验证并冻结。
- 联合模型每 batch 按病例计数，默认 `batch_size=1`；初始热图辅助损失权重默认 `aux_loss_weight=0.25`。旧模型的 `batch_size` 语义保持不变。
- 按原始 3D 支架 `group_id` 切分；Phase 2 fold 的 group OOF 和 held-out 来源规则、`q_pair=null` 未校准约定不变。当前 Gaussian 圆点合成数据仅用于软件 smoke。
- 新训练需显式配置 `model_type=joint_epipolar_v1`；缺省 `model_type=single_view_v1` 保持旧训练行为，旧 checkpoint 继续可推理。新 checkpoint 格式为 `marker_joint_epipolar_checkpoint_v1`。

## Review Focus

以下五类条件必须在对应任务的测试中覆盖。

1. `330°/0°/30°` 与乱序 manifest：联合模型规范化为正确循环顺序，输出仍映射回原 `view_id`（Task 1、5）。
2. 一视角极弱或另一个视角有强假峰：融合不能依赖目标视角硬 Top-K，三图监督和质量权重应允许修复并防止坏图污染（Task 3、4）。
3. 两个无序 Marker 靠近或交换：不得重复使用一个点；评价必须选择整对一对一匹配（Task 5、6）。
4. 固定扫描器之外的 `P`、越界极线和非 4 整除图像：明确错误或有效掩码处理，禁止越界假响应（Task 2、3）。
5. 同支架不同 `n`、不同 fold：不得拆到 train/test 两侧，OOF 模型不得见到被预测组（Task 4、5）。

## File map

| 文件 | 职责 |
|---|---|
| `phase_1/marker_pipeline/view_order.py` | 联合模型的角度规范化与原 `view_id` 回填映射 |
| `phase_1/marker_pipeline/epipolar.py` | 由锥束 `P` 构建/验证固定相对极线关系，分块生成软支持图 |
| `phase_1/marker_pipeline/models/joint_epipolar.py` | 共享 U-Net、来源权重及残差融合头 |
| `phase_1/marker_pipeline/train.py` | 按病例组批、联合损失、新 checkpoint、按格式分派推理 |
| `phase_1/marker_pipeline/export.py` | 每病例一次联合推理，保留 v1 记录和 OOF 来源；修正评价匹配 |
| `phase_1/marker_pipeline/cli.py` | 保留位置参数，根据配置/ checkpoint 格式分派 |
| `phase_1/tests/test_joint_view_order.py` | 固定间隔与跨 0° 排序 |
| `phase_1/tests/test_epipolar_support.py` | 锥束极线、掩码、几何一致性及尺寸 |
| `phase_1/tests/test_joint_model.py` | 三图依赖、梯度、权重和双峰输出 |
| `phase_1/tests/test_joint_pipeline.py` | 训练、CLI、预测包、折外来源及评价回归 |
| `phase_1/docs/marker-runbook.md` | 新模型配置与实际数据验收命令 |

---

### Task 1: 三视角角度规范化

**Files:** Create `phase_1/marker_pipeline/view_order.py`; test `phase_1/tests/test_joint_view_order.py`. Keep `data.py` manifest schema unchanged.

**Interfaces:** `canonical_triplet_indices(angles_deg: Sequence[float], *, tolerance_deg: float = 1e-3) -> tuple[int, int, int]` 返回原 view 数组下标，按 `n·30° → (n+1)·30° → (n+2)·30°` 排列；调用方同时保存该下标到 `view_id` 的反向映射。

- [ ] 写失败测试：`[30,330,0]` 返回 `(1,2,0)`；`[60,0,30]` 返回 `(1,2,0)`；重复角度、`[0,20,60]`、非 30° 网格和非有限角度抛 `ValueError`。
- [ ] 运行 `python -m pytest phase_1/tests/test_joint_view_order.py -q`，确认缺少函数而失败。
- [ ] 实现 `canonical_triplet_indices`：用模 360° 的循环间隔找唯一的 300° 大间隔，从其后起步；验证两段 30° 间隔与起始角为 30° 网格点。不要改变旧 `load_manifest` 的 `≤60°` 规则。
- [ ] 复跑该测试，期望全部通过；提交本任务文件。

### Task 2: 固定锥束极线支持

**Files:** Create `phase_1/marker_pipeline/epipolar.py`; test `phase_1/tests/test_epipolar_support.py`.

**Interfaces:**

- `fundamentals_from_projections(Ps: Sequence[np.ndarray]) -> np.ndarray` 返回 `[3,3,3,3]`，`F[target,source]` 把目标像素映到来源图极线；要求各 `P` 为有限、秩 3、有限锥束源点。
- `validate_relative_geometry(reference_F: np.ndarray, case_Ps: Sequence[np.ndarray], image_shape: tuple[int,int], *, tolerance_px: float = 0.25) -> None`：对 3×3 目标探测器网格及来源探测器四角，比较归一化极线的最大像素位移；超阈值抛含 `case_id` 可由调用方补充的错误。
- `EpipolarSupport(nn.Module)`：构造参数为 `reference_F, stride=4, samples=32, chunk_rows=32, temperature=0.1`；`forward(probabilities: Tensor[B,3,1,H,W]) -> Tensor[B,3,2,H,W]`。来源次序按规范化视角下标升序，跳过目标自身。

- [ ] 写失败测试：用解析锥束 fixture 生成已知 3D 点和三组 `P`，投影点落在对应极线上；`n` 整体旋转 30° 的相对几何通过校验，探测器偏移的 `P` 被拒绝；极线出界不产生响应；`H,W` 非 4 整除仍输出原尺寸。
- [ ] 运行 `python -m pytest phase_1/tests/test_epipolar_support.py -q`，确认失败。
- [ ] 实现 `F` 计算、尺度/符号归一化和物理点校验；按有效极线段采样，1/4 局部最大池化后用归一化 `logsumexp` 聚合，最后上采样；`grid_sample` 按 `chunk_rows` 分块并屏蔽无效位置。
- [ ] 复跑测试，期望全部通过；用 1024×1024、batch 1 的张量做一次前向/反向，确认无非有限值和没有一次性展开全图极线张量；提交本任务文件。

### Task 3: 可训练三视角检测网络

**Files:** Create `phase_1/marker_pipeline/models/joint_epipolar.py`; test `phase_1/tests/test_joint_model.py`. Reuse `models/unet2d.py`，不改变旧 `MarkerUNet2D.forward` 契约。

**Interfaces:** `JointEpipolarMarkerNet(backbone: MarkerUNet2D, support: EpipolarSupport)`；`forward(images: Tensor[B,3,1,H,W]) -> tuple[Tensor,Tensor]` 返回 `(initial_logits, refined_logits)`，形状均为 `[B,3,1,H,W]`。内部允许把 `[B,3]` 展平给共享 backbone，但融合前恢复病例和视角维度。

- [ ] 写失败测试：三个输出尺寸正确；置换来源图像会改变目标最终热图但不改变其初始热图；目标视角最终损失的梯度能到达另外两张输入图；坏视角质量权重有限且归一化；两个无序 marker 仍使用单通道热图。
- [ ] 运行 `python -m pytest phase_1/tests/test_joint_model.py -q`，确认失败。
- [ ] 实现每视角初始热图、来源质量权重与 `[初始 logits, 两张加权支持图]` 的小型卷积残差融合头；末层零初始化。测试跨图依赖时先把该末层设为非零或完成一次优化步，避免零初始化使断言失效。
- [ ] 复跑测试，期望全部通过；提交本任务文件。

### Task 4: 病例级训练与 checkpoint

**Files:** Modify `phase_1/marker_pipeline/train.py` and `phase_1/marker_pipeline/cli.py`; test `phase_1/tests/test_joint_pipeline.py`.

**Interfaces:** `train_joint_detector(cases: Sequence[Case], config: dict, out_dir: str | Path) -> dict`；`predict_joint_heatmaps(checkpoint: str | Path, case: Case) -> tuple[np.ndarray,np.ndarray]`，每个数组形状为 `[3,H,W]`、按原 case 的 view 顺序返回。`train_detector` 根据 `model_type` 分派，缺省继续走旧单图路径。

- [ ] 写失败测试：`model_type=joint_epipolar_v1` 训练一个含 `0/30/60` 与 `330/0/30` 的小病例集，产生新格式 checkpoint；一次训练步有限且更新融合头；checkpoint 恢复的热图一致；`model_type` 缺省及旧 checkpoint 仍走旧路径；同一支架不同 `n` 不被拆成视角样本。
- [ ] 运行 `python -m pytest phase_1/tests/test_joint_pipeline.py -q`，确认失败。
- [ ] 在新路径按病例规范化顺序组批，以三个最终热图平均的既有加权 MSE 加 `0.25 ×` 初始热图辅助损失训练；从排序后首个训练病例建立参考相对几何，对该训练包的所有病例作固定几何与 `[H,W]` 校验，将参考相对几何、`geometry_id`、权重、模型配置和拟合 `group_id` 写入新 checkpoint。`batch_size` 新路径按病例计；不同图尺寸明确报错，不直接拼接。
- [ ] `predict_joint_heatmaps` 从 checkpoint 恢复固定几何，对当前病例的 `P` 校验，做一次三图联合前向并按原 view 顺序回填；复跑测试，期望全部通过；提交本任务文件。

### Task 5: 沿用 CLI、几何搜索与 Phase 2 预测包

**Files:** Modify `phase_1/marker_pipeline/export.py` and necessary `cli.py` dispatch; extend `phase_1/tests/test_joint_pipeline.py`.

**Interfaces:** `_predict_case(case, checkpoint, config, model_id, source, fold_id) -> dict` 保持签名和 `phase1_marker_case_v1` 输出。新 checkpoint 使用 `predict_joint_heatmaps`，对最终热图逐视角 `decode_heatmap`，再使用相同配置的 `fuse_three_views`；旧 checkpoint 保持现有路径。`export_phase2_fold` 的组外来源规则不变。

- [ ] 写失败测试：乱序输入的三个 `view_id` 正确回填；新旧 checkpoint 均通过原 `predict` CLI；V2 输出仍通过 `validate_prediction_record` 和 Phase 2 `validate_prediction_package`；`model_id` 对新权重/几何配置变化敏感；同一支架组的 OOF 预测模型未见该组。
- [ ] 运行 `python -m pytest phase_1/tests/test_joint_pipeline.py phase_1/tests/test_oof_export.py -q`，确认新断言失败、旧断言仍通过。
- [ ] 修改 `_predict_case` 与 `export_phase2_fold` 的模型分派：新模式一病例一次前向，保存初始/最终候选诊断，最终 `fusion_mode=three_view` 或旧规则的逐视角回退；来源登记增加新架构和固定几何摘要，不改 Phase 2 输入字段。
- [ ] 复跑上述测试并跑 `python -m pytest phase_2/nnUNet/nnunetv2/tests/test_marker_prior_channels.py -q`，期望全部通过；提交本任务文件。

### Task 6: 评价修正、消融和运行手册

**Files:** Modify `phase_1/marker_pipeline/export.py`, `phase_1/docs/marker-runbook.md`; extend `phase_1/tests/test_joint_pipeline.py` and `phase_1/tests/test_oof_export.py`.

**Interfaces:** `_pair_metrics(truth, prediction, tau) -> tuple[float,bool]`：平均点误差采用总误差较小的整对排列；只要存在一种整对排列使两个误差都在 `τ` 内，成功状态为 `True`。`evaluate` 保持 `PairAcc@tau` 键名，但分母改为全部视角、`invalid` 计失败；新增 `conditional_PairAcc@tau` 和有效视角数作诊断。

- [ ] 写失败测试：构造两个预测都靠近同一真值的病例，确认一对一匹配不能得到虚假的高 `PairAcc@τ`；交换两个真值或预测的顺序不改变结果；一个视角为 `invalid` 时主 `PairAcc@tau` 分母仍为三个视角，条件准确率只以两个有效视角为分母。另比较相同候选几何配置下 V1 与 V2，并检查三视角全成功率的病例级统计。
- [ ] 运行 `python -m pytest phase_1/tests/test_oof_export.py phase_1/tests/test_joint_pipeline.py -q`，确认新断言失败。
- [ ] 修正 `_pair_metrics` 为保持/交换两种**整体排列**择优；在 runbook 增加联合模型配置、旧模型回退、1024² 内存说明及真实锥束数据评估步骤。明确 Gaussian 圆点 synthetic 仅是 smoke，正式结论需真实/可信 DRR 的 group-wise held-out 对照。
- [ ] 运行 `python -m pytest phase_1/tests -q` 和相关 Phase 2 marker prior/数据校验测试，确认全部通过；在同一组划分上分别运行 V1/V2 的 `export-fold → evaluate`，保存结果和每例耗时，若尚无真实数据仅报告软件验证而不声称性能提升；提交本任务文件。

## Completion Check

- 一条 `0/30/60`、一条跨 0° 的病例均能从原 manifest 经新 checkpoint 生成 v1 三视角记录，Phase 2 数据校验接受。
- 新模型目标视角最终热图确实受另外两个视角影响；旧模型、旧 checkpoint、旧 CLI 位置参数可用。
- 固定锥束几何在已知 3D 点上通过极线检查；不匹配的几何、越界采样及低质量来源有明确测试。
- group OOF 无支架组泄漏；无序点对指标一对一；真实收益仅由相同 fold 的 V1/V2 比较给出。
- 若真实锥束数据/标定资产尚未提供，完成状态仅指软件链路验收；正式几何冻结和性能结论仍待相应数据。
