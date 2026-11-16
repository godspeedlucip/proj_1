# 双基线优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**实施状态（2026-11-16）：** 九项工程任务与 G0 已完成；G1/G2 未完成。修复后证据以 `outputs/g0_validation_final/` 为准；详见 [实施验收记录](../../实施验收记录.md)。

**Goal:** 完成双基线可执行配置、完整推理与评价、在线采样与恢复、可追溯模拟和实验运行工具，并在当前 CPU 环境完成 G0 工程验证。

**Architecture:** 沿用现有网络、几何与 I/O；新增严格的 v2 运行配置和 workflow 入口，将模拟、训练、推理与评价串联。原有 prepared-surrogate 入口保留兼容；正式 G1/G2 实验由数据和资源预检约束，所有未完成条件留在验收报告中。

**Tech Stack:** Python、PyTorch、NumPy、SciPy、PyYAML、tifffile、scikit-image（本机 0.26.0，用 Lee skeletonize）。

**Spec:** [双基线复现优化与静态三投影对比](../specs/2026-11-16-dual-baseline-fidelity-design.md)

## Global Constraints

- 用户已授权“制定 Plan 并实施”，采用当前会话连续执行；不重复请求计划批准。
- 当前目录无 Git；在现有目录实施，保存源码基线快照和实施日志，不创建虚假的分支/提交记录。
- A 保留 DTE→单体 FDK→2.5D DTR；B 保留 DTE→逐视角三通道反投影→3D DTR。
- 默认课题角度 `[0,30,60]°`；原文 A 四幅相邻 45°，B 三幅相邻 60°；单通道及 C 消融独立训练。
- A 深度未明，legacy 与四次下采样候选分别命名。B 的输出激活/BN/Dice 属于明确移植选择。
- 使用真实物理坐标，体数组 z/y/x，spacing/origin x/y/z，单位 mm。拒绝未支持的旋转网格，不静默忽略 direction。
- 所有正式输入数据、标定和权重均有内容哈希；测试真值只进入评价，不进入推理。
- 不能以 G0、测试通过、缩配网络或理想投影诊断冒充论文精度复现。
- 真实患者投影、作者模型/能谱、正式测试集与 CUDA 尚缺；不能完成的 G1/G2 实验须明确标记，不用占位分数替代。

## Review Focus

- 同形状但 origin/spacing 不同的体：必须按物理坐标重采样，并检查视野覆盖（Task 2/3）。
- 空预测或流程失败：保持全部病例分母，结构距离缺失有原因，不能伪造零距离（Task 2/8）。
- 清单不变但数据内容被替换：恢复或推理前必须检测哈希变化（Task 5）。
- 16,000 无法整除 batch 24：每轮样本数量准确，尾批不丢弃；恢复后序列相同（Task 4）。
- 带器械实测图或测试真值被误用：不再次合成器械，推理清单禁止真值字段（Task 3/5）。

## Task 1 配置与网络结构

**Files:** Create `src/stent_repro/configuration.py`, `src/stent_repro/model_registry.py`, `tests/test_configuration.py`; modify `networks.py` and method models as needed; add `configs/workflows/*.yaml`.

**Interfaces:** `resolve_config(source: str|Path|dict) -> dict`; `build_model(config: dict, stage: str) -> nn.Module`; `architecture_report(config: dict, stage: str) -> dict` (meta-safe). Config sections are schema_version/method/profile/seed/geometry/network/training/preprocessing/inference/evaluation/provenance.

- [x] Write tests rejecting unknown nested keys, nonfinite values, incompatible profile/angles/channels, and unmarked reduced channels; require network report counts and stage shapes.
- [x] Run `python -m pytest tests/test_configuration.py -q`; expect missing module/behavior failure.
- [x] Implement strict validation, explicit defaults and architecture factory. Preserve legacy classes' default behavior. Add named A-depth candidate and explicitly labeled compact G0 architecture.
- [x] Re-run new tests and existing model/config tests; all must pass.

## Task 2 独立评价与物理网格

**Files:** Create `src/stent_repro/evaluation.py`, `tests/test_evaluation.py`; add scikit-image dependency.

**Interfaces:** `validate_grid(grid: dict) -> dict`; `resample_volume(array, source_grid, target_grid, order=1) -> np.ndarray`; `segmentation_metrics(pred, truth, grid) -> dict`; `projection_metrics(pred, truth) -> dict`; `summarize_cases(rows) -> dict`; `evaluate_manifest(path, output) -> dict`.

- [x] Test identical/nonoverlap/empty masks, isolated voxel shifted 0.6 mm, different origin resampling, unsupported direction, failed-case denominator and JSON-safe nulls.
- [x] Run evaluation tests and observe failure.
- [x] Implement macro metrics, Lee skeleton and cKDTree directional distances, complete-coverage check, one-time probability resampling then thresholding, CSV/JSON reports. MAPE/tolerance variants must use separate names unless source semantics are confirmed.
- [x] Run `python -m pytest tests/test_evaluation.py -q`; expect all pass.

## Task 3 完整推理

**Files:** Create `src/stent_repro/inference.py`, `tests/test_inference.py`.

**Interfaces:** `reconstruct(projections, config, dte, dtr, *, normalization=None, device='cpu', diagnostic_ideal=False) -> dict`; `load_trained_model(checkpoint, config, stage) -> (model, normalization)`; `run_inference(manifest_path, config, checkpoints, output, device='cpu') -> dict`.

- [x] Test DTE padding/unpadding, A center-slice order and boundary replication, B independent channels and single-channel sum, nonfinite outputs, invalid checkpoint stage/config and forbidden truth field.
- [x] Run inference tests; expect failure.
- [x] Implement no-truth complete pipeline with whole-image/whole-volume inference and explicit memory limits. Do not add unverified automatic tiling; fail with actionable resource message. Save intermediate volumes, probability, binary, geometry, timings and checkpoint hashes.
- [x] Run inference and geometry tests; expect pass.

## Task 4 在线采样与训练协议

**Files:** Create `src/stent_repro/training/sampling.py`, `src/stent_repro/training/online.py`, `tests/test_sampling.py`; modify `training/data.py` and `training/engine.py`.

**Interfaces:** `RunningStatistics.update(tensor)`, `.state_dict()`, `.normalize(tensor)`; `EpochDataset(dataset, count, seed, epoch, stream)` reproducible sampling wrapper; `OnlineDTEDataset(rows, split, params, seed)` loads independent source pools; extend `train_prepared(..., train_pairs=None, validation_pairs=None, normalization_mode='frozen', selection_dataset=None, selection_metric='loss')`.

- [x] Test exact sample count with remainder, distinct repeatable epoch samples, independent pool cropping/placement, data split isolation, A online count excluding validation, DTR synchronized random crops, and uninterrupted/resumed equality.
- [x] Run new tests; expect failure.
- [x] Implement deterministic per-sample RNG, running statistics, explicit pair count, fixed validation panel and checkpoint metric selection. Preserve old defaults for legacy smoke tests. Save Python/NumPy/Torch, sampler and normalization state.
- [x] Run sampling and existing training/preprocessing tests; expect pass.

## Task 5 来源核验与统一 workflow

**Files:** Create `src/stent_repro/provenance.py`, `src/stent_repro/workflow.py`, `tests/test_workflow.py`; minimally extend preprocessing source export.

**Interfaces:** `fingerprint_files(paths) -> dict`; `verify_fingerprints(expected) -> None`; `train_workflow(config, manifest, output, stage, device, resume=None) -> dict`; `prepare_workflow(config, manifest, output) -> dict`; CLI `python -m stent_repro.workflow {validate,prepare,train,infer,evaluate,preflight,simulate,matrix,smoke}`.

- [x] Test changed referenced data rejected on resume; resolved config controls geometry/patch/count; all output run files exist; measured inputs unchanged; incompatible metadata fail before training.
- [x] Run workflow tests; expect failure.
- [x] Implement source fingerprints including sidecar metadata, compatibility checks and provenance output. Strict v2 path requires exact grid/units, recorded deviations and fully resolved configuration. Legacy path remains explicitly surrogate.
- [x] Run workflow and regression tests; expect pass.

## Task 6 可追溯器械与模拟数据生成

**Files:** Create `src/stent_repro/simulation.py`, `tests/test_simulation.py`.

**Interfaces:** `generate_scene(config, seed, kind) -> dict` with mask/mesh/parameters/grid; `write_simulated_dataset(config, output, counts) -> Path` generates split-independent cases and raw/projection-pool manifests.

- [x] Test seed reproducibility, nonempty asymmetric stent/wire/coil geometry, units, per-split entity IDs, mesh export and forward-projection source labels.
- [x] Run simulation tests; expect failure.
- [x] Implement parameterized strut rings and connections, curved guidewire/coil, swept-tube mesh and documented analytic voxelization. Use explicit synthetic background, monoenergetic surrogate and recorded noise. No claim of original eight stent designs or patient data. Large datasets generated incrementally.
- [x] Run simulation tests and small data generation; expect pass with valid manifests.

## Task 7 资源与几何预检

**Files:** Create `src/stent_repro/preflight.py`, `tests/test_preflight.py`.

**Interfaces:** `preflight(config, *, stage=None, device='cpu', measure=False) -> dict`; `geometry_validation(config) -> dict`. Full memory measurement must use actual forward/backward/optimizer step on exact configured shape/batch when explicitly selected.

- [x] Test CPU availability separated from paper readiness, unresolved A/source flags, full-shape validation and nonzero gradient/optimizer allocation, point/line-scale analytic checks.
- [x] Run preflight tests; expect failure.
- [x] Implement evidence report, memory estimate explicitly separate from measurement, production prerequisites and bounded analytical phantom checks. CUDA unavailable yields concrete limitation, not a falsely passed paper gate.
- [x] Run preflight tests; expect pass.

## Task 8 实验矩阵与 G0 验收

**Files:** Create `src/stent_repro/experiments.py`, `tests/test_experiments.py`, `tests/test_end_to_end.py`; add adaptation single-channel config; generate `outputs/g0_validation/` evidence.

**Interfaces:** `experiment_matrix(config_a, config_b, seeds) -> list[dict]`; `run_matrix(matrix, manifests, output, execute=False) -> dict`; `smoke(output, seed=42) -> dict`.

- [x] Test unique seed/run identity, independent A C=1/3/5 and B channel checkpoints, common test signatures, failure records and refusal to overwrite an existing run.
- [x] Run experiment tests; expect failure.
- [x] Implement reviewable job manifests and sequential execution using workflow functions; actual G1 requires calibrated data. G0 generates disjoint synthetic train/val/test, trains both stages of explicit compact A/B networks, infers from held-out projections, evaluates with truth only in evaluation, saves diagnostic outputs and honest completion report.
- [x] Run experiments/end-to-end tests, then `python -m stent_repro.workflow smoke --output outputs/g0_validation`; expect finite artifact outputs and no claim of useful learned accuracy.

## Task 9 集成审查与文档

**Files:** Update `复现进度.md`, `docs/训练流程.md`; create `docs/双基线工作流.md` and implementation ledger/result report.

- [x] Run full suite and relevant CLI commands. Check file links/config loading and output hashes. Record exact command and result.
- [x] Dispatch one fresh-context reviewer for new and modified code, this plan, Spec, Review Focus and execution rulings; use baseline snapshot because no Git history exists.
- [x] Fix important findings with regression tests and re-run affected/full checks as justified. Record remaining limitations and any unimplemented Spec requirements precisely.
- [x] Update plan checkboxes from evidence, keep G1/G2 uncompleted if data/GPU remain missing, and deliver Plan, workflow instructions and G0 result links.

## Spec 覆盖与边界

Task 1/5 cover §4–5; Task 4/6 cover §6; Task 3/7 cover §7; Task 2/8 cover §8–9; Task 5/9 cover §10–13. Exact author code/BN semantics, original datasets/energy spectra, A tolerance morphology formula and authentic large-scale experiments cannot be invented. They remain evidence/data-dependent prerequisites. Tiling and cross-framework equivalence probes are optional branches, not silently approximated; whole-volume inference is implemented first.

Plan 初次自检：接口以 config dict + existing Grid 为边界；训练 legacy defaults 与 v2 explicit overrides 分离；非 Git 环境使用源码快照和日志替代提交；无需用户再次批准已授权实施。
