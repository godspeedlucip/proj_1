# Execution ledger — plan: docs/superpowers/plans/2026-11-16-dual-baseline-implementation.md

2026-11-16 用户授权制定 Plan 并实施。采用 inline 连续实施、一次独立代码审查。

Ruling: 当前目录没有 Git；直接沿用工作目录，tmp/implementation_audit/baseline 保存基线代码快照，日志与测试代替 commit 记录，不创建 Git 仓库 — 既有目录即可承载授权修改 — 代价是回退需依靠快照，不能使用 Git 提交历史。

Ruling: 本机 CPU、缺正式数据/标定；实施可完成的软件与 G0，G1/G2 的实际实验保留为未完成 — 没有证据支持科学复现结论 — 代价是不能据本轮结果判断精度或方法优劣。

Ruling: 采用 v2 严格工作流，保留既有 prepared-surrogate 接口；legacy 不升级为原论文复现 — 旧数据与旧检查点缺乏完整几何契约 — 代价是两个入口并存，旧检查点仍需显式迁移才可用于 v2 推理。

Pre-flight: Task 1 resolved config 被 3/5/6/7/8 消费；Task 2 物理网格与指标被 3/8 消费；Task 4 扩展 train_prepared 由 5 调用且保留旧默认；Task 5 workflow 供 8 运行。无已知接口冲突。

## 任务记录

- Task 1: complete — RED missing module → GREEN 14 configuration/model/profile tests; workflow examples added
- Task 2: complete — RED missing module → GREEN 4 metrics/physical grid/failure tests
- Task 3: complete — RED missing module → GREEN 12 inference/geometry tests
- Task 4: complete — RED missing module → GREEN 35 sampling/training/preprocessing tests; exact resume equality
- Task 5: complete — RED missing module → GREEN workflow, content/sidecar fingerprints; resume horizon regression RED→GREEN
- Task 6: complete — RED missing module → GREEN 5 procedural geometry and dataset tests
- Task 7: complete — RED missing module → GREEN 3 analytic/resource tests
- Task 8: complete — RED missing module → GREEN 3 matrix and complete synthetic chain tests; final artifacts regenerated after review fixes
- Task 9: complete — independent review, Important findings fixed with regression tests, 110-test suite green, fresh G0 and five-arm matrix executed, CLI/source artifacts checked; see docs/实施验收记录.md

Task 5 follow-up: resume failed because resolved_config epoch horizon participated in immutable contract. Only current-stage epochs may extend; data, sampler and other settings remain immutable. Regression verified.

G0 artifact run: python -m stent_repro.workflow smoke --output outputs/g0_validation → completed, A/B each 1 held-out shared case, zero pipeline failures. Compact models, 2 epochs; Dice approximately 0.070/0.089 only demonstrates the software chain and is not a scientific accuracy result.

Task 9: fresh reviewer dispatched; full regression suite and CLI checks running. Artifacts are retained because no Git history exists.

## 最终审查与验收

Reviewer: /root/implementation_review，独立上下文；5 Important、2 Minor、0 Critical。Minor 中 DTE 缓存键按影响升级为 Important：可编辑任务名可能令不同 DTE 配置错误共享权重。该项纳入同一次修复。

Final: fixed 采集几何冲突 — test_inference_rejects_acquisition_metadata_conflict RED→GREEN；另验证 prepare 在输出前拒绝冲突；suite 110/110。

Final: fixed 未冻结测试输入 — test_frozen_inference_contract_detects_replaced_projection RED→GREEN；矩阵逐任务复核投影和真值，checkpoint 指纹保存；suite 110/110。

Final: fixed 损坏数组导致批次中断 — test_corrupt_array_keeps_evaluation_denominator 两种文件 RED→GREEN；另验证坏投影转为推理失败；suite 110/110。

Final: fixed legacy resume 与 best 退化 — test_legacy_contract_migration_preserves_best_checkpoint RED→GREEN；suite 110/110。

Final: fixed 正式训练资源前提 — test_formal_training_requires_resource_evidence_before_read_or_allocate RED→GREEN；阶段预检共享的回归先失败再修复，并验证 batch/stage 改动被拒绝；suite 110/110。

Final: fixed DTE 复用身份 — test_cache_identity_changes_with_dte_training_or_source RED→GREEN；五个任务实际执行，核验每法内 DTE 哈希一致；suite 110/110。

Final: minor (deferred): A reference 逐例随机起始角尚未实现；配置级固定偏移可用，固定课题角度不受影响。已在工作流、验收记录、进度文件明示为软件缺口。

Final: Ruling: reviewer 未判断作者网络完全等价、论文数值/实测精度与 GPU 可训练性；不据本轮结果认证这些内容 — 缺原始数据/权重/作者代码与目标 GPU 实测 — 代价是 G1/G2 保持未完成，原尺寸可行性仍不确定。

Final: Ruling: 保留明确标注的单色模拟、中心线占据真值、B 错误增强替代及整幅/整体积推理 — 它们足以检验 G0 链路，Spec 允许先实现整体积路径 — 代价是存在模拟域差异且大输入受内存限制，不能认为已覆盖原论文全部物理与指标。

Final verification: python -m pytest -q → 110 passed, 1 dependency deprecation warning, 12.45 s；日志 tmp/implementation_audit/final_tests.log。

Final G0: python -m stent_repro.workflow smoke --output outputs/g0_validation_final → exit 0；A/B 各一例，0 执行失败；Dice 0.0700389105 / 0.0888324873，仅工程证据。

Final matrix: experiment_matrix + run_matrix(execute=True)，seed=17，A C=1/3/5、B 3/1 通道 → 5/5 completed，0 评价流程失败。保留各自 DTR best/last 与合法复用的 DTE。

Final artifacts: verify_final_artifacts.py cli_checks_final → 7 配置和六类 CLI，11 份训练来源记录及当前源码哈希核验。首次审核脚本将模拟 provenance 当成训练 provenance，修正类型筛选后完整重跑；无训练源码改动。

Finishing: 已应用 finishing-a-development-branch 的验证要求；本目录无 Git，分支合并/发布菜单不适用。保留用户目录、基线快照和所有证据，不删除审计记录。
