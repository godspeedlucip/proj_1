# 双基线支架三维重建对比实验

实现两条独立训练、推理与评价链路：

- **A / Eulig 2021**：DTE → 单体 FDK → 2.5D DTR，相邻切片通道 C=1/3/5。
- **B / Vöth 2022**：DTE → 三个逐视角反投影通道 → 3D DTR，包含独立训练的单通道消融。

代码作为 `godspeedlucip/proj_1` 仓库的 `contrast_experiments/` 子项目维护，目标分支为 `constrast`。以下命令均在本子项目目录执行。

## 安装与快速验证

Python ≥3.10。需要 CUDA 时先按本机环境安装合适的 PyTorch；默认安装命令不保证得到 CUDA 版本。

```powershell
cd contrast_experiments
python -m pip install -e ".[test,nerve]"
python -m pytest -q -p no:asyncio
python -m stent_repro.workflow --help
python -m stent_repro.workflow validate --config configs/workflows/g0_a.yaml
python -m stent_repro.workflow smoke --output outputs/smoke_example --seed 17
```

输出目录必须是新目录。`smoke` 使用小型合成数据和 compact 网络，仅验证软件链路。`nerve` 依赖只在导入 NIfTI 三维真值时需要；TensorFlow 是可选跨框架探针依赖，不是训练主流程依赖。

## 配置与入口

| 目录/入口 | 用途 |
| --- | --- |
| `src/stent_repro/workflow.py` | v2 prepare/train/infer/evaluate、预检、矩阵及数据导入 CLI |
| `configs/workflows/task_a.yaml`、`task_b.yaml` | 共同 0°/30°/60° 任务配置 |
| `configs/workflows/reference_a.yaml`、`reference_b.yaml` | 论文参考条件；正式数据/几何等字段需要补齐 |
| `configs/workflows/reference_b_cuda.yaml` | B 原 batch/patch 的 FP16 AMP＋激活重计算运行配置 |
| `configs/workflows/a-reference-*.yaml` | A 投影数和角距消融，各自独立准备数据、训练 |
| `configs/workflows/g0_*.yaml` | 小规模工程验证配置 |
| `src/stent_repro/nerve_adapter.py` | 头模与支架投影叠加数据导入，保留原始连续工具投影与三维真值 |

正式训练需要明确的数据版本、标定和匹配的完整训练步预检；示例配置不能直接代表真实设备。CPU 参考几何算子和 CUDA 网络可以组合使用。

## 文档与验证范围

- [双基线工作流](docs/双基线工作流.md)
- [补齐与运行验收](docs/补齐与运行验收.md)
- [优化 Spec](docs/superpowers/specs/2026-11-16-dual-baseline-fidelity-design.md)
- [完成 Plan](docs/superpowers/plans/2026-11-16-completion-plan.md)

2026-11-16 本地运行验收：CPU 217 passed / 2 skipped；CUDA＋TensorFlow 环境 219 passed。A 原 batch/patch 在 FP32、B 原 batch/patch 在 FP16 AMP＋重计算下完成实际前向、反向与 Adam 更新。原生合成数据的三种子五组 compact 短程矩阵共 15 个作业完成，流程失败 0；一次空预测保留在统计分母中。

本仓库只包含代码、配置、测试和文档。不上传头模/投影/三维真值、模型权重、第三方论文 PDF、虚拟环境和运行产物。文档中的 `outputs/`、`tmp/` 链接及本机绝对路径描述原验证环境，需在本地复跑或替换为自己的数据位置，GitHub 中不含这些文件。

本次验收目标是确认正常运行，未执行多周完整训练，不宣称作者数据、结构细节或论文数值结果的完整复现。
