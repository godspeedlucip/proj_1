# Phase 3：FDK U-Net 与 Swin 融合三维分割

本目录于 2026-11-10 从 `codex/fdk-unet-swin-fusion-v3` 的 `b156557`
同步而来，包含该实现的模型、六组核心配置、训练与恢复、可变尺寸 NIfTI
预测、评价、测试和环境说明。新版 FDK U-Net/Swin 链路仅保留在 `phase_3`。
`phase_2` 保留原有 Marker/nnUNet 功能，以及实施前已有的旧版六级
`UNet3D` 和对应两项测试。

后续使用 `phase_3.fdk_unet` 模块入口。从仓库根目录运行：

```powershell
python -m phase_3.fdk_unet.cli --help
python -m pytest -q -p no:cacheprovider phase_3/tests -m "not monai and not prior_artifact and not gpu and not slow"
```

- [安装、数据契约和完整命令](fdk_unet/README.md)
- [原实现的验收记录和已知限制](fdk_unet/VERIFICATION.md)
- [普通 U-Net 配置](fdk_unet/configs/plain_fdk_unet3d_v1.json)
- [Swin 融合配置](fdk_unet/configs/swin_prior_fdk_unet3d_v2.json)

融合环境继续使用仓库内已有的
`tmp/fdk_v3/fusion310/Scripts/python.exe`。官方权重及运行产物保留原位置，
不复制进源码目录。代码支持不同体积尺寸；标签应为经过确认的支架杆件，
`vessel=1` 血管标签不能替代支架标签。正式真实数据实验仍待配对数据和来源资料。

## 数据输入示例

程序直接读取**三投影已经重建得到的单通道三维 FDK 体数据**，每个 case 对应
一个 `.nii` 或 `.nii.gz` 文件。这里不直接输入三张二维投影，也不在训练入口内
执行 FDK 重建。普通 U-Net 和 Swin 融合模型使用同一份图像；程序在内部生成
两条分支各自需要的归一化输入，不需要额外准备第二份 Swin 图像。

### 1. 图像和标签要求

| 内容 | 输入要求 |
|---|---|
| FDK 图像 | 三维单通道，所有体素值有限；可以有负值，不要求事先变成 0/1 或 0–1 |
| 支架标签 | 与图像配对的三维二值体，`0=背景`、`1=支架杆件`；不是 `vessel=1` 血管标签 |
| 图像尺寸 | 不同 case 可以不同，例如 `128×128×128`、`160×128×96`、`97×113×65` |
| 配对网格 | 同一 case 的图像与标签必须具有相同 shape、空间位置和方向（affine） |
| 体素间距 | 当前实现要求三轴均为 `0.2 mm`，容差 `1e-4 mm`；尺寸变化不等于允许任意间距 |
| NIfTI 头信息 | 空间单位必须明确（mm、meter 或 micron）；有效的 qform/sform 不能冲突 |

完整体尺寸不必等于训练 patch，也不必是 32 的倍数。默认训练 patch 为
`128³`：小体会补齐，大体会采样；预测时采用重叠滑窗并恢复原始尺寸和 affine。
程序不会自动重采样到 `0.2 mm`。NIfTI 的强度缩放会在读取时应用，轴顺序也由
程序处理，无需事先转置数据。

你提供的示例实际文件为 `D:/desktop/ICBM_010_sample_00_0-60.nii.gz`
（扩展名为 `.nii.gz`，不是 `.nii.gzz`），已检查为 `128³ / 0.2 mm`。
可将这个绝对路径填入下述 `image` 字段；目前尚无配对支架标签，不能据此开展
监督训练。文件名本身不能证明它的真实三个投影角度或投影来源。

### 2. 推荐目录结构

以下使用四个 case、四个独立来源演示格式，名称、角度和来源信息均为示例，
不是对现有数据的事实认定。使用时替换成真实文件和可核实的来源记录。

```text
data/phase3_demo/
├── manifest.json
├── splits.json
├── predict_manifest.json
├── images/
│   ├── case_001_fdk.nii.gz
│   ├── case_002_fdk.nii.gz
│   ├── case_003_fdk.nii.gz
│   └── case_004_fdk.nii.gz
├── labels/
│   ├── case_001_stent.nii.gz
│   ├── case_002_stent.nii.gz
│   ├── case_003_stent.nii.gz
│   └── case_004_stent.nii.gz
└── provenance/
    └── source_mapping.json
```

直接使用 manifest 时不要求文件带 `_0000`，也不要求使用 `imagesTr/labelsTr`
目录；图像与标签通过清单显式配对。路径相对于 **manifest 所在目录**解析，
也可写 Windows 绝对路径。JSON 中建议使用 `/`，例如 `D:/desktop/example.nii.gz`。

### 3. 训练、验证和测试共用的 manifest.json

本例假定图像由理想掩膜投影重建，因此 `projection_source` 为 `ideal_mask`。
实际数据必须按真实来源选择，不能为了通过检查而照抄这个值。

```json
{
  "schema_version": "fdk_stent_manifest_v1",
  "target_semantics": "stent_struts",
  "group_provenance": {
    "stent_group_001": "provenance/source_mapping.json 中 source_model_001 的来源与独立性记录",
    "stent_group_002": "provenance/source_mapping.json 中 source_model_002 的来源与独立性记录",
    "stent_group_003": "provenance/source_mapping.json 中 source_model_003 的来源与独立性记录",
    "stent_group_004": "provenance/source_mapping.json 中 source_model_004 的来源与独立性记录"
  },
  "upstream_registry": {},
  "cases": [
    {
      "case_id": "case_001",
      "group_id": "stent_group_001",
      "source_model_ids": ["source_model_001"],
      "patient_id": null,
      "image": "images/case_001_fdk.nii.gz",
      "label": "labels/case_001_stent.nii.gz",
      "input_kind": "fdk_3view",
      "angles_deg": [0, 30, 60],
      "projection_source": "ideal_mask",
      "upstream_run_id": null
    },
    {
      "case_id": "case_002",
      "group_id": "stent_group_002",
      "source_model_ids": ["source_model_002"],
      "patient_id": null,
      "image": "images/case_002_fdk.nii.gz",
      "label": "labels/case_002_stent.nii.gz",
      "input_kind": "fdk_3view",
      "angles_deg": [0, 30, 60],
      "projection_source": "ideal_mask",
      "upstream_run_id": null
    },
    {
      "case_id": "case_003",
      "group_id": "stent_group_003",
      "source_model_ids": ["source_model_003"],
      "patient_id": null,
      "image": "images/case_003_fdk.nii.gz",
      "label": "labels/case_003_stent.nii.gz",
      "input_kind": "fdk_3view",
      "angles_deg": [0, 30, 60],
      "projection_source": "ideal_mask",
      "upstream_run_id": null
    },
    {
      "case_id": "case_004",
      "group_id": "stent_group_004",
      "source_model_ids": ["source_model_004"],
      "patient_id": null,
      "image": "images/case_004_fdk.nii.gz",
      "label": "labels/case_004_stent.nii.gz",
      "input_kind": "fdk_3view",
      "angles_deg": [0, 30, 60],
      "projection_source": "ideal_mask",
      "upstream_run_id": null
    }
  ]
}
```

几个容易混淆的字段：

| 字段 | 填写方式 |
|---|---|
| `case_id` | 一个输入体的唯一 ID；以字母或数字开头，仅使用字母、数字、`_`、`.`、`-` |
| `group_id` | 独立支架/患者来源组；同一来源的不同角度组合、重建或 sample 必须归入同组 |
| `source_model_ids` | 该 case 涉及的真实原始模型/实体 ID 列表，至少一个；共享来源不能拆成不同 group |
| `patient_id` | 若存在真实患者来源则填写一致的匿名 ID；纯模拟数据没有患者时填 `null` |
| `angles_deg` | 实际用于重建的三个投影角度，单位为度；不要仅根据文件名猜测 |
| `projection_source` | 仅支持 `measured`（实测投影）、`ideal_mask`（理想掩膜投影）、`predicted_mask`（上游预测掩膜投影） |
| `group_provenance` | 每个 group 的可追溯来源说明或证据路径；程序保存这些声明，不会替你证明其真实性 |

一次正式比较的 manifest 必须使用同一种 `projection_source`。
`group_provenance` 中的说明应指向实际可查证的记录；上面的示例文字不是来源证据。

若使用 `predicted_mask`，每个 case 还需填写对应的 `upstream_run_id`，并在
manifest 顶层 `upstream_registry` 中登记该运行。下面是替换这一字段的示例：

```json
{
  "upstream_registry": {
    "upstream_seg_run_001": {
      "fit_group_ids": ["external_train_group_001"],
      "selection_group_ids": ["external_validation_group_001"],
      "calibration_group_ids": []
    }
  }
}
```

对应 case 设置 `projection_source="predicted_mask"`、
`upstream_run_id="upstream_seg_run_001"`。三个列表要记录上游实际用于拟合、
模型选择和校准的实体，未使用才填空列表；上下游对同一实体使用一致的 group ID。
这些实体不能包含当前下游 fold 的 validation/test 组。

### 4. 按来源分组的 splits.json

划分中填写的是 **group_id，不是 case_id**。每个 fold 必须让所有组恰好出现
一次，且 train、validation、test 三部分都非空。新增同组 sample 时沿用该组划分。
本例只演示一个 fold 的格式，不代表正式研究所需的样本量或交叉验证设计。

```json
{
  "schema_version": "fdk_stent_splits_v1",
  "folds": {
    "0": {
      "train": ["stent_group_001", "stent_group_002"],
      "validation": ["stent_group_003"],
      "test": ["stent_group_004"]
    }
  }
}
```

准备好真实文件并完成来源核实后，先验证，再启动普通 U-Net 基线：

```powershell
$phase3Python = '.\tmp\fdk_v3\fusion310\Scripts\python.exe'
& $phase3Python -m phase_3.fdk_unet.cli validate-data --manifest data/phase3_demo/manifest.json --splits data/phase3_demo/splits.json
& $phase3Python -m phase_3.fdk_unet.cli train --config phase_3/fdk_unet/configs/plain_fdk_unet3d_v1.json --manifest data/phase3_demo/manifest.json --splits data/phase3_demo/splits.json --fold 0 --seed 2026 --device cuda --output runs/fdk_unet_v3/A/fold_0_seed_2026
```

训练命令采用正式默认预算 `10,000` 次更新、`128³` patch、batch 2，输出目录必须
尚不存在。融合模型仍使用这两个数据 JSON；需额外按[完整运行说明](fdk_unet/README.md)
准备训练组统计量和 prior 产物。只有一张未标注图像时，使用下面的预测方式。

### 5. 只有图像时的 predict_manifest.json

预测可省略 `label`。以下复用上例第四个 case 演示无标签清单；对你提供的样例，
将 `image` 换成实际绝对路径，并将 case/source/group/角度/来源改成该样例的真实信息。

```json
{
  "schema_version": "fdk_stent_manifest_v1",
  "target_semantics": "stent_struts",
  "upstream_registry": {},
  "cases": [
    {
      "case_id": "case_004",
      "group_id": "stent_group_004",
      "source_model_ids": ["source_model_004"],
      "patient_id": null,
      "image": "images/case_004_fdk.nii.gz",
      "input_kind": "fdk_3view",
      "angles_deg": [0, 30, 60],
      "projection_source": "ideal_mask",
      "upstream_run_id": null
    }
  ]
}
```

已有训练完成的 checkpoint 后，从仓库根目录运行；这里不传 `--splits`：

```powershell
& $phase3Python -m phase_3.fdk_unet.cli predict --checkpoint runs/fdk_unet_v3/A/fold_0_seed_2026/best.pt --manifest data/phase3_demo/predict_manifest.json --device cuda --output runs/fdk_unet_v3/predict_example
```

输出包含 `case_004_prob.nii.gz`（前景概率）、`case_004_mask.nii.gz`
（阈值 0.5 的二值结果）、该 case 的 telemetry 和 `prediction_run.json`。
此方式属于外部预测，不要求标签，也不能直接作为正式独立测试集评价。
正式测试应使用训练时的 manifest/splits、相同 fold 和 `--split test`；计算指标还需
配对真实标签。只有图像或未训练的模型都不足以给出分割精度结论。

同步结果保存在原工作分支 `codex/phase1-joint-marker`，融合开发分支在
同步验证、提交及合入后删除；实现历史随合入保留。

## 首次同步验证（清理重复代码之前）

- `phase_1/tests`、`phase_2/tests`、`phase_3/tests` 联合基础回归：
  **198 passed，34 deselected**（排除 MONAI、真实权重、GPU、slow 标记）。
- `phase_3/tests` 在固定融合环境下的 MONAI/SSL 集成：
  **6 passed，89 deselected**，包含真实权重与离线恢复。
- 新 CLI 帮助入口通过；新目录 Python 源码无 `phase_2` 模块依赖。
- 54 个原文件的 SHA-256 全部未变；复制后的代码、配置和测试与原文件
  在模块路径替换后逐文件一致。新测试使用独立包名，能与保留的旧测试同时运行。

本次只调整目录与模块引用，没有改变模型计算；原 GPU/slow 验收见历史记录，
本次没有重复执行这些设备和数值实验。测试输出保存在
`tmp/fdk_v3/phase3-sync/`，不纳入源码提交。

## Phase 2 重复代码清理

随后按用户要求删除 `phase_2` 中本次新增的 51 个实现、配置、环境说明、
测试及测试辅助文件；`phase_2/fdk_unet/__init__.py` 只导出旧版 `UNet3D`。
实施前已有的 `model.py`、`test_fdk_unet_model.py` 和其他 Phase 2 文件保留。
原计划与规范文档作为历史记录保留，新版运行命令以本目录说明为准。

清理后验证：

- 三阶段基础回归加旧 Marker 通道、Residual Prior Adapter、增强对齐测试：
  **139 passed，17 deselected**，其中包含旧版 U-Net 的两项测试和
  Marker/nnUNet 的 17 项测试。
- 新目录 MONAI/SSL 集成：**6 passed，89 deselected**。
- 334 个受保护的 Phase 2 原有源码、测试和文档 SHA-256 均未变化。
- 测试输出与保护清单保存在 `tmp/fdk_v3/phase2-cleanup/`，不纳入源码提交。
