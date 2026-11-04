# Phase 2：Marker 引导的 nnU-Net E5-U

本文介绍当前实现的网络、输入数据格式、运行流程和可调参数。首版实现的是 **E5-U**：原始投影加 Marker 点先验、有限线段带状先验，以及有效性门控。它保留 nnU-Net 的 plans、原生 decoder、loss、深监督、checkpoint 和滑窗推理；先验分支只在高分辨率 encoder skip 上调制图像特征。

E5-U 的第四通道是二值有效性图，不是连续置信度。Phase 1 的有效点对即使 `q_pair=null` 也能使用；当前实现不运行依赖校准可靠度的 E5，也不包含 FiLM、DCNv2 或 clDice 训练损失。合成 smoke 只验证软件链路，不代表分割效果。

## 1. 网络结构与具体改动

输入张量形状为 `[N, 4, H, W]`，通道顺序固定：

| 通道 | 内容 | 网络用途 |
|---|---|---|
| 0 | 原始二维投影 | 输入 nnU-Net 原生 `PlainConvUNet` 图像 encoder |
| 1 | 两个 Marker 的点热图 | 和 band 一起输入轻量 prior encoder |
| 2 | 两点有限线段生成的软 band 图 | 和 point 一起输入轻量 prior encoder |
| 3 | validity 图 | 在融合层控制先验调制强度 |

### 图像主干

网络从当前 nnU-Net 2D plans 读取 `n_stages`、`features_per_stage`、卷积核、stride、归一化及 decoder 配置。图像 encoder 只读通道 0；plans 所定义的原生 decoder 仍负责深监督输出。网络会检查输入必须为四通道、plans 必须是二维 `PlainConvUNet` 和 `Conv2d`。其它架构会明确报错。

### 先验 encoder

通道 1–2 进入独立的小型 encoder：两层 `3×3 Conv2d + InstanceNorm2d + LeakyReLU`。默认宽度为 `max(8, int(min(features_per_stage) * 0.25))`。编码结果插值到选定 skip 的空间尺寸，再用 `1×1 Conv2d` 投影到对应图像特征通道数。

默认融合 encoder 的前两个 stage，即全分辨率和下一层较高分辨率特征；plans 少于两个 stage 时只使用可用 stage。它不改 nnU-Net decoder 的层级或结构。

### Residual Prior Adapter

每个融合层将图像特征 `F` 与先验特征拼接，经 `1×1 Conv → SiLU → 1×1 Conv → sigmoid` 得到注意力图 `A`。validity 图重采样到该层尺寸并限制在 `[0,1]`，记为 `q_l`，然后执行：

```text
F' = F × [1 + q_l × beta_l × (2 A_l - 1)]
```

`beta_l` 初值为 `0.05`，上限为 `0.5`。注意力最后一层零初始化，因此初始 `A=0.5`、adapter 初始输出逐元素等于输入特征，训练从接近原生 nnU-Net 的状态开始。若某张图是 `invalid`，其 validity 为 0，该层调制为恒等映射；这不等于整网退化成另一套无先验训练结果。

### 增强行为

空间旋转、缩放、翻转同步作用于四个图像通道和分割标签；增强后 point、band、validity 限制在 `[0,1]`。强度增强只作用于通道 0，避免噪声、亮度或 gamma 变换污染先验图。强度增强默认概率：高斯噪声 0.10、模糊 0.20、亮度 0.15、对比度 0.15、低分辨率模拟 0.25、反相 gamma 0.10、普通 gamma 0.30。空间旋转概率 0.20，缩放概率 0.20、范围 0.7–1.4；镜像轴由 nnU-Net plans 提供。

## 2. Marker 先验的生成

对有效点对 `m1,m2`，点图是两个高斯热图的逐像素最大值。每个点的 sigma 根据误差半径 `r_i` 扩大：

```text
sigma_i = sqrt(sigma_0^2 + (k_sigma × r_i)^2)
point(p) = max_i exp(-||p - m_i||^2 / (2 sigma_i^2))
```

band 是到**有限线段**的高斯距离图，并在两端外继续衰减：

```text
w = w_0 + k_w × max(r_1, r_2)
band(p) = exp(-distance_to_segment(p)^2 / (2 w^2)) × endpoint_taper(p)
```

如果两点距离小于 `d_min`，保留 point 图，band 置零。点顺序不影响结果。有效 `reliable` 和 `uncertain` 点对都会生成先验，validity 整图为 1；`invalid` 时 point、band、validity 全为 0，但原始投影仍保留。Mask 只作为训练标签，绝不参与该样本的先验生成；band 也不用于裁剪投影或硬掩膜。

转换器默认参数如下。它们是工程起点，不是已由真实数据选出的最优值；正式实验应只用当前 Phase 2 fold 的训练组选择。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `sigma_0` | 2.0 px | 点高斯的基础宽度 |
| `k_sigma` | 1.0 | 点定位误差半径对 sigma 的放大系数 |
| `w_0` | 3.0 px | band 基础宽度 |
| `k_w` | 1.0 | 点定位误差半径对 band 宽度的放大系数 |
| `endpoint_taper_px` | 4.0 px | 线段端点外衰减尺度 |
| `d_min` | 1.0 px | 判为退化短线段的距离阈值 |

## 3. 输入数据格式

转换器接受一个 Phase 2 数据清单和一个 Phase 1 预测包。JSON 路径可用绝对路径，或相对对应 JSON 文件所在目录的路径。一个二维视角对应一个 nnU-Net case；同一 `case_id` 的三个视角必须保持同一个 `group_id` 和 split。

### Phase 2 manifest

```json
{
  "phase2_fold_id": "fold_0",
  "splits": {
    "train": ["scaffold_00", "scaffold_01"],
    "val": ["scaffold_02"],
    "test": ["scaffold_03"]
  },
  "samples": [
    {
      "case_id": "case_0001",
      "view_id": 0,
      "nnunet_case_id": "case_0001_v0",
      "group_id": "scaffold_00",
      "split": "train",
      "projection_path": "images/case_0001_v0.tif",
      "mask_path": "masks/case_0001_v0.tif",
      "geometry_id": "sim_geometry_v1",
      "angle_deg": -15.0
    }
  ]
}
```

`samples` 中每个 `(case_id, view_id)` 唯一，`nnunet_case_id` 全局唯一；每个 case 恰好有三个不同视角。投影和标签必须是相同尺寸的有限二维 TIFF；训练/验证 mask 当前要求二值 `0/1`，测试样本不需要 `mask_path`。每个样本必须提供 `angle_deg` 或 `geometry_manifest_key`，并提供 `geometry_id`。`splits` 按 group 划分且互斥。

### Phase 1 prediction case

E5-U 接受有效点对的 `q_pair=null`。每条病例记录使用 `phase1_marker_case_v1`，并包含以下信息：

```json
{
  "schema_version": "phase1_marker_case_v1",
  "case_id": "case_0001",
  "group_id": "scaffold_00",
  "model_id": "marker_model_oof_00",
  "geometry_id": "sim_geometry_v1",
  "calibration_id": null,
  "prediction_source": "group_oof",
  "fold_id": "phase1_oof_excluding_scaffold_00",
  "views": [
    {
      "view_id": 0,
      "image_size": [64, 64],
      "marker_xy": [[25.2, 24.8], [45.1, 41.0]],
      "error_radius_px": [3.0, 3.5],
      "q_pair": null,
      "pair_status": "reliable"
    }
  ]
}
```

实际 `views` 必须包含恰好三个不同视角。有效状态为 `reliable` 或 `uncertain`，各自需要两个有限、在图像范围内的 `(x,y)` 坐标和两个非负有限误差半径。`invalid` 需要 `marker_xy=null`、`error_radius_px=null`、`q_pair=0`。校准 `q_pair` 若为数值，还必须有可追溯的 `calibration_id`、正的 `q_tolerance_px` 和 calibration registry 记录。

Phase 2 侧预测包的当前封装要求是：

```json
{
  "phase2_fold_id": "fold_0",
  "source_registry": [
    {
      "model_id": "marker_model_oof_00",
      "model_artifact_id": "checkpoint-sha256-or-id",
      "pipeline_config_id": "decoder-fusion-config-id",
      "fit_group_ids": ["scaffold_01"],
      "selection_group_ids": []
    }
  ],
  "calibration_registry": [],
  "cases": ["上面的完整病例记录"]
}
```

训练组只能使用 `group_oof` 记录，且该样本所在 group 不能出现在对应模型的拟合组或参数选择组。验证/测试只能使用 `held_out` 记录，模型拟合/参数选择组必须来自该 fold 的训练组。连接只按 `(case_id, view_id)` 做精确匹配，不按文件顺序或相似文件名匹配。

> **Phase 1 当前导出格式与上述封装尚未对接。** `phase_1/marker_pipeline/export.py` 现在写出的是包含 split 元数据的 `package.json`、字典形式的 `source_registry.json`、`predictions/<case_id>.json` 和 `case_view_index.csv`；Phase 2 验证器当前却要求 `package.json` 内联 `cases` 数组，且 `source_registry` 是带 artifact/config ID 的列表。Phase 1 原始 `manifest.json` 也不是 Phase 2 的 `samples` 清单。因此不能把 Phase 1 导出目录直接传给下面的转换命令；需要先写一个小型适配器，按样本映射、来源登记和 group split 生成上面两种 Phase 2 输入 JSON。

### 转换后 nnU-Net 目录

转换器按通道写四个独立 TIFF：

```text
Dataset501_MarkerGuided_fold_0/
  dataset.json
  splits_final.json
  case_mapping.json
  source_registry.json
  imagesTr/
    case_0001_v0_0000.tif  # 原始投影
    case_0001_v0_0001.tif  # point
    case_0001_v0_0002.tif  # band
    case_0001_v0_0003.tif  # validity
  labelsTr/
    case_0001_v0.tif
  imagesTs/
    ...
```

训练和验证进入 `imagesTr/labelsTr`，测试投影进入 `imagesTs`。`dataset.json` 声明通道 0 使用 `zscore`、其它通道 `nonorm`。`splits_final.json` 以 nnU-Net case ID 保存 group-wise train/val；planner 会将它复制到 preprocessed 数据目录。输出目录非空时转换器会拒绝写入，以避免旧 TIFF 混入新数据集。

## 4. 安装与运行

以下命令假定从仓库根目录 `D:\desktop\课题1\code` 运行，并且已经把 Phase 1 导出转换成上一节定义的 Phase 2 manifest/package。

```powershell
python -m pip install -e .\phase_2\nnUNet
$env:nnUNet_raw = 'D:\nnUNet\nnUNet_raw'
$env:nnUNet_preprocessed = 'D:\nnUNet\nnUNet_preprocessed'
$env:nnUNet_results = 'D:\nnUNet\nnUNet_results'
$env:nnUNet_compile = 'false'
```

先用新 Dataset ID 为每个 fold/实验创建独立数据集：

```powershell
python -m phase_2.tools.marker_guided.cli validate `
  --manifest D:\data\phase2_manifest.json `
  --package D:\data\phase2_prediction_package `
  --phase2-fold-id fold_0

python -m phase_2.tools.marker_guided.cli build-e5u `
  --manifest D:\data\phase2_manifest.json `
  --package D:\data\phase2_prediction_package `
  --phase2-fold-id fold_0 `
  --dataset-id 501 `
  --output-dir "$env:nnUNet_raw\Dataset501_MarkerGuided_fold_0"
```

之后走 nnU-Net 正常入口：

```powershell
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

# 短软件 smoke：2 epochs、每 epoch 2 个训练 iteration 和 1 个验证 iteration
nnUNetv2_train 501 2d 0 -tr nnUNetTrainerMarkerGuidedSmoke -device cpu

# 正式预算使用主 Trainer
nnUNetv2_train 501 2d 0 -tr nnUNetTrainerMarkerGuidedFeatureFusion -device cuda

# 输入必须包含 0000–0003 四通道文件
nnUNetv2_predict `
  -i "$env:nnUNet_raw\Dataset501_MarkerGuided_fold_0\imagesTs" `
  -o D:\nnUNet\predictions\Dataset501_fold_0 `
  -d 501 -c 2d -f 0 `
  -tr nnUNetTrainerMarkerGuidedFeatureFusion -device cuda
```

预测时使用与 checkpoint 相同的 Trainer 名称，nnU-Net 会按 Trainer 静态构建方法恢复网络结构。训练前确认 `$env:nnUNet_preprocessed\Dataset501_MarkerGuided_fold_0\splits_final.json` 存在。预测输出可结合 `case_mapping.json` 回溯到原始 `(case_id, view_id)`。

## 5. 可调参数

### 先验生成参数

命令行可直接调整 `--sigma-0`、`--k-sigma`、`--w-0`、`--k-w`、`--endpoint-taper-px` 和 `--d-min`。这些值写进 `dataset.json` 的 `marker_guided.prior_config`。每次修改先验参数应使用独立 Dataset ID，并且只基于当前 fold 训练组选择，避免不同实验覆盖彼此的数据。

### 网络参数

| 参数 | 当前设置 | 如何调整 |
|---|---|---|
| 图像主干 | nnU-Net plans 的 2D `PlainConvUNet` | 用相应 dataset/config 重新运行 planning；需保持 `2d` |
| 融合 stage | 默认 encoder stage `[0, 1]` | 目前是网络构造参数；高级实验需在 plans 的 architecture kwargs 中传 `fusion_stage_indices` 或改构建默认值 |
| prior encoder 宽度比例 | `prior_feature_ratio=0.25` | 可作为自定义网络 kwargs 传入，控制先验分支宽度；必须重新训练 |
| adapter 初始 beta / 上限 | `0.05` / `0.5` | 当前在 adapter 构造代码中设定；需要改源码并重新训练 |
| 输入通道 | 固定 4 | `[projection, point, band, validity]` 顺序不能改变 |

若改 plans 的 patch size、batch size、features 或训练预算，正式对照组应使用一致的设置。更改 prior、plans、增强或网络后都需要重新训练模型；不能将旧 checkpoint 和新输入配置混用。

### 训练增强参数

旋转/缩放概率、缩放区间、各强度变换概率和范围定义在 `nnUNetTrainer/variants/marker_guided/marker_guided_transforms.py`。smoke Trainer 固定短预算；正式训练使用 `nnUNetTrainerMarkerGuidedFeatureFusion`，其余训练、验证和 nnU-Net loss 行为继承自原生 Trainer。若调整增强，强度变换仍必须只覆盖通道 0，空间变换仍需同步覆盖四通道和标签。

## 6. 当前验证状态与运行限制

先验转换、适配器和增强单测曾验证为 **17 passed**；真实 nnU-Net `plan_and_preprocess → train → checkpoint reload → predict` 集成 smoke 仍未完成。当前环境安装的 `batchgenerators 0.25.1` 低于 nnU-Net `pyproject.toml` 要求的 `>=0.25.5`，导入 Trainer 时会因缺少 `SEG_TIEBREAKS` 失败。安装满足仓库依赖的环境后，还需先实现上面说明的 Phase 1 → Phase 2 输入适配器，再运行端到端 smoke。

即使合成 smoke 通过，它也只证明软件链路可运行。真实实验还需核对数据几何、来源审计、GPU 内存、组间划分和先验参数，并在真实折外数据上评价；本网络实现本身不构成性能提升证据。
