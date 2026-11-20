# Phase 2 · Marker 引导 nnU-Net 联合预测 mask 与连续深度

v0.4，2026-11-18。本版固定深度模块和联合训练方法，唯一的实验变量是现有 marker 点先验的开关。

## 1. 目标与输出

输入一张含头模背景的投影和同视角的两个 marker 点。一次整图前向输出两个逐像素对齐的结果：

- 二值支架 mask，包含两端 marker。
- 连续深度图，数值为该像素对应射线在纯支架及 marker 占据体内的累计交长，单位 mm。最终数值非负，在模型自己的预测 mask 外严格为零。

预测 mask 决定了深度能够出现的位置。分割漏掉的真实支架像素在交付图中深度就是零，计算前景误差时仍计入。分割假阳性也可能得到错误深度，因此同时评价定位与回归。现有二值模型保留原 Trainer、权重和入口。新版位于同一个 Phase 2 nnU-Net 源码树，以新增 Trainer 和 Dataset504 独立保存实验。

## 2. 原网络与深度模块如何交互

**分割主干。** 复用原 E5-U 的九级 2D nnU-Net。全图 z-score 后的叠加投影 X 进入图像 encoder。现有 point、band 通道进入先验 encoder，经 validity 门控融合到前两级 skip。原 decoder 保留分割头与多尺度深监督。以下 P 取自最高分辨率的两类 logits，阈值恰好为 0.5 时归入背景。

```text
[X, point, band, validity] → 原 E5-U nnU-Net → 分割 logits Z
P = softmax(Z, dim=class)[:, foreground]       # 支架前景概率
H = stop_gradient(float(P > 0.5))             # 同一次前向的预测 mask
T = concat(X × H, H, P × H)                   # 深度模块的三个输入通道，顺序固定
C = softplus(DepthRefiner(T))                  # 非负条件深度，归一化单位
D_norm = H × C                                # 整张交付深度图
D_mm = train_depth_scale_mm × D_norm
```

| 接口 | 作用 |
| --- | --- |
| 掩膜内的投影 X × H | 为回归提供局部强度和纹理。全图背景仍可由主干用于定位。 |
| 硬 mask H | 显式告诉深度模块支架的形状。即使某个前景像素标准化后的投影值恰好为零，也能与背景区分。 |
| 掩膜内的概率 P × H | 保留分割对每个前景像素的置信度，并把深度误差传回主干。P 始终留在计算图中。 |
| 输出再乘 H | 卷积和偏置可以在 mask 外产生中间响应。最终门控保证输出的深度与导出的二值 mask 严格对齐。 |

**深度模块。** 五级 2D U-Net，通道数依次为 16、32、64、128、256。每级两次 3×3 卷积、带仿射参数的 InstanceNorm 和 LeakyReLU。编码端四次 2×2 最大池化；解码端双线性上采样到同级 skip 尺寸，拼接后再卷积；最后 1×1 卷积输出单通道。高分辨率 skip 保留细丝边缘，低分辨率特征提供支架邻域信息。回归只在全分辨率监督。Softplus 允许预测超过训练集归一化最大值 1 的连续标签。

**梯度。** 硬比较本身没有梯度。深度损失经 C、深度模块、P × H、P 回到分割 logits 和主干。在本次前向中，H 外的 P 没有来自深度支路的直接梯度；漏检位置仍接受分割 CE/Dice 的监督。全图预测为空时输出全零、统计一次空预测，仍使用当前预测执行训练。训练、验证和推理使用相同门控，GT mask 仅用于损失和评价。

**与独立 Phase 2.5 的区别。** 新增的 `depth_refiner` 与 `backbone` 都是 `trainer.network` 的子模块。每批一个联合前向、一次总损失反传，两组参数由同一个 AdamW 更新。深度模块看到的是本批次当前主干产生的 H 和 P，没有两个训练程序之间的 mask 文件。一个 checkpoint 同时恢复两者。默认 `forward(x)` 仍提供原生分割输出，配对推理调用 `forward(x, return_depth=True)`。[组合网络及损失](../../../nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided_depth/network.py) · [原生 Trainer 变体](../../../nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided_depth/nnUNetTrainerMarkerGuidedDepth.py)

## 3. 联合监督和训练预算

令 M 为 GT mask，Y 为归一化真值深度，Q 为探测器原始画布。损失为：

```text
L_seg   = 深监督加权的二类交叉熵 − 前景 soft Dice
L_fg    = 在 Q 内所有 GT 前景上，逐图平均 SmoothL1(D_norm, Y, beta=0.05)
L_bg    = 在 Q 内所有 GT 背景上，逐图平均 abs(D_norm)
L_total = L_seg + L_fg + 0.1 × L_bg
```

无相应区域时，该图对应训练损失项为零并计数。分割深监督按 1、1/2、1/4 … 加权，最低分辨率权重为零，再归一化。粗尺度类别标签取最近邻，只有对应区域全部位于 Q 内时计入监督。验证前景 MAE 使用所有有 GT 前景的图像，包括预测完全漏检的图像。

| 设置 | 正式两组共用的取值 |
| --- | --- |
| 初始化 | 严格加载同一份 Dataset501 E5-U final 分割权重。深度模块采用相同种子、相同初值。 |
| 优化 | 从第一步起联合更新。AdamW，主干学习率 1e-5，深度模块学习率 1e-4，weight decay 1e-4，固定学习率。 |
| 预算 | 100 轮，每轮 250 张训练图，batch 1，累积 2 个微批次，正常每轮 125 次参数更新。AMP。 |
| 采样 | 相同随机种子、样本顺序和水平/垂直翻转。按训练集洗牌依次取图；每轮完整验证 720 张。 |
| 选模 | 最小化验证集逐图 GT 前景 MAE。best、latest 和 final 的用途分别为效果评价、轮次边界续训和训练终点。 |

## 4. 唯一消融变量为 marker 先验

| 实验 | 进入 E5-U 的先验 | 深度模块及其训练 |
| --- | --- | --- |
| Marker 开启，主实验 | 从同视角两个几何 marker 坐标生成原 point、band、validity | 固定上述三通道深度模块，从第一步联合训练 |
| Marker 关闭，独立训练对照 | 将 point、band、validity 同时置零，图像通道不变 | 相同结构、初值、样本顺序、损失、学习率和训练预算 |

两组都保留完整 E5-U 架构。关闭三张图测量整个点先验的贡献，可避免通过连接带或有效性图保留点位信息。两个实验都继承经过 marker 引导预训练的分割权重，因此两组差异解释为**在相同预训练基础上，联合训练与推理能从 marker 先验获得多少增益**。

另在开启组的同一份 best 权重上，对同一验证图做开、关先验的配对推理，测量训练后模型对输入先验的即时依赖。两组报告分割 Dice/precision/recall、全部 GT 前景上的 MAE（归一化及 mm）、RMSE、漏检误差、GT 背景误差、空预测率与预测 mask 外的非零深度像素数。比较按同一物理视图配对。测试集 60 张单独留出，用于选模结束后的评价。

此前讨论的软概率门控 B 保留为备选设计：可取 `T=[X×P,P]`、`D=P×C`，其输出支持域不同。本次实验固定硬门控、三通道输入和联合训练，不安排软门控、冻结主干、去掉深度模块或去掉概率通道的实验。

## 5. 数据、单位与预处理

以物理清单的病例与视角为主键，194 个物理病例、每例 30 个视角。训练、验证、测试按支架系列分开，分别有 **5,040、720、60** 张。当前训练/验证使用投影几何给出的 marker 真值，部署时换成 Phase 1 预测点的表现需单独评价。

采用保存的 TIGRE 几何对支架与两端 marker 的二值占据体重新做 Siddon 射线积分。生成 300 张独立 float32 原始深度图及配对的 uint8 mask，由相同支架和投影几何的多个背景共享。5,820 张输入已逐张核对，新深度正值支持域与原标签的最小 IoU 为 1。已知厚度体模在 0°、90° 下的最大绝对误差为 0.000786 mm。训练集 210 张独立目标决定尺度 **1.6377511024475098 mm**。测试集最大值为 1.722058 mm，超出训练范围的值按原量保存。[数据清单及几何核验](/data1/liuyang/stent/proj_1/training_data/training/nnunet/phase2/Dataset504_MarkerDepth/manifest.json)

原图从清单引用的既有完整 TIFF 读取。point 和 band 使用原 E5-U 生成器，validity 依点对是否可用取值。原图先在 1056×1056 画布做 z-score，再将各字段同步向右、向下补零至 1280×1280。独立的有效区域 Q 用于排除 padding。训练只做同步水平/垂直翻转，预测恢复原尺寸。数据加载按同一记录读取图像、mask 和浮点深度。[整图读取代码](../../../nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided_depth/data.py) · [目标生成代码](../../../nnUNet/nnunetv2/dataset_conversion/phase2_joint_depth.py)

## 6. 训练、重载与验收

本机入口 `nnUNet/phase2_joint.py train` 调用项目原生 `nnunetv2.run.run_training`，分别使用 `nnUNetTrainerMarkerGuidedDepth` 与 `nnUNetTrainerMarkerGuidedDepthNoMarker`。初始化权重来自 Dataset501 的 final；已核对其训练系列与本数据的验证、测试系列没有交集。正式运行保存 PID、日志、TensorBoard 及两组各自的 checkpoints。[本机入口](../../../nnUNet/phase2_joint.py) · [原始权重来源](/data1/liuyang/stent/proj_1/training_data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/results/pretrained_from_Dataset501/weights_manifest.json)

1. 在同一输入上比较包装前后的分割 logits；在非空 mask 上单独反传深度损失，检查主干和深度模块均收到有限非零梯度。检查 H 外的 P 没有直接深度梯度。
2. 两个配置分别通过整图短训练，核对实际优化器更新，保存和重载联合 checkpoint，导出同次前向得到的二值 mask 和 float32 毫米深度。
3. 用原尺寸导出图逐像素检查非负、有限、`depth[mask==0] == 0`。完整验证集前景 MAE 用于选择各自 best。没有标签的真实帧可用于观察，精度结论以有标签数据为准。

独立的真实图像推理提供原图及同帧 marker 坐标，或显式关闭先验。一次前向导出二值 mask、归一化深度、毫米深度及元数据，可选保存前景概率。最终验证也使用这套联合接口。[预测与导出代码](../../../nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided_depth/inference.py)

### 本版实测记录

使用正式初始化权重及训练集中一个 512×256 原像素裁块，单独反传真实毫米标签对应的归一化深度损失。分割输出头的权重梯度 L2 范数为 **0.02118**，深度输出头为 **4.75128**。H 外的 P 梯度非零数为 **0**，包装前后的分割 logits 最大差为 **0**。此检查验证两个模块之间的梯度连接。[原始梯度记录](/data1/liuyang/stent/proj_1/training_data/training/nnunet/phase2/Dataset504_MarkerDepth/logs/initial_depth_gradient.json)

两组都通过 2 轮全尺寸短测。重载各自 best 后导出的四组 mask / 深度 TIFF 均为 1056×1056，毫米深度 float32、有限、非负，预测 mask 外逐像素为零。2026-11-18 已启动两个 100 轮正式实验，分别在物理 GPU 3 和 2 上运行，首次更新均观测到主干和深度模块的非零梯度。[marker 开启训练日志](/data1/liuyang/stent/proj_1/training_data/training/nnunet/phase2/Dataset504_MarkerDepth/logs/nnUNetTrainerMarkerGuidedDepth.log) · [marker 关闭训练日志](/data1/liuyang/stent/proj_1/training_data/training/nnunet/phase2/Dataset504_MarkerDepth/logs/nnUNetTrainerMarkerGuidedDepthNoMarker.log)
