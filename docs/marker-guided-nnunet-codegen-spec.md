# Marker 引导的 nnU-Net v2 网络改造 Spec（任务优先版）

首版实施步骤见 [Phase 2 Plan](marker-guided-nnunet-implementation-plan.md)，跨阶段数据定义见[交接契约](../../docs/phase1-phase2-handoff-design.md)。

## 1. 设计原则

本方案优先适配当前支架投影任务，不追求复现某一篇论文的完整网络。

论文只承担三种作用：

1. 提供可引用的模块思想；
2. 帮助设计消融实验；
3. 为损失和评价指标提供理论依据。

最终网络必须服从以下任务事实：

- Marker 检测由 Phase 1 完成；Phase 2 主实验接收其三视角融合后的逐视角预测，不能假设坐标是真值或始终可靠；
- 两个 Marker 中心在每张投影的视野内；它们提供有误差的端点线索，但两点连线可能偏离弯曲支架，不能预设为 Mask 的中心线或完整覆盖范围；
- 支架金属丝可能只有几个像素宽；优先验证全分辨率与半分辨率 skip 的贡献，具体宽度需由训练集测量；
- 支架是网状/细长结构，断裂和端部漏检比少量边界误差更严重；
- 仿真数据量通常有限，不能同时堆叠 Attention U-Net、FiLM、DCNv2、复杂拓扑损失而失去可归因性；
- nnU-Net 已经负责 plans、patch、归一化、深监督、训练和推理，改动应尽量局部。

## 2. 论文参考边界

| 论文 | 在本任务中的定位 |
|---|---|
| nnU-Net | 保留其数据驱动的预处理、plans、训练和推理流程，不固定主干结构 |
| Attention U-Net | 作为“先验引导空间选择”的参考；不强制使用完整 custom decoder |
| FiLM | 作为可选的 prior feature modulation 对照；不是主网络必需组件 |
| DCNv2 | 只作为后续局部形变采样对照；不用于第一版主网络 |
| clDice | 第一阶段作为连通性评价指标，第二阶段再考虑作为辅助损失 |
| Soft Rasterizer | 不作为核心实现；中心线先验使用更简单、可验证的 soft distance field |

因此，Marker point/band prior、轻量 residual prior adapter 和 Marker-constrained centerline 是本课题的任务特定设计，不应写成上述论文已经提出的方法。

## 3. 目标和非目标

### 3.1 目标

1. 首版先交付可运行的 Phase 1 预测结果到二维支架 Mask 分割链路；真实数据上的性能收益随后验证，真值坐标先验仅作上限对照。
2. 在现有 `phase_2/nnUNet/` 源码中，以新增自定义 Trainer/网络的方式实现轻量 prior branch，在高分辨率 encoder skip 上调制原图特征。
3. 保留 nnU-Net 原生 decoder、Dice+CE、深监督、checkpoint、DDP 和 inference。
4. 后续通过 E0/E1-P/E1/E2/E5/E6 对照分别验证点先验、直线带先验、特征融合和弯曲建模；这些实验不阻塞首版代码交付。
5. 记录细丝断裂、端部漏检和连通性变化。

### 3.2 非目标

- 不实现 Marker 检测网络。
- 不把 Marker 坐标直接拼接到全连接层或 bottleneck。
- 不使用先验做硬 ROI 裁剪或 hard mask。
- 不在第一版加入 DCNv2、完整 Attention U-Net decoder 和 soft-clDice 训练损失。
- 不修改 nnU-Net 全局 Trainer、默认 loss、planner 或 inference predictor。

## 4. 首版实现与后续实验路线

```
E0: raw                                  -> 原生 nnU-Net
E1-P: raw + marker_point                 -> 原生 nnU-Net，点先验单独对照
E1: raw + marker_band                    -> 原生 nnU-Net
E2: raw + marker_point + band            -> 三通道拼接基线
E2-U: raw + point + band + 有效性常数图   -> 四通道无可靠度拼接基线
E2-Q: raw + point + band + q             -> 四通道拼接基线
E5-U: raw + point + band + validity + gated adapter -> 首版可运行分割网络
E5: raw + point + band + 校准 q + gated adapter      -> 后续可靠度感知网络
E5-F: E5 + spatial FiLM         -> 可选特征调制对照
E5-T: E5 + soft-clDice          -> 可选拓扑损失对照
E6: 五通道端点先验 + deformable centerline -> 条件性的弯曲支架研究扩展
```

首版只要求 E5-U 的完整可运行链路：核验 Phase 1 折外预测包，生成 `[projection, point, band, validity]` 二维 TIFF 数据，在现有 nnU-Net v2 源码中新增自定义 Trainer/网络，完成 plan、preprocess、训练、checkpoint 恢复及滑窗推理。`validity` 对有效点对为 1、明确 `invalid` 为 0，不冒充校准概率。point 与 band 作为软信息输入，原始投影始终保留；弯曲支架可能离开直线 band，因此绝不以 band 做硬裁剪或硬掩膜。E0/E1-P/E1/E2/E2-U、适用性审计和消融属于后续实验；E5-U 的代码实现不以它们的性能结果为前提。只有 Phase 1 提供经独立病例审核的 `q_pair` 后才运行 E2-Q/E5；E6 还需解决全局端点坐标与 patch/推理的一致性。

## 5. 输入和先验契约

### 5.1 跨阶段输入与网络通道

Phase 1 的正式接口见 [Phase 1 spec 第 5 节](../../phase_1/docs/marker-detection-codegen-spec.md)，字段、空值及样本映射统一遵守[两阶段交接契约](../../docs/phase1-phase2-handoff-design.md)：每个病例 JSON 含同一 `case_id` 的三个视角，每个 `(case_id, view_id)` 对应原始图像上的两个无序浮点坐标、每点 `error_radius_px`、`pair_status` 和可选的已校准 `q_pair`，并保留 `schema_version`、`group_id`、`image_size=[H,W]`、`geometry_id`、`prediction_source`、`fold_id`、具体冻结检测流程 `model_id`、`calibration_id`、`q_tolerance_px`。先验生成器消费该结构化记录；原始 U-Net 检测热图、Top-K 和三维点只是诊断，不作为 E5-U 主输入。`association_status` 不是 E5-U 的必需条件，因为 point/max 和 band 对两个端点交换不敏感。

首版 E5-U 网络输入：

```
x[:, 0:1] = projection
x[:, 1:2] = marker_point
x[:, 2:3] = marker_band
x[:, 3:4] = validity_map  # 有效点对为 1，invalid 为 0
```

E2 保留前三通道；E2-U、E2-Q、E5-U、E5 使用相同四通道顺序。U 组第四通道对有效检测恒为 1，对明确 `invalid` 为 0；Q/E5 组使用校准 `q_pair`。point、band、第四通道在进入 nnU-Net raw dataset 前离线生成并作为独立图像通道保存。point/band 不再乘 `q_pair`，防止与门控重复衰减。

网络 `forward` 只接收堆叠的图像通道，不直接接收坐标参数。训练、验证和测试推理必须使用同一先验生成函数、坐标约定、先验配置版本和通道顺序。Phase 2 数据清单显式保存 `(case_id, view_id) → nnunet_case_id`、原投影、标签、`group_id`、split 与 `geometry_id` 的映射；每个 `phase2_fold_id` 使用独立的 Phase 1 预测包，在包内按二元键精确关联记录，不依赖文件排序。逐例核对实际投影尺寸、坐标界限、标签几何、支架组和几何版本。缺失记录、尺寸不符、非有限数值或无法解释的 `null` 必须报错，不能默默当作检测失败。明确 `invalid` 且 `marker_xy=null`、`error_radius_px=null`、`q_pair=0` 时允许生成零先验和零可靠度，原投影仍进入分割网络。

`q_pair` 是两个 Marker 在指定原始像素容差内同时正确的校准概率，不是峰值分数。逐视角 `q_tolerance_px` 如存在，覆盖病例级默认值。有效检测的 `q_pair=null` 表示校准不足：Q/E5 组不得填 0、1 或峰值分数来冒充概率；首版直接使用 E5-U，后续可使用 E1-P/E1/E2/E2-U 对照。U 组的有效性常数图只由 `pair_status` 决定，不从 `q_pair` 推得；`uncertain` 但有两个有效点的常数图仍为 1。只有明确 `invalid` 的约定零值可在没有校准器时出现；Q/E5 整个数据集仍须满足所有**有效**记录具有已审核的数值 `q_pair`。Phase 2 必须记录生效的 `q_tolerance_px` 与 `calibration_id`，并检查 Phase 1 的独立病例校准报告；未通过校准门槛不能把 E5 称为可靠度感知主方法。

E5-U 的 `dataset.json` 关键字段示例；E5 以后使用独立 dataset/config，并将第四通道改为校准 `q_pair`：

```json
{
  "channel_names": {
    "0": "zscore",
    "1": "nonorm",
    "2": "nonorm",
    "3": "nonorm"
  },
  "marker_guided": {
    "raw_channel": 0,
    "prior_channels": [1, 2],
    "gate_channel": 3,
    "prior_names": ["point", "band"],
    "gate_name": "validity",
    "fusion_stages": "plan_selected_highres",
    "prior_feature_ratio": 0.25
  }
}
```

### 5.2 先验生成与失败处理

- point：以 Phase 1 最终二维点为中心，按每点宽度 `sigma_i` 生成 Gaussian 并逐像素取 max；`sigma_i` 由 Marker 尺寸基础宽度与该点 `error_radius_px` 决定，规则在当前 fold 训练组内部确定；
- band：两个预测点之间的有限线段柔性距离场，带端点范围衰减；基础宽度从训练集 Mask 的覆盖与背景代价共同确定，再随定位误差适度增加。它仅表示待验证的宽松空间走廊；弯曲支架的金属丝可能明显离开 Marker 连线，不能用增加宽度掩盖先验已失去空间选择性的情况；
- 两点距离过短、无法稳定定义线段方向时，使用当前 fold 训练组内部冻结的退化规则生成零 band、保留有效 point 并记录原因；不能让退化线段产生跨越全图的先验；
- reliability：E5/E2-Q 的有效检测用 `q_pair` 生成值域 `[0,1]` 的常数图；E5-U/E2-U 则用有效性 1/0 常数图；明确 `invalid` 时 point/band/第四通道全零。低 `q_pair` 不会再次乘到 point/band 幅值；
- E1-P 只使用 point；E1 只使用 band；E2 使用两者。三者使用相同预测坐标及同一先验生成规则，均不提供第四通道；真值坐标生成先验的实验单独标为 oracle，不能混入主结果；
- 先验只依赖 Phase 1 记录、投影尺寸及在当前 fold 训练组内部固定的生成规则，不能读取当前支架 Mask；坐标统一为原始图像 `x=列、y=行`，保留亚像素值；
- 生成前检查 `(case_id,view_id)`、`image_size=[H,W]`、图像几何和原始投影一一对应；四通道与标签具有一致的 geometry；
- point/band/reliability 使用 `nonorm`，不能进行逐图 z-score；可靠度图不得被强度增强改写；
- 用当前 fold 训练组 Mask 统计不同带宽下金属丝覆盖率和背景覆盖比例，在该训练组内部划分内层验证组选择宽度/误差放大规则；外层验证与测试 Mask 不用于调整先验规则；
- nnU-Net 重采样和几何增强后检查峰值、零图、可靠度范围、边界插值及通道同步；全局常数 `q` 经插值/填充进入 patch 时可出现边界过渡，门控始终以局部图值为准，不从 patch 内热图峰值反推整例 `q`。

### 5.2.1 直线先验适用性审计（后续研究评价）

首版 E5-U 使用直线 band 作为可被网络忽略的软输入，不以本审计为代码实现门槛。后续要声称 band 对弯曲支架有益、比较各先验方案或启用校准 q 的 E5 时，在当前 Phase 2 fold 的训练组及其内层验证组上完成以下审计，并保存每个 fold 的报告；外层验证和测试组只用于冻结方案的评价：

1. 明确二维 Mask 是否包含 Marker 本体，以及 Marker 到实际支架端部和金属丝区域的关系；用已有 3D 模型或训练集 Mask 统计，不能把 Marker 点自动当作 Mask 端点。
2. 对真值 Marker 和 Phase 1 折外预测分别生成 point/band，随 band 宽度绘制金属丝 Mask 覆盖率与背景覆盖比例；报告全体、弯曲程度、角度间隔、定位误差及 `pair_status` 分桶。先固定 prior 阈值和统计区域，再比较宽度，避免仅凭热图视觉效果判断覆盖。
3. 若弯曲样本只能靠过宽 band 才覆盖金属丝、同时大量纳入背景，记录直线先验不适用的样本比例；以 E1-P 和 E2 对照判断 band 是否增加净收益。此时不把 E5 的新增容量当作直线 band 有效的证据，也不自动转向 E6。
4. 在当前 fold 训练组内部用真值 Marker 坐标施加不同像素扰动，测量先验覆盖和分割指标的变化，确定定位容差 `τ`、point 宽度和误差半径放大规则；外层验证与测试组不参与选择。`q_pair` 只表示点对在 `τ` 内正确的概率，不表示直线 band 覆盖弯曲支架的概率。

### 5.3 预测来源与数据划分

主实验按 `phase2_fold_id` 分别准备预测包。某个 Phase 2 fold 的训练先验来自 Phase 1 在**该 fold 的原始 3D 支架训练组内部**进行的折外预测，记录 `prediction_source=group_oof`：生成该病例记录的检测器不能训练过该组，三视角必须一起留出，也不能借用该 fold 验证组训练。该 fold 验证记录由只在该 fold 训练组内拟合、调参和审核的冻结 Phase 1 流程产生，记录 `prediction_source=held_out`；最终测试记录同理只使用允许的开发训练组确定。不能在验证/测试组上重新调检测器、几何评分、校准器或先验规则。同一病例在不同 Phase 2 fold 可对应不同 Phase 1 记录，不能复用上一 fold 的预测。若要重新训练 Phase 1 全量训练组模型用于最终推理，应单独检查其输出坐标误差、失败率和 `q_pair` 分布与折外训练数据的差异。

`q_pair` 与误差半径的校准也要尊重此划分：Phase 2 训练记录的校准值应来自对该支架组未拟合过、且只使用该 Phase 2 fold 训练组数据的校准器，或来自其训练组内事先隔离的校准组；Phase 2 验证记录只能由该 fold 训练组内完成拟合和独立审核的固定校准器产生，最终测试同理。若数据量不足以同时做到折外检测和可靠校准，首版仍可运行 E5-U，使用明确的有效性 1/0；不得以未校准 `q` 作为 E5 输入。保存每例 `prediction_source`、Phase 1 `fold_id`、`model_id`、`calibration_id`、先验生成版本以及包级 `phase2_fold_id`；另由来源登记表列出各检测流程的权重拟合组、参数选择组及校准器拟合组与审核报告，逐例核对当前 `group_id` 不在对应拟合组内。同一支架组不得跨 Phase 1/Phase 2 的训练与评估边界；只有 `fold_id` 字符串不足以完成泄漏审计。

### 5.4 E6 输入

若采用显式中心线，E6 需要保持两个端点的身份，候选五通道数据为：

```
channel 0 = projection
channel 1 = marker_1_heatmap
channel 2 = marker_2_heatmap
channel 3 = marker_band
channel 4 = marker_reliability
```

这只是候选输入契约，不表示可以直接送入现有 patch 训练。nnU-Net 的裁剪/空间增强可能让一个或两个 Marker 落在 patch 外；从当前 patch 的热图做 soft-argmax 将得到错误端点。实现显式中心线前，必须先验证全局端点如何随预处理、patch 坐标和增强同步传递到网络，以及训练与 sliding-window inference 如何一致处理。若没有可靠适配器，E6 保持研究设计，不进入实现和主结果。E6 必须有相同输入、相同训练设置的静态带状先验对照 E6-base。

Phase 1 的 `association_status=resolved` 只表示三视角间临时身份一致，不自动提供解剖学端点身份。若两个 Marker 无此身份，生成独立 heatmap 时必须先定义可复现的端点排序规则，并验证旋转/翻转后坐标与通道身份同步；否则 E6 预测的法向偏移符号不稳定。E6 也必须沿用折外预测、可靠度和无效检测规则：`invalid` 且无坐标时直接跳过曲线构造、输出零先验并令 q=0，不能对零热图做 soft-argmax；不能换成真值坐标后与 E5 主结果直接比较。

## 6. E5-U 首版网络与 E5 扩展：Residual Prior Adapter

### 6.1 结构选择

首版按本节实现 E5-U：第四通道是有效性常数图，调制中的 `q_l` 实际取该图的局部值。网络保留原始投影特征路径，point/band 只提供软线索。直线 band 对弯曲样本的覆盖和净收益尚待第 5.2.1 节及后续对照验证；不能仅凭 E5-U 可运行就宣称 band 有益。经校准的 `q_pair` 与 E5 是后续扩展。

E5-U/E5 不实现完整 Attention U-Net decoder，也不替换 nnU-Net decoder。

网络由四部分组成：

```
projection ──> nnU-Net image encoder ──> image skips F_l ──────┐
                                                               │
point/band ──> lightweight prior encoder ──> prior skips P_l ──┤
gate map（首版 validity）──> stage-aligned q_l ────────────────┤
                                                               ▼
                                  ResidualPriorAdapter(F_l, P_l, q_l)
                                                               │
                                  modified skips F'_l
                                                               ▼
                                  original nnU-Net decoder
                                                               │
                                  segmentation logits
```

这样做的原因：

- Phase 1 已提供 Marker 预测，不需要在 Phase 2 再用 decoder gating signal 学习粗定位；
- 金属丝细节主要存在于高分辨率 skip；
- 保留原生 decoder 可降低 nnU-Net 改造风险；
- E2-U→E5-U、E2-Q→E5 分别检验同输入下完整先验分支和融合架构是否优于输入拼接；新增参数量也会影响结果，需要同复杂度及错配先验对照才能进一步归因。

### 6.2 融合层选择

根据 plans 的累计 stride 自动选择高分辨率 skip：

1. 主配置优先选择全分辨率和半分辨率、且实际传给 decoder 的两个 skip；金属丝像素宽度与显存开销决定是否保留全分辨率 stage；
2. 另设半分辨率和四分之一分辨率消融，验证更大的上下文是否更有利；
3. 对各向异性 stride，按 plans 中真实累计 stride 和 decoder skip 列表选择，不根据固定 stage 号推断；
4. 如果 plans 层数不足，选择 bottleneck 前可用的最高分辨率 skip；
5. 不允许写死 stage index；
6. 所选 stage、累计 stride 和 feature channels 必须写入 resolved config。

### 6.3 Prior encoder

prior encoder 使用与 image encoder 一致的 spatial stride，但规模较小：

- 每个 stage 使用一个卷积块；
- prior channels = max(8, round(image_channels × prior_feature_ratio))；
- prior_feature_ratio 默认 0.25，可配置；
- 只构建到最深的融合 stage，避免 DDP unused parameters；
- 不使用独立输出头，不预测 Marker 或 segmentation。

### 6.4 Residual prior modulation

对选定 stage l：

```
P_l' = Conv1x1(P_l)
A_l  = sigmoid(Conv1x1(concat(F_l, P_l')))
q_l  = clamp(resize(gate_map, size(F_l)), 0, 1)
M_l  = 1 + q_l * beta_l * (2*A_l - 1)
F_l' = F_l * M_l
```

其中：

- A_l 是空间注意力图，shape 为 B×C_l×H_l×W_l；
- q_l 是每个样本在该 stage 的局部门控图：E5-U 为有效性图，后续 E5 才为校准可靠度图。它经相同几何坐标系下重采样并广播到特征通道；prior encoder 只编码 point/band，不从局部峰值估计门控值；
- q_l=0 时该 adapter 严格恒等，包含 prior encoder bias 的情况也应成立；这不表示整个 E5 等同 E0，因为训练权重与架构不同；
- 直接以 `q_pair` 线性缩放残差是待验证的设计假设；`q_pair` 的统计含义是点对正确概率，不自动等于最优融合权重。E5-U 和按 q 分桶的消融用于检验它是否有利；
- beta_l 是每个 stage 一个可学习标量；
- beta_l = beta_max × sigmoid(beta_raw_l)；
- beta_max=0.5，beta_init=0.05；
- beta_raw_l 使用 logit(beta_init / beta_max) 初始化；当前参数应为 logit(0.1)，使实际 beta 初值等于 0.05；
- 末层 attention 卷积的权重和 bias 初始化为 0，使 A_l 初始精确为 0.5，M_l 初始精确为 1；
- 此精确 identity 初始化会让 prior encoder 在第一次 backward 时梯度为 0；末层 attention 权重应先获得梯度，更新后 prior encoder 才获得梯度。测试应验证两次优化步骤的梯度传播，不能要求第一次 backward 所有模块梯度都非零；
- M_l 初始为 1，但原图 encoder 的输入为单通道且权重从头训练，不能把整个 E5 网络称为与四通道 E2-Q 初始等价；
- 2A_l-1 允许在先验支持区域增强、在不可靠区域轻微抑制；
- 任何位置都不能被直接置零；
- `invalid` 的 point/band/q 全零时网络仍必须产生有限 logits，且各 adapter 输出与对应原图 skip 相等。

这个模块借鉴了 Attention U-Net 的“让条件信息影响空间选择”思想，但它是为“Marker 预测有误差、先验可能弯曲失配、细丝需要保留”而设计的残差调制，不是 Attention U-Net 原结构的直接复现。

## 7. 可选 E5-F：Spatial FiLM

只有 E5 基线稳定后才实现 E5-F，用于判断在 Residual Prior Adapter 之外增加 FiLM 是否有额外收益。

```
[gamma_delta_l, bias_delta_l] = Conv1x1(P_l)
F_l_film = F_l' + q_l * (gamma_delta_l * F_l' + bias_delta_l)
```

约束：

- gamma_delta 和 bias_delta 初始化为 0；
- FiLM 增量也受 q_l 控制，明确 `invalid` 时不得绕过 E5 的恒等融合约束；
- E5-F 增加一个调制模块，但保持数据、plans、decoder、loss 和训练预算一致；额外记录参数量和显存；
- 如果 E5-F 没有稳定优于 E5，则论文主方法不采用 FiLM；
- 不同时加入 Attention U-Net custom decoder，避免变量过多。

FiLM 论文只作为该可选模块的理论参考；本任务中使用的是 spatial prior conditioning，而不是论文的语言条件向量。

## 8. E6：Marker-constrained Deformable Centerline

### 8.1 启用条件

E6 不是默认网络。只有同时满足以下条件才启用：

- 已有合格 `q_pair`，且 E5 显著优于 E2-Q；
- E5 的主要错误集中在弯曲支架；
- 静态 band prior 在弯曲部分覆盖不足；
- 训练集有足够弯曲样本。

### 8.2 中心线预测

初始中心线：

```
C0(t) = M1 + t*(M2-M1)
```

网络从 image feature、marker_1_heatmap 和 marker_2_heatmap 预测 K=5 个控制点的法向偏移：

```
C(t_j) = C0(t_j) + delta_j*n
delta_0 = delta_K-1 = 0
```

两端固定，内部控制点可弯曲。使用分段线性或低阶样条插值得到曲线。偏移幅度需要相对 Marker 间距设置上限，并在端点重叠或距离过短时返回可解释的静态先验路径。

### 8.3 曲线先验

若全局坐标适配器验证通过，对曲线采样点使用平方距离的 soft-min：

```
d2_s(p) = ||p-C_s||^2
d2_soft(p,C) = max(0, -tau*log((1/S)*sum_s exp(-d2_s(p)/tau)))
H_curve(p) = exp(-d2_soft(p,C)/(2*sigma_curve^2))
```

实现 soft-min 时应使用数值稳定的 logsumexp。tau 的单位是像素平方，数值须按重采样后的像素尺度在验证集上确定。H_curve 替换静态 marker_band 进入 prior encoder。原始 marker_band 仍保留为 fallback 和 ablation。E6-base 必须使用相同的曲线采样及光栅化函数、令所有 delta=0；否则“零偏移等于 E6-base”的测试不成立。E6-base 的直线曲线先验与 E2 的解析有限线段 band 可能有离散化差异，不能强行要求逐像素相等。

### 8.4 E6 损失

```
L_smooth = mean((delta[j+1] - 2*delta[j] + delta[j-1])^2)
L_total = L_nnUNet + lambda_s * L_smooth
```

- lambda_s 初始为 0.01，搜索 0.001、0.01、0.05；
- 不使用中心线真值回归损失；
- 显式中心线分支只有在全局坐标到 patch/inference 的适配器通过集成测试后才可训练；
- 不把 E6 称为 DCNv2；
- DCNv2 只作为未来的 E6-D 对照，不进入第一版。

## 9. clDice 使用方式

第一阶段 clDice 只作为指标，报告：

- clDice；
- 断裂数量；
- 连通组件数量；
- 端部 Recall；
- 弯曲段 Recall。

二维投影里的金属丝交叉可能只是投影重叠，未必对应三维实体连接；clDice 与连通组件数只描述二维 mask 的形态，不能据此直接宣称三维支架拓扑得到保持。所有指标在全分辨率原投影上计算，并注明二值化阈值、骨架化方法与空 mask 处理。

如果 E5 仍然出现明显断裂，再增加 E5-T：

```
L_total = L_DiceCE + lambda_topo * (1 - soft_clDice(pred_prob, target))
```

soft-clDice 使用 soft-skeletonization，lambda_topo 从 0.1、0.3、0.5 中搜索。E5-T 必须作为独立消融，不能把拓扑损失直接混进 E5 主结果。

## 10. 实验矩阵

| 实验 | 输入 | 网络 | 目的 |
|---|---|---|---|
| E0 | raw | 原生 nnU-Net | 无先验 baseline |
| E1-P | raw + point | 原生 nnU-Net | Marker 点先验单独基线，检验无需假定支架沿直线 |
| E1 | raw + band | 原生 nnU-Net | 预测点生成的宽松带状先验 |
| E2 | raw + point + band | 原生 nnU-Net | 三通道拼接 baseline |
| E2-U | raw + point + band + validity | 原生 nnU-Net | 四通道无连续可靠度拼接 baseline |
| E2-Q | raw + point + band + q | 原生 nnU-Net | 同输入四通道拼接 baseline |
| E5-U | E2-U 输入；有效检测 `q_l=1` | Residual Prior Adapter | 同架构不使用连续可靠度 |
| E5 | raw + point + band + q | q-gated Residual Prior Adapter | 有校准 `q_pair` 时的主网络 |
| E5-F | E5 输入 | E5 + Spatial FiLM | 可选融合消融 |
| E5-T | E5 输入 | E5 + soft-clDice | 可选拓扑损失 |
| E6-base | projection + marker_1 + marker_2 + band + q | 同一曲线光栅化器、全部偏移为零 | E6 控制组 |
| E6 | E6-base 输入 | 可变形中心线 prior | 弯曲支架扩展 |

关键比较：

- E0 vs E1-P/E1/E2：分别判断端点线索、直线带及两者组合的价值；按弯曲程度检查 E1/E2 是否因直线假设而退化；
- E2-U vs E5-U：不使用连续可靠度时，匹配四通道输入的先验融合是否优于拼接；
- E2-U vs E2-Q：把有效性常数替换为校准可靠度是否改善原生拼接网络；
- E2-Q vs E5：相同四通道输入下完整双分支融合架构是否优于输入拼接；新增参数量仍需额外控制，不能仅归因于门控公式；
- E5-U vs E5：同一四通道架构下，校准可靠度是否优于常数门控；另按检测可靠度、定位误差和 `invalid` 分桶报告；
- E5 与相同网络、训练时使用错配 Marker 先验的负对照：检验真实 Marker 几何信息的价值；若未运行 E5，使用 E5-U 的同构负对照；
- E5 vs E5-F：增加 FiLM 是否值得其参数量和复杂度；
- E5 vs E5-T：拓扑损失是否减少断裂；
- E6-base vs E6：在相同五通道输入和同一栅格化函数下，可变形中心线是否改善弯曲样本。
- 真值 Marker oracle 与 Phase 1 预测 Marker 分开报告，前者估计理想定位条件下的参考表现，不保证数值上必然高于预测先验，也不参与主方法模型选择或主结论；oracle band 仍可能因支架弯曲而失配。

## 11. 代码结构

首版只创建数据转换、验证、prior encoder、residual adapter、网络、Trainer、raw-only 增强及对应的关键测试。FiLM、额外损失和曲线相关模块留到后续研究阶段。

```
phase_2/tools/marker_guided/
  build_prior_channels.py
  validate_dataset.py
  cli.py

phase_2/nnUNet/nnunetv2/training/nnUNetTrainer/variants/marker_guided/
  __init__.py
  marker_guided_encoder.py
  residual_prior_adapter.py
  marker_guided_network.py
  marker_guided_trainer.py
  marker_guided_transforms.py

phase_2/nnUNet/nnunetv2/tests/
  test_marker_prior_channels.py
  test_residual_prior_adapter.py
  test_marker_guided_transforms.py
  integration_tests/test_marker_guided_smoke.py
```

FiLM、额外损失、centerline/curve 及其测试在后续需要时再增加。不得直接修改 nnUNetTrainer.py、get_network_from_plans.py、默认 decoder、默认 loss 或 inference predictor。

## 12. 关键接口

```python
class ResidualPriorAdapter(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        prior_channels: int,
        beta_init: float = 0.05,
        beta_max: float = 0.5,
    ) -> None: ...

    def forward(
        self,
        image_features: Tensor,
        prior_features: Tensor,
        gate_map: Tensor,
    ) -> Tensor: ...


class MarkerGuidedPlainConvUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool,
        fusion_stage_indices: Sequence[int],
        prior_channels: int = 2,
        gate_channel: int = 3,
        prior_feature_ratio: float = 0.25,
        **plain_conv_unet_kwargs,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor | list[Tensor]: ...


class nnUNetTrainerMarkerGuidedFeatureFusion(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module: ...
```

`build_network_architecture` 必须保持上述静态方法签名：本地 nnU-Net v2.8.1 在训练时通过实例调用，在 checkpoint 推理时通过 Trainer 类直接调用，不能依赖 `self.dataset_json`。构建时检查 `num_input_channels==4` 与二维 PlainConvUNet plans；原图 encoder 只接收通道 0，prior encoder 接收 1–2，通道 3 作为 gate。`dataset.json` 的 `marker_guided` 元数据由数据转换/校验程序检查，不作为构建网络所需的实例状态。推理通过同一 Trainer 名称恢复相同结构。

## 13. 数据增强和训练约束

- 空间变换同时作用于 projection、prior channels 和 segmentation；
- 强度变换只作用于 projection；
- E1-P、E1、E2、E2-U、E2-Q、E5-U 也必须使用与 E5 相同的 raw-only 强度增强包装器，同时保留拼接基线的原生 nnU-Net 网络；直接使用原生多通道增强会随机改写先验和第四通道数值；空间变换必须同步作用于全部通道与标签，第四通道使用适合其标量含义的插值并限制在 `[0,1]`；
- 对所有实验记录数据集版本、Phase 1 `model_id`/`fold_id`/`prediction_source`/`calibration_id`、`q_tolerance_px`、先验宽度、Phase 2 fold、plans 和随机种子；验证/测试推理必须提供与训练相同顺序的先验通道；
- Phase 2 主实验使用 Phase 1 折外训练预测；应检查折外与验证/测试的坐标误差、低 `q` 比例和 `invalid` 比例，并在训练中保留失败病例。训练集可按折外误差分布补充坐标扰动或先验 dropout，但不能只用真值坐标加噪声替代真实折外预测；
- 错配 Marker 负对照在同一 split 内按病例固定置换两个 Marker 的整对坐标及误差半径，随后重新生成 point/band；保持原始 `q` 分布并明确标为故意错配的负对照，不把此时的 `q` 解释为错配坐标的正确概率，也不混入验证或测试标签信息；
- E5/E5-U 的 prior 全零、弱先验、小幅坐标扰动和 `q=0` 分支都必须有定义明确的处理并产生有限 logits；`q_pair=null` 只能进入 E0/E1-P/E1/E2/E2-U/E5-U 或明确标记的探索配置，不能静默转成四通道概率；
- prior branch、adapter 和 fusion stages 在每次 forward 都参与计算；
- 首版至少完成单进程 CPU 或 GPU 训练与滑窗推理 smoke test；具备多卡环境后再用 `find_unused_parameters=False` 完成 DDP smoke test；
- 第一版设置 nnUNet_compile=false，网络稳定后再测试 torch.compile；
- E0、E1-P、E1、E2、E2-U、E2-Q、E5-U、E5 使用相同 group-wise split、fold、seed、patch size、batch size、采样规则和训练预算；如显存导致 E5 无法保持这些设置，应统一降低所有组的设置并重新规划，不混合比较；
- 当前源码的网络构建来自 plans 中的 architecture class；第一版只支持 PlainConvUNet 2D，并在其它架构时明确报错。生成真实 plans 后必须检查所选 architecture、实际 strides 和 decoder skip 尺寸；
- 同一支架的三张投影不得跨 train/val/test。

## 14. 验收标准

### 14.1 单元测试

首版聚焦 E5-U 的输入、零先验、adapter、增强和 nnU-Net 契约；以下涉及 Q/E5、E6 及消融的检查在对应功能实现时执行。

- point/band prior 的坐标、峰值、有限线段距离和边界处理正确；
- E1-P 数据集只包含 raw/point 两通道；E1-P、E1、E2 使用相同坐标来源和先验生成版本，且不会读取当前样本 Mask 生成先验；
- Phase 1 JSON 解析正确处理 `schema_version`、同例三个不同视角、`group_id`、无序点、亚像素坐标、每点误差半径、`image_size=[H,W]`、`invalid` 零图、`q_pair=null` 拒绝 E2-Q/E5 生成但允许 E2-U/E5-U、非有限值及错配 `(case_id,view_id)`、`geometry_id` 和实际投影尺寸；
- 有效 `q_pair` 在 `[0,1]`，point/band 幅值不因 `q` 重复缩小；可靠度图在原图内取规定常数，重采样和增强后仍在 `[0,1]`；
- ResidualPriorAdapter 输出 shape 与 image feature 相同；
- beta 初始化后 adapter 接近 identity；
- beta 的有效初值必须等于 0.05，attention 末层零初始化时 adapter 输出须与输入逐元素相等；
- 全零 prior、扰动 prior 和正常 prior 都能产生有限 logits；identity 初始化后的第一次 backward 检查末层 attention 有梯度，第二次优化后再检查 prior encoder 有梯度；
- 在非零 attention 权重下分别测试 `q=0` 时各 adapter 逐元素恒等，以及 `q=1`、中间 `q` 对调制幅度的影响；`q=0` 不要求 E5 整网 logits 等于 E0；
- plans 改变时 fusion stages 自动改变；
- deep supervision 输出与原生 nnU-Net 尺度顺序一致；
- 强度增强不修改 point/band/q，空间增强同步变换所有空间数据；
- 用来源登记表审计每个 Phase 2 训练样本的 Phase 1 具体检测权重与校准器未见过该支架组；验证/测试模型未使用相应评估组拟合；缺失或未经校准的 `q` 不得进入 E5 主实验；
- E6 零偏移时 curve prior 与 E6-base 的零偏移光栅化结果一致；
- 对 E6 的 patch 裁剪、空间增强及滑窗推理，端点坐标映射必须可验证；若 Marker 落在 patch 外，不能从局部 heatmap 估计全局端点；
- E6 端点偏移严格为零；
- smooth loss 和 soft curve rasterization 可反向传播。

### 14.2 集成测试

使用小型合成二维数据验证：

1. E5-U 完成真实 nnU-Net 命令入口下的 plan、preprocess、至少一步 forward/backward、checkpoint 保存和恢复；smoke 可用仅缩短 epoch/迭代数的专用 Trainer 子类，正式训练仍使用完整训练预算；
2. E5-U inference 能由 checkpoint 正确解析自定义 Trainer，并完成滑窗推理与分割结果导出；
3. 自定义实现不修改原生 loss、decoder 和预测导出；
4. E5-U 的 prior/validity 不受强度增强影响，训练与推理的输入通道顺序相同；
5. 端到端以 Phase 1 结构化记录按 `(case_id,view_id) → nnunet_case_id` 映射生成四通道原图，随后完成 nnU-Net preprocessing、patch 训练及滑窗推理；对同一病例核对输入通道、先验配置版本、Phase 1 记录版本和 group-wise split。

E0/E1-P/E1/E2/E2-U/E2-Q/E5/E6 的集成与 DDP 检查随对应后续实验完成，不计入首版代码验收。

### 14.3 研究指标

二维指标：

- Dice、IoU、Precision、Recall、HD95；
- clDice；
- 端部 Recall；
- 1–2 像素容差 F1；
- 断裂数量和连通组件数量；
- 按弯曲程度和 Marker 误差分桶。
- 报告 band 宽度下的 Mask 覆盖率、背景覆盖比例及其在弯曲样本上的变化；明确 Mask 是否包含 Marker 本体。
- 有合格校准时按 Phase 1 `q_pair` 分桶；所有实验按 `pair_status` 分桶，报告检测失败病例的分割表现及其占比；同时报告不同宽度规则与 oracle Marker 的差距。

## 15. 实现顺序

1. 确认 Phase 1 按 Phase 2 fold 导出的折外/冻结预测包、二维投影与 Mask 清单、组划分和来源登记；明确 Mask 是否包含 Marker，但不以先验性能审计阻塞代码。
2. 实现 Phase 1 JSON 校验、按 `(case_id,view_id)` 精确关联、point/band/validity 生成及独立 E5-U TIFF 数据集导出；固定并保存先验配置，不读当前样本 Mask 生成先验。
3. 在 `phase_2/nnUNet/` 内新增只支持二维 PlainConvUNet plans 的自定义网络与 Trainer，接入 prior encoder、Residual Prior Adapter、raw-only 强度增强，保留 nnU-Net 原生 decoder、loss、训练和预测流程。
4. 用带三视角及 `invalid` 的合成病例完成 Phase 1 记录→E5-U 数据集→nnU-Net plan/preprocess→短训练→checkpoint 恢复→滑窗推理与导出；随后用真实数据配置核对 plans、尺寸、内存和来源审计。
5. 后续在相同划分和预算下运行 E0/E1-P/E1/E2/E2-U、直线先验适用性审计与 E2-U vs E5-U；只有独立审核的 `q_pair` 可用于 E2-Q/E5。
6. 若研究结果支持，再做错配 Marker、坐标扰动、fusion stage、FiLM、拓扑损失或 E6 弯曲建模；E6 先验证全局端点到 patch 和滑窗推理的正确映射。

## 15.1 首版实施门槛与后续研究门槛

- E5-U 首版的门槛是可解析的 Phase 1 预测包、正确的四通道数据和来源登记、二维 plans、可用的训练/推理环境，以及第 14.2 节的端到端 smoke test。没有真实病例时以合成数据证明软件链路可运行；真实数据效果与带宽选择仍待实测。
- E5-U 首版必须实现 `beta` 初始化、有效性图显式门控、raw-only 增强、输入通道一致性、检测失败零先验、折外来源审计和有限值检查。直线先验审计、错配先验对照与分割收益验证属于后续研究，不是首版实现门槛。
- E2-Q/E5 需要通过独立病例审核的 `q_pair`；所有正式性能比较须在真实 plans 上核对二维架构、patch 大小、显存预算和同等训练设置。
- E6：当前是研究假设，尚缺少可证明正确的全局端点坐标传递方式。若沿用随机 patch 和标准滑窗推理，不能仅靠局部 Marker 热图实现显式中心线。
- 如果适配器成本过高，可另行设计“局部可变形带状先验”作为新实验；这属于另一种方法，不能把它直接标成 Marker 端点约束中心线。

## 16. 论文表述边界

推荐表述：

若 E5 通过第 5.2.1 节的先验审计、完成实现且独立对照支持其收益，可表述为：“本文提出一种任务特定的 Marker-guided residual prior adapter，将三视角定位得到的 Marker 预测及经校准的点对可靠度转换为连续空间先验，并在 nnU-Net 高分辨率 encoder features 上进行可靠度门控的残差调制。”若仅点先验有效，论文主线应如实聚焦端点辅助分割，不宣称直线 band 或 E5 已被证明有效。只有 E6 真正实现且通过独立消融后，才补充弯曲支架的中心线表述。

论文中应将 Attention U-Net、FiLM、DCNv2 和 clDice 写成参考依据或可选对照，不声称本方法直接复现它们。nnU-Net 作为基础分割框架引用，Marker prior、Residual Prior Adapter 和中心线先验属于本课题的方法设计。
