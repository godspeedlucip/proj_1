# Phase 3：nnU-Net + Swin UNETR 融合 + TubularEnhancement + 投影一致性 Spec

- 日期：2026-11-11
- 状态：设计草案，供审阅；官方源码已下载，自定义功能尚未实现。
- 目标目录：`D:/desktop/课题1/code/phase_3/nnunet`
- 用户已明确：迁入 Swin UNETR 改动、TubularEnhancement 网络模块、投影一致性损失；不将管状增强误称为独立损失，不加入 DTM/GSL。
- 下文“建议/首版决定”是本 spec 的设计选择，不代表用户已批准全部实现细节。

## 1. 目标与迁移边界

以官方 nnU-Net v2 为训练、数据管理、模型保存及滑窗推理框架，用其原生可配置 CNN U-Net 替换当前手写 PlainUNet3D，并迁入以下改动：

1. 当前 phase_3 的冻结预训练 Swin UNETR 编码器、瓶颈融合、解码器引导的多尺度 skip 门控。
2. 用户指定参考工程中的 TubularEnhancement：四路膨胀卷积对最高分辨率 logits 做残差细化。
3. 基础分割损失加可微投影一致性损失；分项日志、开关和消融必须完整。
4. 保留现有三投影 FDK 输入、支架杆件标签、来源分组、原网格预测和组级评价契约。

首版任务仍为单通道三维 FDK → 三维二值支架分割。Swin 是辅助编码器，共享 nnU-Net decoder，不启用另一套 Swin UNETR decoder。TubularEnhancement 同时参与训练与推理。

范围不含重写 FDK、Marker、各向异性注意力、双网络联合训练、DTM/GSL、clDice、新的真实二维投影数据采集。保留 `phase_3/fdk_unet`、`phase_2/nnUNet` 和参考工程原样，作为历史实现及回归依据。

## 2. 上游版本与源码交付

“最新”按查询日的官方最新稳定发布解释，不使用持续变动的 master HEAD。

| 项目 | 固定值 |
|---|---|
| 仓库 | https://github.com/MIC-DKFZ/nnUNet |
| 发布版本 | v2.8.1；官方 tag 与 PyPI 最新版本一致 |
| 发布日期 | 2026-07-01 |
| commit | `468cf803df9b267150ae2b6c0c59b8ac84f16227` |
| 下载方式 | 官方 commit ZIP，解压至本目录；无嵌套 `.git` |
| ZIP SHA-256 | `e1a0d1df341c4ca19adf18a135816a30808821096c9bc308a480d4d19bdaba55` |
| 许可证 | Apache-2.0，保留原 LICENSE 和文件声明 |

证据：[官方 tags](https://github.com/MIC-DKFZ/nnUNet/tags)、[PyPI 2.8.1](https://pypi.org/project/nnunetv2/2.8.1/)、[固定版本 pyproject](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/pyproject.toml)。当前 master 同样声明 2.8.1，但包含发布后的改动，不能把两者视为相同源码。

已下载源码；本轮仅新增本 spec 与来源记录，不安装包，不启动训练，不修改上游 Python。后续不自动升级版本。正式环境采用独立虚拟环境并 editable install 本目录，启动时断言 `nnunetv2.__file__` 位于此目录，避免误导入其他两个本地同名包。

## 3. 已核实的源实现

| 来源 | 实际行为 | 迁移处理 |
|---|---|---|
| `../fdk_unet/plain_unet.py` | 五级 BN/ReLU，通道 16/32/64/128/256，单 logit | 作为旧对照；CNN 改用 nnU-Net 原生组件 |
| `../fdk_unet/swin_prior.py` | MONAI 1.3.0；feature_size=48；冻结 Swin；执行至 layers3 | 保留实现与预训练契约 |
| `../fdk_unet/hybrid_model.py`、`fusion.py` | /16 瓶颈、/8 和 /4 skip；零残差初始化；decoder 条件门控 | 适配 nnU-Net encoder/decoder 的对应特征 |
| `../fdk_unet/losses.py` | 逐样本前景 soft Dice+BCE；smooth=1；valid_mask | 精确保留基础损失的数值定义 |
| 参考工程 `nnunetv2/network_modules/tubular_enhancement.py` | dilation=1/2/3/5 的可训练残差模块 | 迁入最高分辨率 logits 后 |
| 参考工程 `nnunetv2/network_modules/projection_consistency.py` | 预测概率与 3D 标签分别投影，然后 MSE | 保留监督思路，修正下述几何与掩膜问题 |
| 参考工程 `nnunetv2/training/loss/composite_loss.py` | 分割损失可深监督，投影只取最高分辨率 | 采用组合方式，不整体复制旧 Trainer |

参考工程根目录为 `D:/desktop/课题1/nnUNet`。它的 Trainer 使用旧版 build_network_architecture 参数形式；本项目须按 2.8.1 的新接口适配。

### 3.1 必须纠正的参考实现差异

- **名称**：TubularEnhancement 是网络模块，没有独立的 `L_tubular`。总损失不会凭空增加此项。
- **监督来源**：代码接收 3D one-hot 标签，实际计算 `MSE(P(pred), P(label))`；文档提到“原始投影图”不代表代码已读取二维实测图像。
- **投影方向**：旧实现绕张量 D 轴旋转 H/W，然后仍沿 D 求均值。各角度相当于同一深度投影的平面旋转，不能代表独立方向射线。
- **ignore**：旧组合器先将 logits 乘 valid，再 softmax；无效位置的零 logits 会变成非零概率，不能正确排除 padding/ignore。

已做只读最小诊断：两个仅在 D 方向位置不同的点体，在旧实现 0/30/60 度投影上的最大差异为 0；全无效位置的零 logits 和零 target 在旧投影损失中仍产生 0.25。这里只确认缺陷，尚未实现修复。新实现采用版本名 `label_parallel_v1`，不得宣称与旧投影算子数值完全一致。

## 4. 方案选择

| 方案 | 得失 | 本 spec 决定 |
|---|---|---|
| 旧 PlainUNet3D 直接挂进 Trainer | 迁移少，但主干仍是原手写 U-Net | 不作为目标实现 |
| nnU-Net 原生 CNN + Swin 辅助编码器 + 输出增强 + 组合损失 | 使用 nnU-Net 主干与生命周期，保留已有研究结构 | 推荐；下文按此设计 |
| 完整 Swin UNETR 替代 CNN encoder/decoder | 改变共享 decoder 和融合研究问题 | 首版不采用 |

官方支持通过自定义 Trainer 的 `build_network_architecture` 与 `_build_loss` 扩展；必须同时适配训练和 predictor 的网络重建路径。[版本化扩展文档](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/documentation/extending_nnunet.md)

## 5. 数据契约与预处理

### 5.1 输入和划分

- 外部输入保持单通道 FDK NIfTI；原标签只有 `0=背景、1=stent_struts`。必须校验真实语义，不能将 vessel 命名直接视为支架。
- 保留 manifest 的 case_id、group_id、source_model_ids、patient_id、angles_deg、projection_source 和上游训练来源审计。
- 图像/标签 shape 与 affine 匹配，体素有限、空间单位明确；首版继续要求各轴 0.2 mm，容差 1e-4 mm。禁止静默重采样细杆。
- 固定现有 NIfTI `(i,j,k)` → tensor `(D,H,W)=(k,j,i)` 轴顺序；不隐式 canonical reorientation。沿用 slope/intercept、单位换算、qform/sform 冲突校验，保存原 affine 与头信息；采用匹配该契约的 reader/writer，plans 的后续 transpose 固定为 identity，逆转换按同一约定导出。容差内略有不同的三轴 spacing 一律保持原网格，不触发重采样；投影坐标按规范化 0.2 mm 等距 patch 定义，并记录这个容差内近似。
- 复用现有按来源组 train/validation/test 划分。转换为 nnU-Net `splits_final.json` 的 train/val case ID，测试组不进入 imagesTr/labelsTr 或 fingerprint。
- 每个原始 fold 使用独立 dataset 名称及 raw/preprocessed/results 根；其 nnU-Net 内部仅使用 fold 0，记录 `source_fold → dataset → nnunet_fold=0`。外层 seed 也使用独立 results 根。实际 dataset ID 可由实现者选取，必须检查冲突。
- 不允许 nnU-Net 默认随机文件级交叉验证替代来源组划分，也不允许 `fold=all` 绕过它。

### 5.2 单源图像、两个内部视图

外部仍提供一份 FDK。转换器在每个 fold 内生成两个同网格内部通道：

- `_0000`：整个原体逐病例 z-score，与旧 CNN 分支一致。
- `_0001`：只用该 fold 训练组拟合 0.5/99.5 分位范围后裁剪映射到 [0,1]，与旧 Swin 分支一致。

训练组统计按现有 `prior_stats.py` 的分组采样方法计算，写入统计文件与配置摘要。验证、测试和外部预测只读取已保存统计量。不得对 z-score 数据再次估计原始强度分位数，不得跨 fold 共用 Swin 缓存。

所有受控消融共用同一双通道数据：CNN 只读取通道 0，Swin 只读取通道 1；关闭 Swin 时网络忽略通道 1。因此官方网络工厂接收的外部 num_input_channels=2，构建 CNN encoder 时显式使用 1。此适配规则必须写入网络配置并在推理时恢复。

`FDKPreprocessor` 继承 nnU-Net 的预处理/存储协议：两通道使用 NoNormalization，关闭默认 nonzero crop，记录全图 bbox；相同 spacing 采用恒等网格处理，仍正确记录 transpose、shape、spacing、bbox 等逆变换属性。不能只替换归一化而继续让默认裁剪改变原体统计含义。

首版数据增强使用同步裁剪、padding 和三轴翻转，不启用会分别改变两个强度视图的独立 gamma/noise/brightness。所有消融使用相同规则。影像、标签与有效体素掩膜必须同网格变换。

### 5.3 无效体素

nnU-Net 数据集内部标签声明 `background=0, stent_struts=1, ignore=2`，输出头仍为两类。原始数据不使用 2；padding/无效区域由适配器生成。必须将 loader/transform 的 padding=-1 转为 ignore=2，覆盖默认转为背景的行为；在生成各尺度 target 前完成此处理。监督、在线指标、全体验证都排除 ignore。全部无效的样本须拒绝或重新采样，并记录原因。

## 6. 网络结构

```text
FDK → 两种同网格强度视图
  ├─ z-score → nnU-Net CNN encoder → F0,F1,F2,F3,F4
  └─ train-percentile → frozen Swin → G1(/4),G2(/8),G3(/16)
                    G3 → bottleneck fusion(F4)
                    G2 → decoder-guided skip fusion(F3)
                    G1 → decoder-guided skip fusion(F2)
          → 一套 nnU-Net decoder → 最高分辨率 2-channel logits
          → TubularEnhancement → final logits → softmax → 前景概率/二值输出
          └─ 辅助分辨率 logits → 分割深监督
```

### 6.1 nnU-Net 主干与 plans

首版采用 `dynamic_network_architectures` 的 PlainConvUNet encoder/decoder 组件，InstanceNorm3d、LeakyReLU 及上采样按其配置实现。不可继续用旧 PlainUNet3D 冒充 nnU-Net 主干。

为了迁入已有 /16 瓶颈而不额外执行 Swin layers4，首版自定义 `FDKnnUNetPlanner` 明确约束为五级、各向同性累计步幅 1/2/4/8/16，初始通道建议 `[32,64,128,256,320]`，每级卷积数沿用所固定原生配置，patch 默认 128³，batch 默认 2。这些参数全部写入 plans，不硬编码在 fusion 模块里。

这是受约束的 nnU-Net 3d_fullres 配置，不是未经修改的全自动 nnU-Net 基线。单独的官方默认自动规划实验可作为补充基线，不能和受控消融混作同一训练条件。首版不接受 /16 不是最深层或缺失 /4、/8 的 plans；不得把 /16 skip 默称为 bottleneck。未来支持六级/各向异性须新增显式特征映射和验证。

fusion 根据 plans 获取实际 CNN 通道，检查累计步幅、feature shape 与物理网格。采用薄层可配置 decoder 适配，不修改已安装的第三方 DNA 包，不依赖全局 forward hook。零融合系数下，输出必须与相同权重原生 CNN 对齐。

### 6.2 Swin 与融合

- MONAI 1.3.0；embed_dim=48，depths=(2,2,2,2)，heads=(3,6,12,24)，window=(7,7,7)，patch=(2,2,2)，merging，use_v2=False。
- G1/G2/G3 通道 96/192/384；prefix 执行和 `proj_out` 归一化方式保持现有实现。
- 沿用已有 SSL 权重审计、SHA-256、100% 可训练主干张量匹配要求；无权重不得默默降级为随机 prior。
- 主实验 prior 冻结且 eval，optimizer 排除其参数；其输出到可训练 Align/Gate 的梯度仍保留。
- 融合定义 `F' = F + tanh(alpha) * sigmoid(Gate(...)) * Align(G)`；alpha=0，Align 为 1×1×1。
- /16 gate 读取 F 与 Align(G)；/8 和 /4 gate 同时读取当前 decoder 上采样特征。禁止用任意插值掩盖尺度错误。
- 保留 random-prior、H00/H01/H10/H11 的结构开关与执行深度语义，作为补充消融配置。
- patch 每边至少 32 且为 32 倍数；原始体通过 padding/滑窗支持任意尺寸。

### 6.3 TubularEnhancement

参考实现四路 3×3×3 卷积，dilation=1/2/3/5，hidden=max(2*num_classes,8)，各分支 InstanceNorm+LeakyReLU；拼接后 1×1 融合并残差相加。最后一层按参考代码 std=1e-4、bias=0 初始化，属于近似恒等，不能声明完全恒等。

只处理最高分辨率两通道 logits。深监督时处理 `outputs[0]`，其余输出不变；推理 tensor 输出走相同模块。它是网络的一部分，不能在训练后省略，也不能误当作阈值后二值形态学操作。

### 6.4 深监督

首版建议启用 nnU-Net 深监督；五级网络输出 1、1/2、1/4、1/8 四个尺度。按固定上游单 GPU 规则，最低尺度权重为 0，其余归一化为 4/7、2/7、1/7。另提供 `deep_supervision=false` 复现对照。

网络须暴露有效的 deep_supervision 开关，使训练返回列表，预测返回最高分辨率 tensor。输出不附带 softmax/sigmoid。启用深监督是相对旧 phase_3 的显式训练变更，不能宣称两者训练条件完全相同。

## 7. 损失函数规格

### 7.1 基础分割损失

两类 logits 为 `z_bg,z_fg`，定义 `z=z_fg-z_bg`，则 `sigmoid(z)=softmax(logits)[fg]`。以该差值调用旧版逐样本前景 Dice+BCE 定义，既适配 nnU-Net 的 label manager 又保留旧损失语义。

对样本 b，有效掩膜 m、二值标签 y、概率 p：

- `BCE_b = sum(m * BCEWithLogits(z,y)) / sum(m)`。
- `DiceLoss_b = 1 - (2*sum(m*p*y)+1)/(sum(m*p)+sum(m*y)+1)`。
- `L_seg = mean_b(BCE_b + DiceLoss_b)`。

各项 float32 累积；深监督时逐尺度计算该损失并按 6.4 加权。空前景仍有限，全部无效拒绝。单 logit 到双 logit 是接口迁移；旧 checkpoint 不可作为新主干的直接续训权重。

### 7.2 投影监督含义

首版为**标签派生的局部平行束投影正则**：最高分辨率预测概率与同一训练 patch 的 3D 标签经过相同投影算子。参考代码本身无需原始二维投影，本版也不新增此输入要求。

默认使用规范化 patch 坐标中的合成方向 `[0,30,60]`，`coordinate_frame=canonical_patch`、`target_source=label`。这些是正则化方向，不冒称病例真实三个采集角度。manifest 的真实 angles_deg 继续用于来源审计。增强后在共同 patch 上计算投影；由于该正则明确使用当前 patch 坐标，翻转不需要假装维护实测 detector 配准。

如果后续要求与原始实测投影一致，必须另立 geometry 版本：包含旋转轴、源/探测器位置、距离、像素间距、外参、体素到世界变换、裁剪位置和投影目标语义；锥束 FDK 不能由本版平行束占据率模型代替。也不能把裁剪 patch 的投影与完整 detector 图直接做 MSE。

### 7.3 可微算子 `label_parallel_v1`

- 张量约定 `[B,1,D,H,W]`；绕 D 轴旋转 H/W 平面，**沿 W 求和**，探测器保留 D/H。旋转轴与射线方向必须不同。
- 在物理等距坐标建立采样网格，使用 `grid_sample`、`align_corners=False`；H/W 正方形旋转画布扩展到覆盖原截面对角线，避免角落裁切。不能对非方形张量直接套归一化坐标旋转而造成尺度失真。
- 先计算前景概率，再乘 m；分别投影 `m*p`、`m*y`、`m`。绝不通过将 logits 乘 m 来屏蔽无效位置。
- 对角度 a 和探测器像素 u，令 `r=sum_W R_a(m)`，`q_p=sum_W R_a(m*p)/max(r,eps)`，`q_y=sum_W R_a(m*y)/max(r,eps)`。
- 仅对 `r>eps` 的有效射线计算 `(q_p-q_y)^2`；先对每病例每视角有效像素平均，再对视角及病例平均。eps=1e-6；无有效射线的样本报错。有效区域之外的 padding 不改变分母。
- softmax、插值、积分和 MSE 均在禁用 autocast 的 float32 中完成；target 路径无需梯度，prediction 路径不能 detach 或阈值化。
- 缓存键包含角度、画布/原体 shape、spacing、device、dtype，禁止跨 shape 复用错误网格；采用有界缓存。

只计算前景通道，仅作用于经过 TubularEnhancement 后的 final logits；不对每个深监督头重复增加投影项。算子对细杆仍存在插值误差，需用解析 phantom 验证，而不是宣称恢复真实成像物理。

### 7.4 总损失与日志

`L_total = L_seg_DS + lambda_proj * L_proj`。

`lambda_proj=0.1` 作为参考工程一致的初始实验值，可配置，非负且有限；通过 validation 选择时固定调参预算和记录，不查看 test。权重为 0 时不执行投影算子。关闭深监督时 `L_seg_DS=L_seg`。

记录 total、seg、最高尺度 dice/bce、各尺度 seg、projection_raw、projection_weighted、有效射线数量、融合 beta/gate/残差比、AMP、显存及运行时间。不得输出虚构的 tubular_loss；管状增强通过分割/投影梯度学习。非有限 loss 或梯度须停止更新并报告 batch。

## 8. Trainer、训练与模型恢复

- 新增 `nnUNetTrainerPhase3`，使用上游 2.8.1 接口：`build_network_architecture(plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision)`。
- 主要扩展 `_build_loss`、数据加载/增强、train_step/validation_step 的分项日志、完整体验证及 checkpoint；保留 nnU-Net 训练生命周期、optimizer/scaler 机制及 predictor。
- 网络构造只依赖随模型保存的 plans 配置。SSL 初始加载发生在新训练初始化路径；恢复及预测直接使用 checkpoint 中完整 prior 权重，不重新访问外部权重文件。
- 首版 CUDA 单卡；CPU 仅 smoke，显式设置 `nnUNet_compile=false`，不在首版引入编译图、动态诊断与第三方 Swin 的兼容问题。DDP、2D、cascade、regions 明确拒绝，不允许未经验证的功能路径看似可用。
- 初始实验预算建议 10,000 optimizer steps，250 steps 验证一次，即 40 个研究 epoch。使用 nnU-Net SGD/PolyLR 方案，初值 lr=0.01、momentum=0.99、weight_decay=3e-5、nesterov=true、PolyLR power=0.9，精确配置和总步数写入运行记录；所有新消融一致。不得暗用上游默认 1000 epochs 替代此预算。
- 同一 seed 共享 CNN 初始化，fusion/tubular/prior 使用命名随机流；每来源组等概率、组内 case 等概率的采样适配原生 loader，前景采样概率 0.33。实际 sample 序列和 RNG 策略可审计。
- 每次完整体验证使用相同滑窗设置，在原网格计算 group-mean Dice，以此选择 `checkpoint_best.pth`；nnU-Net patch pseudo-Dice 仅为训练诊断。
- checkpoint 保存模型、optimizer/scaler/scheduler、step、RNG、采样状态、plans/config/数据划分 digest、统计量与 prior 来源。首版验证 epoch 边界续训；多进程异步预取暂不开启，不承诺 GPU 位级一致。
- 推理期通过保存的 trainer class + plans 重建网络，包括 Swin/Tubular/DS 开关。旧 phase_3 `best.pt` 仍由旧入口读取，不悄悄转换成新 checkpoint。

## 9. 推理与评价

使用 nnU-Net 滑窗及网格导出流程；默认 overlap=0.5、Gaussian 权重融合、TTA 关闭、无连通域后处理，显式记录与旧推理实现的差异。最终前景概率与阈值规则固定；两类 argmax 的平局采用背景，导出 mask 使用 `p_fg>0.5`，避免两条路径不一致。

预测分为 external 与 experimental：external 不传 splits，允许无标签；experimental 必须提供 splits/source-fold/split，匹配 checkpoint 的 fold、划分 digest、case ID 和输入图像 hash。生成的 `prediction_run.json` 绑定 checkpoint、输入图像、已有标签和输出文件 SHA-256。正式 evaluate 只接受 experimental 记录，重新核查 split 与文件 hash，拒绝把 external 预测当作独立测试结果。模型目录自带强度统计量，external 预测不要求访问原训练数据。

输出 `case_prob.nii.gz`、`case_mask.nii.gz`、运行 provenance 和 JSON/CSV 指标。恢复完整原始 shape/affine/spacing，概率有限且在 [0,1]。单张无标签 FDK 可预测，需随模型保存的训练强度统计量；它不能生成准确率结论。

复用 phase_3 的原网格 Dice、precision、recall、假阳性体积、26 连通分量、对称平均骨架距离和空标签约定。先病例内、再组内、最后独立来源组等权汇总；与旧 evaluator 对同一预测做数值对齐。

## 10. 配置、文件与命令契约

上游源码保存在本目录根，扩展集中到以下位置，实施时可调整内部拆分但不能改变公共语义：

```text
phase_3/nnunet/
  PHASE3_MIGRATION_SPEC.md
  UPSTREAM_SOURCE.json
  pyproject.toml, LICENSE, documentation/, ...          # 官方源码
  nnunetv2/phase3/
    config.py, data_conversion.py, prior_stats.py
    swin_prior.py, fusion.py, network.py, tubular.py
    projection.py, losses.py, evaluation.py, cli.py
  nnunetv2/experiment_planning/experiment_planners/fdk_planner.py
  nnunetv2/preprocessing/preprocessors/fdk_preprocessor.py
  nnunetv2/training/nnUNetTrainer/phase3/nnUNetTrainerPhase3.py
  configs/phase3/*.json
  tests/phase3/
  env/requirements-phase3.lock.txt
  PHASE3_README.md
```

运行数据、虚拟环境、预训练权重及 checkpoint 放在源码外的指定目录；禁止提交大体数据。核心模块迁入新包后不依赖 `D:/desktop/课题1/nnUNet` 的绝对路径，也不通过 sys.path 导入旧包；仅测试可将旧实现作为参照。

以下为**拟实现命令**，当前不能执行自定义入口：

```text
python -m nnunetv2.phase3.cli validate-data --manifest ... --splits ...
python -m nnunetv2.phase3.cli prepare --manifest ... --splits ... --source-fold 0 --config configs/phase3/full.json --output ...
python -m nnunetv2.phase3.cli train --prepared ... --seed 2026 --prior-artifact ...
python -m nnunetv2.phase3.cli predict --model-folder ... --manifest ... --output ...
python -m nnunetv2.phase3.cli predict --model-folder ... --manifest ... --splits ... --source-fold 0 --split test --output ...
python -m nnunetv2.phase3.cli evaluate --manifest ... --splits ... --source-fold 0 --split test --predictions ...
```

wrapper 应分派到原生 Trainer/nnUNetPredictor，不另写一套独立训练循环。应同时提供等价原生命令例子：`nnUNetv2_train <dataset> 3d_fullres 0 -tr nnUNetTrainerPhase3 -p <resolved_plans>`。resolved plans 绑定唯一 config digest，避免不同开关覆盖同一结果路径。

配置必须显式包含：upstream/version、CNN plans、swin/fusion、tubular.enabled、segmentation.loss、deep_supervision、projection.enabled/weight/angles/coordinate_frame/target_source/operator_version、normalization、fold/seed、预算、checkpoint selection 与 inference。未知键或冲突配置报错。

环境首选从已验证的 Python 3.10 / torch 2.3.1+cu118 / numpy 1.26.4 / MONAI 1.3.0 组合开始解析上游依赖，安装到新环境；该组合与 nnU-Net 2.8.1 全依赖的兼容性尚未验收。必须执行依赖检查及 CPU/CUDA/MONAI smoke 后生成完整锁文件。不能把旧融合环境的通过记录当作新环境通过，不能升级 MONAI 后仍声称遵守 1.3.0 权重契约。

## 11. 消融设计

八组主消融采用同一受约束 nnU-Net 主干、数据、初始化、训练预算、DS、模型选择及推理协议：

| ID | Swin 融合 S | Tubular T | Projection P |
|---|---:|---:|---:|
| B | 0 | 0 | 0 |
| S | 1 | 0 | 0 |
| T | 0 | 1 | 0 |
| P | 0 | 0 | 1 |
| ST | 1 | 1 | 0 |
| SP | 1 | 0 | 1 |
| TP | 0 | 1 | 1 |
| STP | 1 | 1 | 1 |

S 默认冻结 SSL prior。补充随机 prior 与 H00/H01/H10/H11 分析仍保留；官方全自动 nnU-Net 和旧 PlainUNet3D 作为外部基线单独报告训练差异，不混入单因素因果比较。交付包括这些配置，不代表本轮或实现验收必须完成全部正式训练。不得预设提升幅度或宣称新模型必然优于基线。

## 12. 验收标准

1. **来源/隔离**：上游每文件 hash 对下载归档一致；本地包导入路径正确；外部参考工程及旧 phase_3 不变。
2. **数据**：异常标签/affine/spacing、泄漏分组、跨 fold 统计量被拒绝；准备/预测双视图一致；任意原尺寸如 97×113×65 往返保持网格。
3. **CNN/Swin**：128³ 前向/反向通过；32 倍数 patch 与错误 plans 验证明确；alpha=0 时匹配相同 CNN；SSL prefix 与 MONAI 官方数值匹配，冻结 prior 无梯度、fusion 随初始化阶段获得梯度。
4. **Tubular**：开关正确、近似恒等初始化、只有最高 DS 输出变化；训练及重建推理模块一致，梯度有限。
5. **基础损失**：两 logit 差值与旧 foreground_dice_bce 在相同输入/target/mask 下 loss 和梯度对应一致；空前景和 ignore 正确；禁用投影时总损失严格退化到基础项。
6. **投影几何**：0/90 度解析投影、离轴点、不同深度点、非对称细杆 phantom 检验真实方向变化；非方形画布不扭曲、不裁角；与独立小体素射线求和参考在声明容差内一致。
7. **投影掩膜/梯度**：在保持同一 detector 坐标网格、物理有效支持及投影中心的条件下，额外无效 padding/ignore 不产生惩罚；不承诺任意改变 patch shape、中心或奇偶性后离散插值逐值不变。改变被 mask 的 logits 不影响 loss；全无效明确失败；输出有限且可反传，float64 小型差分检查与 float32/AMP 结果相符。相关数学算子为自定义方法，不能只测试输出 shape。
8. **集成**：真实 Trainer 跑一个短 CPU smoke 和一个 CUDA optimizer step；不同 S/T/P 配置可构造；三类损失/模块路径按开关执行；最高尺度之外不重复投影。
9. **恢复**：epoch 边界恢复 optimizer/step/采样及配置；删除或移走外部 SSL 文件后仍能重建预测；训练 DS 开启、推理关闭的切换可用。
10. **端到端**：合成/具备配对标签的小数据上完成 prepare→train→predict→evaluate；原网格、分组评价、输出 manifest 可核查。随后执行现有 phase_3 回归及必要跨阶段导入回归。
11. **显存**：现有环境记录为 RTX 4070 Ti SUPER 16 GiB，实施时重新核实实际设备。实测主方案 128³、batch 2、AMP 的峰值显存和步时；若 OOM，先提出所有消融统一的替代 plans，禁止仅为某组静默改 batch/patch。

软件验收不等于真实支架精度验收。现有 phase_3 文档指出真实标签及独立来源资料仍需完善；正式实验须以实际数据审计结果为准。

## 13. 实施顺序与交付物

1. 固定并验证上游来源，独立环境安装，记录 imports 和依赖。
2. 实现分组数据转换、双视图、plans/preprocessor 与无增强 CNN 基线闭环。
3. 迁入 Swin prior/fusion，再接入 TubularEnhancement，完成单项数值验收。
4. 实现并验证修正后的投影算子，接入组合损失和分项日志。
5. 完成 Trainer、恢复、原生 predictor、组级模型选择与评价。
6. 交付八组主配置及补充消融、锁文件、运行手册、测试结果和实际显存记录。

完成本 spec 后再进入实现；本轮不把拟实现接口描述成现有功能。首版几何算子、受约束五级 plans、深监督开启、固定局部投影方向及 0.1 投影权重是需要随 spec 审阅的主要设计决定。

## 14. 本轮核验记录

- 上游归档逐文件核对：281 个文件，0 个 hash 差异。
- 独立可实施性审阅完成；已补齐正式/外部预测 provenance、NIfTI 轴序、容差内不重采样与投影 padding 验收边界。
- 仅完成 spec、源码下载及旧投影缺陷诊断；新模型、修正投影、依赖组合与训练效果均尚未实现或验收。
