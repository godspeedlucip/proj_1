# Vöth 2022 三投影方法复现实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 以 PyTorch 移植论文的二维 DTE、三通道逐视角反投影和三维 DTR；先保持原文 60° 间隔，再独立训练 0–60° 静态三投影配置。

**Architecture:** DTE 对三幅投影共享权重；每幅提取结果以自身几何反投影到独立三维通道；3D U-Net 从三通道体预测支架与导丝的联合二值分割。原文的一通道消融由另一个独立训练的 DTR 实现。

**Tech Stack:** Python、PyTorch、NumPy、SciPy、pytest；经验证的锥束前向/反投影后端；三维网格生成与体素化工具。

**Spec:** ../../../SPEC_静态三投影支架重建双基线.md，§3–6。原文：根目录 3D Reconstruction of Stents and Guidewires in an Anthropomorphic Phantom From Three X-Ray Projections3D Reconstruction of Stents and Guidewires in an Anthropomorphic Phantom From Three X-Ray Projections.pdf，§2.1–2.5、图 1、表 1。

## Global Constraints

- 原文为三视角同时获取、相邻 60°；0/30/60 三视角是 B-adapted-3view，不是作者报告的原条件。
- 两个网络都使用 PyTorch，但论文原程序使用 TensorFlow 2.5.0；精确检查通道轴、BN、上采样、padding、初始化与损失归约，不声称框架随机轨迹一致。
- B-DTR 原文用三幅**分别反投影**的通道训练，且在训练投影中模拟 DTE 假阳性/假阴性。不能用 A 的稀疏 FDK 单体代替。
- B-DTE 原文 12,000 幅支架 + 12,000 幅导丝模拟投影与九名患者 2,751 幅不含器械的临床 CBCT 投影在线组合；B-DTR 另有 40,000 个三维场景。两组数量不得混写。
- 主模型输出支架与导丝的联合二值分割。仅含支架的测试场景才可直接计算支架专属指标。
- 当前目录无论文源数据/权重；本机有 16 GB RTX 4070 Ti SUPER，但 PyTorch 2.11.0 为 CPU 版。保持原文 batch 24/8 与网络通道数不变；先做 CUDA 与显存预检，若资源不足需记录适配条件或使用更大显存设备，不能暗中减层、减通道或减批量。
- 当前目录不是 Git 仓库；任务结束时执行验证并保存可审查的配置/结果，不写不可执行的 git commit 步骤。

## Review Focus

1. 视角顺序与反投影通道错位：任务 2、5 的三彩色/点目标测试必须检出。
2. 将三个反投影体求和后送入三通道 DTR：任务 5 必须拒绝；求和只允许任务 7 的单通道消融。
3. DTE 连续线积分目标误用成二值掩膜：任务 3、4 的数据值域测试必须检出。
4. 40,000 个场景按投影而非模型分割造成泄漏：任务 1 的 manifest 测试必须检出。
5. 论文骨架距离实现的单位或方向错误：任务 8 的合成偏移测试必须检出。

## 复现资料清单与可执行边界

| 资料 | 原文用途 | 当前状态与门槛 |
| --- | --- | --- |
| 九名患者的 2,751 幅无器械 CBCT 投影 | DTE 在线组合的背景 | 文件夹未提供；替代背景必须作为新的数据设置报告。 |
| 12,000 幅导丝及 12,000 幅支架模拟投影 | DTE 训练/验证 | 文件夹未提供；可按论文参数重建模拟流程，但不能声称使用原始样本。 |
| 40,000 个三维场景/网格和真值 | DTR 70/30 训练/验证 | 文件夹未提供；需依 §2.2 重新生成，并记录所有原文未给的随机范围。 |
| 三种体模的精细角度扫描 | 三投影回选与 FDK 真值 | 文件夹未提供；无此数据时只做模拟评估。 |
| CUDA 版 PyTorch 与足够显存 | 原文 384²、128³ 和 batch 24/8 | 当前 PyTorch 为 CPU 版；16 GB GPU 容量须先做完整形状的内存预检。 |

## File Map

- src/stent_repro/voth2022/config.py：原文/适配几何、训练参数与来源标记。
- src/stent_repro/voth2022/geometry.py：三视角投影、逐视角反投影、体素网格。
- src/stent_repro/voth2022/simulation.py：支架/导丝网格、场景与器械投影。
- src/stent_repro/voth2022/datasets.py：DTE 在线组合与 DTR 三通道块。
- src/stent_repro/voth2022/models.py：B_DTE、B_DTR 和一通道消融模型。
- src/stent_repro/voth2022/losses.py：MAE、带 Laplace 平滑的 soft Dice。
- src/stent_repro/voth2022/train.py：DTE 与 DTR 的独立训练、验证、检查点。
- src/stent_repro/voth2022/infer.py：完整 DTE→逐视角反投影→DTR 推理。
- src/stent_repro/voth2022/evaluate.py：体模真值、Dice 和双向骨架距离。
- configs/voth2022/original_3view_60step.yaml、configs/voth2022/adapted_3view_0_30_60.yaml、configs/voth2022/one_channel_ablation.yaml。
- tests/voth2022/：每项任务对应测试；outputs/voth2022/：运行结果。

---

### Task 1: 配置、数据来源与显存门槛

**Interfaces:** RunConfig.load(path) -> RunConfig；SceneManifest.validate()；split_scenes(scene_ids, model_ids, ratios, seed) -> manifest；preflight_memory(config) -> resource report。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_config.py 检查原文三视角间隔 60°、适配角度 0/30/60；源探距 1100 mm、源到等中心 600 mm、探测器 1024² 和 0.3 mm、体素网格 256³/0.6 mm；按三维场景/模型划分 70/30，阻止跨集泄漏；模拟资源不足时 preflight_memory 必须报告失败而不能自动缩小模型或 batch。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_config.py -q。
- [ ] **Step 3: 实现配置与 provenance。** 区分论文明确值、移植决定、目标场景适配；保留原文 batch 24/8、epoch 200/150，资源预检读取实际 CUDA/显存状态并报告能否训练，不能自动改模型。
- [ ] **Step 4: 验证通过。** 同一 pytest 通过；运行 python -c "import torch; print(torch.cuda.is_available())" 并把结果写入环境记录。当前预计为 False，后续完整训练前须在可用 CUDA 环境重新检查。

### Task 2: 三维场景、几何与正/反投影

**Interfaces:** generate_scene(seed) -> mesh + labels + metadata；forward_project(scene, views) -> [3,H,W]；backproject_each(projections, views, grid) -> [3,D,Hv,Wv]。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_geometry.py 检查三视角投影的中心位置、1 mm 位移导致的像素移动量、将仅在一个视角出现的标记反投影时只激活对应通道；测试通道置换不被静默接受。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_geometry.py -q。
- [ ] **Step 3: 实现论文场景。** 一个支架加一或两根导丝；导丝为随机游走样条周围的曲圆柱；支架沿中心轴堆叠杆件环并沿样条弯曲，随机化原文列出的直径、段高、波纹数、杆厚、样式、段数及偏移，主动脉支架直径 10–30 mm。生成 40,000 场景的 manifest，训练/验证 70/30。
- [ ] **Step 4: 实现锥束几何与三通道反投影。** 先在小体模上做 CPU 参考，再使用显存受控的生产后端；每幅投影独立写入一个通道，输入输出都记录物理网格。原文未公开的插值/归一化参数写入实现记录。
- [ ] **Step 5: 验证通过。** 同一 pytest 通过；输出一个含支架和导丝的网格、三个投影及三个反投影体的对应切片用于人工核对。

### Task 3: B-DTE 数据管道与模型

**Interfaces:** make_dte_pair(tool_projection, patient_projection, rng) -> (input,target)；B_DTE.forward([B,1,H,W]) -> [B,1,H,W]。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_dte.py 检查 12,000 支架 + 12,000 导丝投影、2,751 患者 CBCT 投影的 manifest 类型，384×384 在线配对；连续线积分 target；七级通道 64/128/256/512/1024/2048/4096、每级两次 3×3 Conv-BN-ReLU、max pool、最近邻上采样；用固定小张量核对 PyTorch 的通道顺序、上采样位置及所选 BN 参数语义。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_dte.py -q。
- [ ] **Step 3: 实现 DTE。** 加入原文提到的模糊、噪声、散射增强；使用共享权重处理任意视角。最终激活、精确跳连与输出层在论文中未明确，必须记录为移植决定；不能写作原文明确参数。
- [ ] **Step 4: 验证通过。** 同一测试通过；小尺寸输入可前向/反向传播，384×384 输入输出形状一致；检查峰值内存而非悄悄减少通道。

### Task 4: B-DTE 训练

**Interfaces:** train_dte(config, manifest) -> checkpoint + validation log。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_train_dte.py 检查 MAE、Adam 学习率 1e-5/β=(0.9,0.999)、batch 24、200 epoch、每 epoch 16,000/4,000 训练/验证对；检查小样本可降低损失且训练/验证来源不交叉。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_train_dte.py -q。
- [ ] **Step 3: 实现训练循环与日志。** 保存权重、优化器、数据生成种子、增强参数、论文未知的输出激活选择及显存峰值；完整训练只在预检通过且数据齐备时启动。
- [ ] **Step 4: 验证通过。** 同一测试通过；在小数据上完成一轮 smoke run。若只能做 smoke run，结果标为结构验证。

### Task 5: B-DTR 训练样本与 3D 网络

**Interfaces:** make_dtr_pair(scene, views, grid, error_rng) -> ([3,128,128,128],[1,128,128,128])；B_DTR.forward(x) -> [B,1,D,H,W]。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_dtr.py 检查 256³/0.6 mm 网格中保存 128³ 块、z 翻转与绕 z 轴 90° 整数倍旋转、目标为支架+导丝联合体素化；对正投影添加受控假阳性和假阴性，再逐视角反投影；三通道在网络入口仍独立，且 PyTorch 的 [B,C,D,H,W] 与论文 TensorFlow 张量轴对应正确。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_dtr.py -q。
- [ ] **Step 3: 实现 3D U-Net。** 五级通道 16/32/64/128/256；每级两次 3×3×3 Conv-BN-ReLU、2³ max pool、最近邻上采样再接 3×3×3 卷积。输出单通道器械概率体；原文未写明最终激活和精确跳连，记录选择与层表。
- [ ] **Step 4: 验证通过。** 同一测试通过；先用较小体块验证梯度，再在资源允许时验证原文 128³ 输入，记录参数量和峰值显存。

### Task 6: B-DTR 损失与训练

**Interfaces:** soft_dice_B(pred,target,smoothing=1) -> scalar；train_dtr(config, manifest, input_channels=3) -> checkpoint。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_train_dtr.py 检查 Dice 在空/非空体上有限且反向可微；Adam 学习率 1e-4/β=(0.9,0.999)、batch 8、150 epoch、每 epoch 16,000/4,000 对；检查训练集增强不会写入验证标签。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_train_dtr.py -q。
- [ ] **Step 3: 实现训练。** 原文只说 soft Dice + Laplace 平滑，未给精确公式/归约方式；把选用的式子和归约轴写入配置与报告。保留三通道输入、DTE 错误增强、原文 batch 与网络结构；显存不足时停止 strict run 并提供单独标注的资源适配方案。
- [ ] **Step 4: 验证通过。** 同一测试与小体块过拟合测试通过；真实原文尺寸训练只在 CUDA/数据门槛满足后验收。

### Task 7: 完整推理与一通道消融

**Interfaces:** reconstruct(projections, views, dte, dtr, grid) -> probability volume + binary volume；train_one_channel_ablation(config, manifest) -> separate checkpoint。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_pipeline.py 检查完整顺序 DTE→三次独立反投影→三通道 DTR；一通道消融必须先求和再训练一个新 DTR，不允许把三通道权重直接加载到一通道模型。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_pipeline.py -q。
- [ ] **Step 3: 实现推理与消融。** 保存三幅 DTE 结果、三通道反投影切片、概率体和二值体；阈值若原文未给，应显式配置并在论文复现报告中注明。保留一通道消融为独立实验。
- [ ] **Step 4: 验证通过。** 同一测试与一例模拟场景端到端检查通过；三通道/一通道运行配置互不覆盖。

### Task 8: 体模验证与静态 0–60° 适配

**Interfaces:** skeleton_distances(pred, truth, spacing_mm) -> (pred_to_gt_mm, gt_to_pred_mm)；其中每个方向都是该方向骨架点到另一骨架最近点的欧氏距离均值，先将体素索引乘以 spacing_mm；run_profile(profile, manifest) -> results.json。

- [ ] **Step 1: 写失败测试。** tests/voth2022/test_evaluate.py 检查预测骨架整体平移 1 个 0.6 mm 体素时两向距离均按 mm 报告；漏检/多检的方向不同；adapted profile 只用 0/30/60 并重新生成 DTR 训练样本；纯支架指标不混入导丝场景。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/voth2022/test_evaluate.py -q。
- [ ] **Step 3: 实现论文验证协议。** 若有密集角度体模扫描，回选三幅 60° 间隔投影，密集扫描 FDK 阈值化作真值；重现三类场景及双向骨架平均距离。实现与原文所用 3D Lee 骨架化同等的处理，并记录当前库/版本；论文测量表数值只作参照，不作为无原始数据时的通过条件。
- [ ] **Step 4: 实现目标适配协议。** 三幅静态投影在 0–60° 内，默认 0/30/60；保持 B 网络和三通道输入定义，只修改视角配置并从头生成/训练相应 DTR 数据。使用与 A 相同的测试支架、几何与三维真值，在物理网格对齐后评估。
- [ ] **Step 5: 最终验证。** 运行 python -m pytest tests/voth2022 -q；生成报告，列出论文原条件、PyTorch 移植决定、0–60° 适配、来源缺口、完整训练是否完成、Dice 和双向骨架距离。缺源数据或 CUDA 环境时只声明已验证的结构/流程，不宣称达到论文结果。
