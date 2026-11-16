# Eulig 2021 四投影方法复现实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 复现论文的器械重建分支：二维 DTE、四视角稀疏 FDK、连续切片二维 DTR；再单独训练和评测 0–60° 静态三投影版本。

**Architecture:** DTE 对每幅患者投影回归器械线积分；已知几何的 FDK 将多视角结果变为一个稀疏体；DTR 读取相邻 C 张切片并预测中心切片。原论文配置与本课题适配配置分离，网络结构保持相同。

**Tech Stack:** Python、PyTorch、NumPy、SciPy、pytest；三维模型生成使用 Blender Python API；投影/FDK 使用经过解析体模验证的几何后端。

**Spec:** ../../../SPEC_静态三投影支架重建双基线.md，§2、§4–6。原文：根目录 Deep learning‐based reconstruction of interventional tools and devices from four X‐ray.pdf，§2.1–2.6、§3.2–3.5、表 1–2。

## Global Constraints

- 原文主配置是四幅投影、相邻 45°、随机起始角；三幅投影 0–60° 只能标为 A-adapted-3view。
- PyTorch 实现不得把 DTR 换成 3D 卷积，也不得把不同投影分成多个 DTR 输入通道。
- A-DTR 原文用理想、无噪声、仅器械投影训练；不要套用 2022 年论文的 DTE 假阳性/假阴性增强。
- A 原文有导丝、支架、线圈三类模型，DTR 输出其联合二值分割；纯支架评价只在纯支架测试场景计算。
- 原文的数据可用性声明为“研究数据不共享”；本文件夹无源数据、模型权重或八种基础支架模型。必须分别记录“结构/流程复现”和“论文数值复现”的状态。
- 当前本机 PyTorch 2.11.0 是 CPU 版，虽然有 16 GB RTX 4070 Ti SUPER。完整训练前必须检查 CUDA 版 PyTorch 和峰值显存；不足时停止严格复现并记录实际改动，不能暗中缩小网络或 batch。
- 当前目录不是 Git 仓库；此方案不包含不可执行的提交命令。若日后建立仓库，再按任务提交。

## Review Focus

1. 同一 STL 的不同视角或变形进入不同数据集：任务 1 的 split 测试必须阻止。
2. FDK 的毫米坐标、探测器 u/v 方向或角度顺序出错：任务 3 的点/圆柱解析体模测试必须检出。
3. DTE 目标误写成二值掩膜或把输入统计量用于目标：任务 2、5 的值域与配对测试必须检出。
4. DTR 训练时误接 DTE 的噪声预测：任务 6 的来源追踪测试必须检出。
5. 三视角结果被当成论文原条件成绩：任务 8 的 profile 与报告测试必须检出。

## 复现资料清单与可执行边界

| 资料 | 原文用途 | 当前状态与门槛 |
| --- | --- | --- |
| 八种基础支架模型及三类器械 STL/生成参数 | 共 600 个三维模型与训练真值 | 文件夹未提供；可先实现生成器接口，严格数值复现须取得原模型或如实标注替代模型。 |
| 六次患者 CBCT 投影 | DTE 四次训练、两次验证的背景 | 文件夹未提供；没有这批数据时只能用另行取得的数据做替代实验。 |
| Zeego 能谱、探测器与校准几何 | 多色能谱前向投影及 FDK | 论文给出主要尺寸，未给出全部能谱/数值实现；缺失部分须记录实现决定。 |
| 体模有/无器械扫描和密集角度真值 | 论文测量验证 | 文件夹未提供；不能用模拟测试结果冒充测量验证。 |
| CUDA 版 PyTorch 与足够显存 | 按原尺寸训练 | 当前 PyTorch 是 CPU 版；16 GB GPU 是否足够需在完整模型上预检。 |

## File Map

- src/stent_repro/eulig2021/config.py：原文/适配配置、几何参数、数据来源标记。
- src/stent_repro/eulig2021/models.py：A_DTE 与 A_DTR 网络。
- src/stent_repro/eulig2021/simulation.py：三类 STL 模型、器械前向投影与 DTE 配对样本。
- src/stent_repro/eulig2021/geometry.py：Zeego 锥束几何、前向投影、稀疏 FDK。
- src/stent_repro/eulig2021/datasets.py：DTE 图像块、DTR C 切片块与数据划分。
- src/stent_repro/eulig2021/losses.py：MAE 与论文式 soft Dice。
- src/stent_repro/eulig2021/train.py：两个网络各自训练、验证、检查点。
- src/stent_repro/eulig2021/infer.py：四投影/三投影推理与切片拼接。
- src/stent_repro/eulig2021/evaluate.py：论文指标与统一物理网格评价。
- configs/eulig2021/original_4view.yaml、configs/eulig2021/adapted_3view_0_30_60.yaml：互不覆盖的实验配置。
- tests/eulig2021/：每项任务对应测试；outputs/eulig2021/：运行后产物，不作为源文件提交。

---

### Task 1: 配置、来源和划分

**Interfaces:** RunConfig.load(path) -> RunConfig；RunConfig.validate()；split_models(model_ids, seed, ratios) -> split manifest。配置包含 profile、angles_deg、SDD/SID、探测器、体素网格、模型类别、数据来源与随机种子。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_config.py 检查原文配置四角度等间隔 45°、适配配置恰为 0/30/60；测试同一 model_id 不会同时出现在训练与验证/测试；测试 adapted 结果标签不会显示为 original；缺 CUDA 或显存时 strict 训练预检应报失败而不能静默调整论文设置。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_config.py -q；预期因接口不存在而失败。
- [ ] **Step 3: 实现并写两份配置。** 原文 Zeego：源探距 1200 mm、源到等中心 785 mm、探测器 1240×960、像素间距 0.308 mm；DTR 体素 0.1 mm。缺少的谱、滤波核、基础模型来源写入 provenance，不以猜测值冒充论文参数。
- [ ] **Step 4: 验证通过。** 同一 pytest 命令通过，配置导出后可再读且字段无丢失。

### Task 2: 原文器械模型与 DTE 样本生成

**Interfaces:** generate_model(spec, seed) -> STL + metadata；render_tool_projection(stl, geometry, spectrum) -> 连续线积分；make_dte_pair(tool_projection, patient_projection, rng) -> (input, target)。源数据的文件清单由 manifest 管理。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_simulation.py 检查三类各 200 个模型的配置、八种支架基础模板的引用、支架直径 4.8–6.3 mm 与杆件厚度 0.4–0.7 mm；每模型 17 个相邻 10° 的 DTE 投影；target 为连续投影值，不是阈值图。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_simulation.py -q。
- [ ] **Step 3: 实现生成器与来源门槛。** Blender 输出 STL；记录导丝 NURBS、线圈与支架参数。若原始八种支架模板、患者 CBCT 或能谱不可得，strict profile 必须明确报缺失；可用 surrogate profile 继续结构验证，但要把替代模型/谱写入 manifest。
- [ ] **Step 4: 实现 DTE 在线混合。** 患者背景按原文 4 次扫描训练、2 次扫描验证；384×384 块、拒绝无/低衰减块、x–y 翻转、随管电流变化的 Poisson 噪声、0–1.4 px 高斯模糊；不额外加入原文未模拟的造影剂或散射。
- [ ] **Step 5: 验证通过。** 运行同一 pytest 命令，另输出一组输入/目标可视化以检查配准和灰度尺度。

### Task 3: 锥束前向投影和稀疏 FDK

**Interfaces:** ConeBeamGeometry(source, detector_origin, u_axis, v_axis, pixel_size_mm)；forward_project(volume, grid, views) -> [V,H,W]；fdk(tool_projections, views, grid) -> [D,Hv,Wv]。几何参数使用 mm，角度只在输入配置以度数表示。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_geometry.py 检查等中心点投影落在探测器中心、已知直径圆柱的线积分尺度、旋转后图像方向、四视角 FDK 输出中心位置以及投影顺序置换的一致性。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_geometry.py -q。
- [ ] **Step 3: 实现可分块的已知几何算子。** 先提供小体模 CPU 参考，后接可在 CUDA 上批处理的后端；FDK 的探测器权重、滤波、角度权重与插值必须在配置中可追溯。论文未给的数值细节标为实现决定，不改变 A 的“合成一个 FDK 体”信息路径。
- [ ] **Step 4: 验证通过。** 解析体模测试通过；保存几何配置、后端版本与重建可视化。完整模拟前向投影还需通过物理尺度检查。

### Task 4: A-DTE 网络

**Interfaces:** A_DTE.forward(x: [B,1,H,W]) -> [B,1,H,W]；create_dte() -> A_DTE。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_models.py 检查 384×384 形状、四级编码/解码、首级 64 通道、两次 3×3 Conv-BN-Dropout2d(0.2)-ReLU、max pool、最近邻上采样、拼接跳连、非负输出与反向传播。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_models.py::test_dte -q。
- [ ] **Step 3: 实现网络。** 卷积 He 初始化，BN scale=1/bias=0；论文没有给出的瓶颈/输出连接在模型结构记录中标为实现决定，并导出网络层表与参数量。
- [ ] **Step 4: 验证通过。** 同一测试通过；对小张量完成梯度检查。

### Task 5: A-DTE 训练

**Interfaces:** train_dte(config, manifest) -> checkpoint + metrics；normalize_projection(x, running_stats) -> x_norm。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_train_dte.py 检查在线运行均值/方差只从训练输入更新、推理统计量冻结、MAE 在连续标签上计算、Adam 学习率 1e-4 与 β=(0.9,0.999)、batch 6、一个微型样本可过拟合。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_train_dte.py -q。
- [ ] **Step 3: 实现训练循环。** 记录输入统计量、模型权重、优化器、随机种子、数据 manifest；总 epoch 原文未报告，运行时指定并写入报告，不能声称该值来自论文。
- [ ] **Step 4: 验证通过。** 同一测试通过；strict 数据缺失时清楚失败，surrogate 数据只生成结构/流程结果。

### Task 6: A-DTR 数据与网络

**Interfaces:** make_dtr_sample(stl, four_views, grid, C) -> ([C,224,224], [1,224,224])；A_DTR.forward(x) -> [B,1,224,224]；soft_dice_A(pred,target) -> scalar。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_dtr.py 检查 DTR 数据来自理想无噪器械投影，四视角间隔 45°、随机起始角、FDK 单体输入；C=1/3/5 的中心层标签正确；sigmoid 输出；Dice 按原文式 (4) 对 batch 与像素统一求和、分子分母各加 1。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_dtr.py -q。
- [ ] **Step 3: 实现数据管道和模型。** 体素化 STL 作真值；稀疏体按零均值单位方差归一；DTR 沿用 A-DTE 的二维 U-Net 结构，输入通道为 C、输出中心层。C=1/3/5 建独立模型配置。
- [ ] **Step 4: 验证通过。** 同一测试通过；三种 C 的形状和梯度均正确。

### Task 7: A-DTR 训练与完整推理

**Interfaces:** train_dtr(config, manifest, C) -> checkpoint；reconstruct(projections, views, dte, dtr, grid) -> probability volume + binary volume。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_pipeline.py 检查 Adam 学习率 1e-4、batch 16、DTE→FDK→DTR 顺序、推理逐层滑窗/边界填充、0.5 阈值与三维位置保持。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_pipeline.py -q。
- [ ] **Step 3: 实现训练和推理。** 保留 C=1/3/5 消融；实际测量数据须经 DTE，不能在报告中把理想投影输入成绩混作完整管线成绩。
- [ ] **Step 4: 验证通过。** 同一测试及一例小体模端到端检查通过；保存每阶段中间结果。

### Task 8: 静态 0–60° 适配与评价

**Interfaces:** evaluate_volume(pred, truth, grid) -> Dice/precision/recall/100 μm 容差 precision 和 recall/双向骨架距离；run_profile(profile, manifest) -> results.json。

- [ ] **Step 1: 写失败测试。** tests/eulig2021/test_adaptation.py 检查 adapted profile 只用 0/30/60 三视角、DTR 训练输入重新生成、训练/测试模型不泄漏、结果 JSON 标注 adapted；测试不同体素网格先按 mm 对齐再评分，并用已知 0.1 mm 位移核对原文的 100 μm 容差 precision/recall。
- [ ] **Step 2: 验证失败。** 运行 python -m pytest tests/eulig2021/test_adaptation.py -q。
- [ ] **Step 3: 实现配置与评价。** A 网络和理想投影训练规则不变；只改变目标实验所需视角和相应 DTR 数据。按论文 §3.3 同时报告常规 Dice/precision/recall 与以真值膨胀/腐蚀得到的 100 μm 容差 precision/recall；结果表并列显示原文条件与适配条件，不把 A 原文表 2 的 Dice 0.794 当作本课题指标。
- [ ] **Step 4: 最终验证。** 运行 python -m pytest tests/eulig2021 -q；输出配置、样本数量、训练耗时、峰值显存、体素尺度、指标与 strict/surrogate/adapted 状态。无真实训练数据时只交付可执行网络与结构/流程验证，不填写论文精度复现成功。
