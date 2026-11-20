# 从原始投影构建 Phase 2 数据

`build.py` 使用 Phase 1 清单中的几何真值 marker 坐标，读取清单指向的原始投影与标签 TIFF，生成 Phase 2 E5-U 的四通道 nnU-Net 数据。四通道依次为投影原图、point、band 和 validity；这里没有运行 Phase 1 检测器。生成器复用 `tools.marker_guided.validate_dataset` 的交接校验和 `tools.marker_guided.build_prior_channels` 的先验光栅化函数。输出 TIFF 使用 zlib 压缩。

当前 `Dataset503_MarkerGuided_fold_0` 来自 `proj_1/output/data/phase1_200/phase1/manifest.json`：194 个物理病例、1940 个三视角样本、5820 条视角记录，按清单分为训练 5040、验证 720、测试 60。测试仅包含 `topcow_ct_027` 与 `stent5`、`w_stent` 的组合。旧 Dataset501 的原始包已不在当前数据目录；其预处理缓存、历史实验记录和权重档案仍保留。

```text
data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/
  raw/Dataset503_MarkerGuided_fold_0/
    imagesTr/           # train + val，_0000 原图，_0001 point，_0002 band，_0003 validity
    labelsTr/
    imagesTs/           # test，仅四通道图像
    labelsTs/           # 60 张测试真值 mask；不参与训练
    test_reference_manifest.json  # 测试 ID、标签相对路径和来源
    ground_truth_fold_0_v2/
      manifest.json     # 原始投影、标签、组划分和视角映射
      package.json      # 三视角 marker 真值和来源登记
    dataset.json
    splits_final.json
    case_mapping.json
    source_registry.json
    ground_truth_provenance.json
```

`raw/` 是 nnU-Net 查找 Dataset ID 503 的父目录；同级的 `prepared/` 用于
预处理，`results/` 当前只保存 Dataset501 的权重档案。目录说明见
[`Dataset503 README`](../../../data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/README.md)。

## 运行

如需重新生成，请写入新的空暂存目录，不要覆盖现有 Dataset503。生成器要求
交接目录与 raw 输出目录分开；生成后还需按上表归档交接清单和测试标签：

```bash
/data1/liuyang/conda/envs/nnunet/bin/python \
  /data1/liuyang/stent/proj_1/segment/phase2/get_data/build.py \
  --phase1-manifest /data1/liuyang/stent/proj_1/output/data/phase1_200/phase1/manifest.json \
  --handoff-dir /data1/liuyang/stent/proj_1/data/training/nnunet/phase2/Dataset503_rebuild_stage/ground_truth_fold_0_v2 \
  --output-dir /data1/liuyang/stent/proj_1/data/training/nnunet/phase2/Dataset503_rebuild_stage/raw/Dataset503_MarkerGuided_fold_0 \
  --dataset-id 503 --fold-id fold_0
```

默认的 Phase 1 清单、`--regenerate-source` 使用的 stacked/stent_models 目录以及
Dataset ID 已更新为当前 `phase1_200`/503；默认输出是独立的
`Dataset503_rebuild_stage/`，不会覆盖正式 Dataset503。`--metadata-only` 只生成交接
JSON；正式写图像时需使用新的空 `--handoff-dir`。生成器不会自动创建
`labelsTs/` 或 `test_reference_manifest.json`；它们是生成后归档的测试标签。
调整先验参数可用 `--sigma-0`、`--k-sigma`、`--w-0`、`--k-w`、
`--endpoint-taper-px`、`--d-min`，每套参数应使用新的 dataset ID 和输出目录。

本数据的 marker 来自仿真支架的几何真值，`error_radius_px=[0,0]`、`q_pair=null`，有效性通道恒为 1。`package.json` 的 `prediction_source=group_oof/held_out` 是为了符合当前 Phase 2 校验器字段要求；`model_id`、`oracle_source` 和 `ground_truth_provenance.json` 明确表明没有训练或运行 Phase 1 检测器。它适合验证 Phase 2 流程和做真值先验实验，不可当作 Phase 1 检测结果或折外性能评估。

如需让 nnU-Net 发现当前数据，设置
`nnUNet_raw=/data1/liuyang/stent/proj_1/data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/raw`、
`nnUNet_preprocessed=/data1/liuyang/stent/proj_1/data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/prepared` 和
`nnUNet_results=/data1/liuyang/stent/proj_1/data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/results`。
随后按 `../README.md` 用 Dataset ID `503` 执行 planning、preprocessing 和短训练。
正式训练前核对预处理目录中的 `splits_final.json` 与当前划分一致。Dataset501
的旧训练配置不能直接用于新数据集。

## 校验

```bash
cd /data1/liuyang/stent/proj_1
PYTHONPATH=segment/phase2 /data1/liuyang/conda/envs/nnunet/bin/python \
  -m unittest get_data.test_build -v

PYTHONPATH=segment/phase2 /data1/liuyang/conda/envs/nnunet/bin/python \
  -m tools.marker_guided.cli validate \
  --manifest data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/raw/Dataset503_MarkerGuided_fold_0/ground_truth_fold_0_v2/manifest.json \
  --package data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/raw/Dataset503_MarkerGuided_fold_0/ground_truth_fold_0_v2/package.json \
  --phase2-fold-id fold_0
```

输出目录存在且非空时脚本拒绝覆盖。转换意外中断后，应先检查和移走不完整目录，再重跑。
