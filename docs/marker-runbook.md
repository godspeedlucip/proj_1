# Phase 1 Marker Pipeline Runbook

Run commands from the repository root. Python dependencies are PyTorch, NumPy, SciPy, tifffile, and pytest.

```powershell
Set-Content -Path tmp/phase1-config.json -Encoding utf8 -Value '{"epochs":1,"batch_size":12,"widths":[4,8,16,32],"seed":7,"top_k":10,"top_l":5}'
python -m phase_1.marker_pipeline.cli make-synthetic tmp/phase1-smoke --seed 7
python -m phase_1.marker_pipeline.cli validate tmp/phase1-smoke/manifest.json --split fold_0
python -m phase_1.marker_pipeline.cli train tmp/phase1-smoke/manifest.json tmp/phase1-model --config tmp/phase1-config.json
python -m phase_1.marker_pipeline.cli predict tmp/phase1-smoke/manifest.json tmp/phase1-model/checkpoint.pt tmp/phase1-predictions
python -m phase_1.marker_pipeline.cli evaluate tmp/phase1-smoke/manifest.json tmp/phase1-predictions --tau 3
python -m phase_1.marker_pipeline.cli export-fold tmp/phase1-smoke/manifest.json tmp/phase1-fold --split fold_0 --config tmp/phase1-config.json
```

The config JSON may specify `epochs`, `batch_size`, `learning_rate`, `widths`, `sigma`, `alpha`, `lambda_fn`, `seed`, `top_k`, `nms_radius`, `top_l`, `lambda_reproj`, `scale_px`, `min_parallax_deg` (defaults to a 1° fallback threshold), `uncertain_margin`, `fallback_error_radius_px`, and `q_tolerance_px`. The default architecture is the planned 32/64/128/256 U-Net; smoke runs should use narrow widths and one epoch. `train` saves `checkpoint.pt`, `config.json`, and `training_groups.json`. `export-fold` writes one v1 JSON per case, `source_registry.json`, `package.json`, and `case_view_index.csv`; its OOF models exclude the predicted group, and held-out models fit only the fold's train groups.

The simple `train → predict → evaluate` sequence above trains on all synthetic cases, so its metrics are only a CLI smoke check. For group-held-out synthetic evaluation, run `evaluate` on `tmp/phase1-fold/predictions` produced by `export-fold`. Synthetic results verify software behavior only and are not evidence of real projection localization performance. The default error radius is a conservative configured placeholder until retained real fold predictions support an estimate; no calibrated `q_pair` is emitted.

## Fixed cone-beam three-view joint model

Set `"model_type":"joint_epipolar_v1"` in the same JSON config and use the same `train`, `predict`, `evaluate`, and `export-fold` commands. The manifest still contains three views per case, with `projection_path`, `angle_deg`, `P`, and training `marker_xy` in original image `(x, y)` coordinates. The three angles must be consecutive multiples of 30° modulo 360°, such as `0/30/60` or `330/0/30`; manifest order may vary. Within one checkpoint, detector size, `geometry_id`, and relative cone-beam projection geometry must stay fixed. If images are resampled, update every `P` to the new pixel grid. `mask_path` is never used as a Phase 1 model input.

Example config for a real manifest with the required geometry:

```json
{
  "model_type": "joint_epipolar_v1",
  "epochs": 20,
  "batch_size": 1,
  "widths": [32, 64, 128, 256],
  "seed": 7,
  "sigma": 2.0,
  "aux_loss_weight": 0.25,
  "support_stride": 4,
  "support_samples": 32,
  "support_chunk_rows": 32,
  "support_temperature": 0.1,
  "geometry_tolerance_px": 0.25,
  "top_k": 10,
  "top_l": 5,
  "phase2_fold_id": "fold_0"
}
```

`batch_size` counts three-view cases in this mode. The network jointly refines three heatmaps with cross-view epipolar support, then the existing peak decoder and three-view geometry search return two unordered marker coordinates per view. The checkpoint stores the fixed relative geometry. A case with incompatible geometry or detector dimensions raises an error. At 1024×1024, start with `batch_size=1`; reduce `support_chunk_rows` if memory is tight. No production memory or throughput guarantee has been measured.

The default config retains the legacy single-view model, and `predict` recognizes either checkpoint format. `export-fold/package.json` contains the v1 case records and Phase 2 source registry; `source_registry.json` additionally records architecture and a relative-geometry digest. The main `PairAcc@tau` counts invalid views as failures; `conditional_PairAcc@tau` reports accuracy only among valid predicted views.

For the joint checkpoint, `predict` writes `diagnostics/{case_id}.npz` (initial and refined heatmaps) and `diagnostics/{case_id}.json` (candidate coordinates, scores, and top-two predictions without geometric search). `export-fold` writes the same sidecars next to `predictions/`. `evaluate` uses them, when present, to report initial/refined Top-K pair recall, refined Top-L pair recall, and `ablation_no_geometry_PairAcc@tau`; these metrics are omitted as `null` for legacy predictions. The diagnostics do not alter the Phase 2 v1 records.

The `make-synthetic` command above produces 15°-spaced orthographic-like images for the legacy smoke and is not a joint-mode dataset. For joint-mode software verification, use a manifest with analytic cone-beam `P` and 30° spacing, as in `phase_1/tests/test_joint_pipeline.py`. For a performance comparison, freeze a real or credible DRR group-wise split and all decoding/geometry-search settings, run `export-fold` once with each architecture, then compare `evaluate` results and per-case latency on the same cases. Keep all projections of one physical scaffold in the same `group_id`. Synthetic Gaussian spots establish only that the software contract works; they do not establish a marker-localization improvement on CT projections.
