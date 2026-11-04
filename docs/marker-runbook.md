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
