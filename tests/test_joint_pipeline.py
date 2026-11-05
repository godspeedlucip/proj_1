import json
import numpy as np
import pytest
import torch
import tifffile

from phase_1.marker_pipeline.schema import Case, View
from phase_1.marker_pipeline.train import train_detector, predict_joint_heatmaps
from phase_1.marker_pipeline.cli import main
from phase_1.marker_pipeline.export import _model_id, validate_prediction_record, evaluate
from phase_1.tests.test_epipolar_support import cone_matrices, project
from phase_2.tools.marker_guided.validate_dataset import validate_prediction_package


def make_case(case_id, group_id, start, order=(0, 1, 2), size=(32, 32)):
    Ps = cone_matrices(start, size)
    h, w = size
    points_3d = (np.array([-4.0, -3.0, -2.0]), np.array([4.0, 3.0, 2.0]))
    views = []
    yy, xx = np.mgrid[:h, :w]
    for index in order:
        xy = np.array([project(Ps[index], X) for X in points_3d])
        image = sum(np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3.0) for x, y in xy).astype(np.float32)
        views.append(View(index, float((start + 30 * index) % 360), "", image, xy, Ps[index]))
    return Case(case_id, group_id, views, metadata={"geometry_id": "fixed_cone"})


def test_joint_training_checkpoint_and_original_view_order(tmp_path):
    cases = [make_case("a", "a", 0, (2, 0, 1)), make_case("b", "b", 330, (1, 2, 0))]
    cfg = {"model_type": "joint_epipolar_v1", "epochs": 1, "batch_size": 1,
           "widths": [4, 8, 16, 32], "sigma": 1.5, "seed": 3}
    result = train_detector(cases, cfg, tmp_path / "joint")
    ckpt = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["format"] == "marker_joint_epipolar_checkpoint_v1"
    assert ckpt["fit_groups"] == ["a", "b"]
    assert np.isfinite(result["loss"])
    initial, refined = predict_joint_heatmaps(result["checkpoint"], cases[0])
    assert initial.shape == refined.shape == (3, 32, 32)
    assert np.isfinite(refined).all()
    initial2, refined2 = predict_joint_heatmaps(result["checkpoint"], cases[0])
    assert np.allclose(initial, initial2)
    assert np.allclose(refined, refined2)
    assert np.max(np.abs(initial - refined)) > 0
    bad = make_case("bad", "bad", 0)
    bad.views[1].P[0,3] += 1000
    with pytest.raises(ValueError, match="case bad"):
        predict_joint_heatmaps(result["checkpoint"], bad)


def test_default_training_keeps_legacy_checkpoint(tmp_path):
    case = make_case("legacy", "legacy", 0)
    result = train_detector([case], {"epochs": 1, "batch_size": 3, "widths": [4, 8, 16, 32]}, tmp_path)
    ckpt = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["format"] == "marker_unet_checkpoint_v1"


def write_manifest(tmp_path, cases, split):
    records = []
    samples = []
    for case in cases:
        views = []
        for view in case.views:
            stem = f"{case.case_id}_{view.view_id}"
            tifffile.imwrite(tmp_path / f"{stem}.tif", view.image)
            tifffile.imwrite(tmp_path / f"{stem}_mask.tif", np.zeros_like(view.image, dtype=np.uint8))
            views.append({"view_id": view.view_id, "angle_deg": view.angle_deg,
                          "projection_path": f"{stem}.tif", "mask_path": f"{stem}_mask.tif",
                          "marker_xy": view.marker_xy.tolist(), "P": view.P.tolist()})
            split_name = next(k for k, groups in split.items() if case.group_id in groups)
            samples.append({"case_id": case.case_id, "view_id": view.view_id,
                            "nnunet_case_id": stem, "group_id": case.group_id,
                            "split": split_name, "projection_path": f"{stem}.tif",
                            "mask_path": f"{stem}_mask.tif", "geometry_id": "fixed_cone",
                            "angle_deg": view.angle_deg})
        records.append({"case_id": case.case_id, "group_id": case.group_id,
                        "metadata": case.metadata, "views": views})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": records, "folds": {"fold0": split}}), encoding="utf-8")
    phase2_manifest = tmp_path / "phase2_manifest.json"
    phase2_manifest.write_text(json.dumps({"phase2_fold_id": "fold0", "splits": split,
                                           "samples": samples}), encoding="utf-8")
    return manifest, phase2_manifest


def test_joint_predict_cli_and_phase2_oof_package(tmp_path):
    cases = [make_case("a", "a", 0, (2, 0, 1)),
             make_case("b", "b", 330, (1, 2, 0)),
             make_case("c", "c", 30, (1, 0, 2))]
    split = {"train": ["a", "b"], "val": [], "test": ["c"]}
    manifest, phase2_manifest = write_manifest(tmp_path, cases, split)
    cfg = {"model_type": "joint_epipolar_v1", "epochs": 1, "batch_size": 1,
           "widths": [4, 8, 16, 32], "sigma": 1.5, "seed": 3,
           "top_k": 4, "top_l": 4, "phase2_fold_id": "fold0"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert main(["train", str(manifest), str(tmp_path / "trained"), "--config", str(config_path)]) == 0
    checkpoint = tmp_path / "trained" / "checkpoint.pt"
    assert main(["predict", str(manifest), str(checkpoint), str(tmp_path / "predictions")]) == 0
    for case in cases:
        record = json.loads((tmp_path / "predictions" / f"{case.case_id}.json").read_text())
        validate_prediction_record(record)
        assert [v["view_id"] for v in record["views"]] == [v.view_id for v in case.views]
        diagnostics = json.loads((tmp_path / "predictions" / "diagnostics" / f"{case.case_id}.json").read_text())
        assert len(diagnostics["views"]) == 3
        assert all("initial_candidates" in view and "refined_candidates" in view
                   and "no_geometry_marker_xy" in view for view in diagnostics["views"])
        with np.load(tmp_path / "predictions" / "diagnostics" / f"{case.case_id}.npz") as maps:
            assert maps["initial"].shape == maps["refined"].shape == (3, 32, 32)
    metrics = evaluate(cases, tmp_path / "predictions")
    assert metrics["diagnostic_view_count"] == 9
    assert metrics["ablation_no_geometry_PairAcc@tau"] is not None
    assert metrics["refined_topk_pair_recall"] is not None
    assert _model_id(checkpoint, cfg) != _model_id(checkpoint, {**cfg, "support_samples": 16})
    assert main(["export-fold", str(manifest), str(tmp_path / "export"),
                 "--split", "fold0", "--config", str(config_path)]) == 0
    validated = validate_prediction_package(phase2_manifest, tmp_path / "export", "fold0")
    assert len(validated["joined"]) == 9
    assert (tmp_path / "export" / "diagnostics" / "a.npz").is_file()
    registry = json.loads((tmp_path / "export" / "source_registry.json").read_text())
    for case in cases:
        record = json.loads((tmp_path / "export" / "predictions" / f"{case.case_id}.json").read_text())
        if case.group_id in split["train"]:
            assert record["prediction_source"] == "group_oof"
            assert case.group_id not in registry[record["model_id"]]["fit_groups"]
        else:
            assert record["prediction_source"] == "held_out"


def test_legacy_checkpoint_predict_cli(tmp_path):
    case = make_case("legacy", "legacy", 0)
    manifest, _ = write_manifest(tmp_path, [case], {"train": ["legacy"], "val": [], "test": []})
    checkpoint = train_detector([case], {"epochs": 1, "batch_size": 3,
                                         "widths": [4, 8, 16, 32]}, tmp_path / "legacy_model")["checkpoint"]
    assert main(["predict", str(manifest), checkpoint, str(tmp_path / "legacy_predictions")]) == 0
    record = json.loads((tmp_path / "legacy_predictions" / "legacy.json").read_text())
    validate_prediction_record(record)
