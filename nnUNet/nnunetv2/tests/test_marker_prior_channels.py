from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from phase_2.tools.marker_guided.build_prior_channels import PriorConfig, build_e5u_dataset, build_marker_channels
from phase_2.tools.marker_guided.validate_dataset import DatasetValidationError, validate_prediction_package


def test_prior_gaussian_subpixel_endpoint_and_swap_invariance():
    record = {"pair_status": "reliable", "marker_xy": [[2.4, 3.2], [8.1, 3.7]], "error_radius_px": [0.2, 0.4]}
    point, band, valid = build_marker_channels(record, (12, 12), PriorConfig(endpoint_taper_px=1.0))
    swapped = {**record, "marker_xy": list(reversed(record["marker_xy"])), "error_radius_px": list(reversed(record["error_radius_px"]))}
    point2, band2, valid2 = build_marker_channels(swapped, (12, 12), PriorConfig(endpoint_taper_px=1.0))
    assert point[3, 2] > point[3, 1]
    assert band[3, 2] > band[3, 0]
    assert np.allclose(point, point2) and np.allclose(band, band2)
    assert np.all(valid == 1) and np.array_equal(valid, valid2)
    assert band[3, 0] < band[3, 2]  # finite segment taper beyond the endpoint


def test_degenerate_band_invalid_and_uncertain_validity():
    degenerate = {"pair_status": "reliable", "marker_xy": [[4.0, 4.0], [4.2, 4.0]], "error_radius_px": [1, 1]}
    point, band, valid = build_marker_channels(degenerate, (9, 9), PriorConfig(d_min=0.5))
    assert point.max() > 0 and band.max() == 0 and valid.min() == 1
    invalid = {"pair_status": "invalid", "marker_xy": None, "error_radius_px": None}
    assert all(np.count_nonzero(x) == 0 for x in build_marker_channels(invalid, (9, 9)))
    uncertain = {**degenerate, "pair_status": "uncertain"}
    assert np.all(build_marker_channels(uncertain, (9, 9))[2] == 1)


def _package_fixture(tmp_path):
    samples, records, split_groups = [], [], {"train": ["g0", "g1"], "val": ["g2"], "test": ["g3"]}
    for case_index, (split, group) in enumerate((("train", "g0"), ("train", "g1"), ("val", "g2"), ("test", "g3"))):
        case_id = f"case{case_index}"
        views = []
        for view_id in range(3):
            projection = tmp_path / f"{case_id}_{view_id}.tif"
            mask = tmp_path / f"{case_id}_{view_id}_mask.tif"
            image = np.full((12, 12), view_id + case_index, dtype=np.uint8)
            tifffile.imwrite(projection, image)
            tifffile.imwrite(mask, np.zeros_like(image, dtype=np.uint8))
            samples.append({"case_id": case_id, "view_id": view_id, "nnunet_case_id": f"c{case_index}v{view_id}",
                            "group_id": group, "split": split, "projection_path": str(projection),
                            "mask_path": str(mask), "geometry_id": "geo1", "angle_deg": view_id * 10.0})
            status = "invalid" if case_index == 3 and view_id == 0 else ("uncertain" if view_id == 2 else "reliable")
            views.append({"view_id": view_id, "image_size": [12, 12],
                          "marker_xy": None if status == "invalid" else [[2.2, 3.3], [8.1, 7.4]],
                          "error_radius_px": None if status == "invalid" else [0.5, 0.7],
                          "q_pair": 0 if status == "invalid" else None, "pair_status": status})
        records.append({"schema_version": "phase1_marker_case_v1", "case_id": case_id, "group_id": group,
                        "model_id": "model0" if case_index == 0 else ("model1" if case_index == 1 else "model2"),
                        "geometry_id": "geo1", "prediction_source": "group_oof" if split == "train" else "held_out",
                        "fold_id": "p1fold0", "views": views})
    manifest = {"phase2_fold_id": "fold0", "splits": split_groups, "samples": samples}
    package = {"phase2_fold_id": "fold0", "source_registry": [
        {"model_id": "model0", "model_artifact_id": "weights0", "pipeline_config_id": "decode0", "fit_group_ids": ["g1"], "selection_group_ids": []},
        {"model_id": "model1", "model_artifact_id": "weights1", "pipeline_config_id": "decode1", "fit_group_ids": ["g0"], "selection_group_ids": []},
        {"model_id": "model2", "model_artifact_id": "weights2", "pipeline_config_id": "decode2", "fit_group_ids": ["g0", "g1"], "selection_group_ids": []}],
        "calibration_registry": [], "cases": records}
    return manifest, package


def test_package_validation_and_dataset_export(tmp_path):
    manifest, package = _package_fixture(tmp_path)
    validated = validate_prediction_package(manifest, package, "fold0")
    assert len(validated["joined"]) == 12
    out = build_e5u_dataset(manifest, package, "fold0", tmp_path / "out", dataset_id=701)
    dataset = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["marker_guided"]["channel_order"] == ["projection", "point", "band", "validity"]
    assert (out / "imagesTr" / "c0v0_0000.tif").is_file()
    assert (out / "imagesTr" / "c0v0_0003.tif").is_file()
    mapping = json.loads((out / "case_mapping.json").read_text(encoding="utf-8"))
    assert mapping[0]["angle_deg"] == 0
    assert np.array_equal(tifffile.imread(out / "imagesTs" / "c3v0_0000.tif"), tifffile.imread(tmp_path / "case3_0.tif"))
    for split in ("train", "val"):
        rows = [r for r in manifest["samples"] if r["split"] == split]
        assert len(rows) % 3 == 0


@pytest.mark.parametrize("mutator", ["duplicate_view", "geometry", "image_size", "missing_view", "leakage"])
def test_package_rejects_contract_mismatches(tmp_path, mutator):
    manifest, package = _package_fixture(tmp_path)
    if mutator == "duplicate_view":
        package["cases"][0]["views"][1]["view_id"] = 0
    elif mutator == "geometry":
        package["cases"][0]["geometry_id"] = "wrong"
    elif mutator == "image_size":
        package["cases"][0]["views"][0]["image_size"] = [11, 12]
    elif mutator == "missing_view":
        package["cases"][0]["views"].pop()
    elif mutator == "leakage":
        package["source_registry"][0]["fit_group_ids"] = ["g0"]
    with pytest.raises(DatasetValidationError):
        validate_prediction_package(manifest, package, "fold0")


def test_missing_q_pair_is_not_treated_as_explicit_null(tmp_path):
    manifest, package = _package_fixture(tmp_path)
    del package["cases"][0]["views"][0]["q_pair"]
    with pytest.raises(DatasetValidationError, match="explicitly include q_pair"):
        validate_prediction_package(manifest, package, "fold0")


def test_export_refuses_nonempty_output_directory(tmp_path):
    manifest, package = _package_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "stale.tif").write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="refusing to mix"):
        build_e5u_dataset(manifest, package, "fold0", output)
    assert (output / "stale.tif").read_bytes() == b"keep"


def test_prior_config_rejects_nonfinite_parameters():
    with pytest.raises(ValueError, match="finite numeric"):
        build_marker_channels({"pair_status": "invalid"}, (8, 8), PriorConfig(sigma_0=float("nan")))
