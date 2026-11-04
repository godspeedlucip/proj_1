import json

import numpy as np
import pytest
import tifffile

from phase_1.marker_pipeline.data import load_manifest, validate_group_split
from phase_1.marker_pipeline.synthetic import make_synthetic_dataset


def _case(tmp_path, angles=(350.0, 0.0, 10.0)):
    image = np.zeros((24, 32), dtype=np.uint8)
    tifffile.imwrite(tmp_path / "image.tif", image)
    p = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]
    return {
        "case_id": "case_a", "group_id": "group_a",
        "views": [
            {"view_id": i, "angle_deg": angle, "projection_path": "image.tif",
             "marker_xy": [[3.5, 4.5], [28.0, 18.0]], "P": p}
            for i, angle in enumerate(angles)
        ],
    }


def _write_manifest(tmp_path, case):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    return path


def test_manifest_accepts_wrapped_angles_and_checks_image_contract(tmp_path):
    cases = load_manifest(_write_manifest(tmp_path, _case(tmp_path)))
    assert len(cases) == 1
    assert cases[0].views[0].image.shape == (24, 32)
    assert cases[0].views[0].marker_xy.shape == (2, 2)


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["views"].pop(), "exactly three"),
    (lambda c: c["views"][2].update(view_id=1), "distinct"),
    (lambda c: c["views"][0].update(marker_xy=[[float("nan"), 1], [2, 2]]), "finite"),
    (lambda c: c["views"][0].update(marker_xy=[[32, 1], [2, 2]]), "bounds"),
    (lambda c: c["views"][0].update(P=[[1, 2], [3, 4]]), "3x4"),
    (lambda c: c["views"][0].update(projection_path="missing.tif"), "image"),
])
def test_manifest_rejects_invalid_view_contract(tmp_path, mutate, message):
    case = _case(tmp_path)
    mutate(case)
    with pytest.raises(ValueError, match=message):
        load_manifest(_write_manifest(tmp_path, case))


def test_angles_must_fit_one_circular_sixty_degree_window(tmp_path):
    with pytest.raises(ValueError, match="60"):
        load_manifest(_write_manifest(tmp_path, _case(tmp_path, (0, 30, 70))))


def test_group_split_is_mutually_exclusive():
    cases = [type("Case", (), {"group_id": "g1"})(), type("Case", (), {"group_id": "g2"})()]
    validate_group_split(cases, {"train": ["g1"], "val": [], "test": ["g2"]})
    with pytest.raises(ValueError, match="overlap"):
        validate_group_split(cases, {"train": ["g1"], "val": ["g1"], "test": []})


def test_synthetic_dataset_has_six_groups_and_three_difficulty_types(tmp_path):
    manifest_path = make_synthetic_dataset(tmp_path / "synthetic", seed=7)
    cases = load_manifest(manifest_path)
    assert len({case.group_id for case in cases}) >= 6
    assert {case.difficulty for case in cases} >= {"normal", "close", "low_disparity"}
    split = json.loads(manifest_path.read_text(encoding="utf-8"))["folds"]["fold_0"]
    validate_group_split(cases, split)
