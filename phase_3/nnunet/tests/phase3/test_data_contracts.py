import json

import nibabel as nib
import numpy as np
import pytest

from nnunetv2.imageio.fdk_reader_writer import FDKNiftiIO
from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.phase3.manifest import load_manifest, load_splits, validate_data
from nnunetv2.phase3.nifti_io import load_volume, read_nifti, same_grid
from nnunetv2.phase3.provenance import file_sha256
from .fixtures import synthetic_dataset


def test_valid_grouped_manifest_and_hash_snapshot(tmp_path):
    mp, sp = synthetic_dataset(tmp_path)
    manifest = load_manifest(mp, require_labels=True)
    audit = validate_data(manifest, load_splits(sp), formal=True)
    assert audit["case_count"] == 4
    assert audit["group_count"] == 4
    assert audit["files"][0]["image_sha256"] == file_sha256(manifest.cases[0].image)
    assert manifest.to_dict()["cases"][0]["source_model_ids"] == ["source_0"]


@pytest.mark.parametrize("change", ["shared_source", "shared_image", "upstream_fit", "upstream_selection",
                                     "upstream_calibration", "wrong_semantics", "mixed_projection"])
def test_rejects_source_and_target_leakage(tmp_path, change):
    mp, sp = synthetic_dataset(tmp_path)
    raw = json.loads(mp.read_text(encoding="utf-8"))
    cases = raw["cases"]
    if change == "shared_source":
        cases[3]["source_model_ids"] = cases[0]["source_model_ids"]
        cases[3]["angles_deg"] = [10.0, 40.0, 70.0]
    elif change == "shared_image":
        cases[3]["image"] = cases[0]["image"]
    elif change.startswith("upstream_"):
        cases[3].update(projection_source="predicted_mask", upstream_run_id="run")
        raw["upstream_registry"]["run"] = {
            "fit_group_ids": [], "selection_group_ids": [], "calibration_group_ids": []}
        raw["upstream_registry"]["run"][change.removeprefix("upstream_") + "_group_ids"] = ["g3"]
    elif change == "wrong_semantics":
        raw["target_semantics"] = "vessel"
    elif change == "mixed_projection":
        cases[3]["projection_source"] = "measured"
    mp.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_data(load_manifest(mp, require_labels=True), load_splits(sp), formal=True)


def test_non_cubic_tensor_order_and_reader_writer_preserve_grid(tmp_path):
    array = np.zeros((13, 15, 17), np.float32)
    array[3, 7, 11] = 19
    affine = np.array([[0, -0.2, 0, 10], [0.2, 0, 0, -4],
                       [0, 0, 0.2, 7], [0, 0, 0, 1]], dtype=float)
    paths = []
    for channel in range(2):
        path = tmp_path / f"sample_{channel:04d}.nii.gz"
        nii = nib.Nifti1Image(array + channel, affine)
        nii.header.set_xyzt_units("mm")
        nii.set_qform(affine, 1)
        nii.set_sform(affine, 1)
        nib.save(nii, path)
        paths.append(str(path))
    loaded, meta = read_nifti(paths[0])
    assert loaded.shape == (17, 15, 13)
    assert loaded[11, 7, 3] == 19
    assert meta.unit == "mm"
    io = FDKNiftiIO()
    assert recursive_find_reader_writer_by_name("FDKNiftiIO") is FDKNiftiIO
    images, properties = io.read_images(paths)
    assert images.shape == (2, 17, 15, 13)
    assert images[1, 11, 7, 3] == 20
    assert tuple(properties["spacing"]) == pytest.approx((0.2, 0.2, 0.2), abs=1e-6)
    seg = (images[0] > 0).astype(np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    prob_path = tmp_path / "prob.nii.gz"
    io.write_seg(seg, str(mask_path), properties)
    io.write_probability(seg.astype(np.float32) * 0.7, str(prob_path), properties)
    mask, mask_meta = read_nifti(mask_path)
    prob, prob_meta = read_nifti(prob_path)
    np.testing.assert_array_equal(mask, seg)
    np.testing.assert_allclose(prob, seg * 0.7, atol=1e-7)
    assert same_grid(meta, mask_meta) and same_grid(meta, prob_meta)
    assert nib.load(mask_path).header.get_xyzt_units()[0] == "mm"
    assert nib.load(prob_path).get_data_dtype() == np.float32


@pytest.mark.parametrize("unit,scale", [("mm", 1.0), ("meter", 0.001), ("micron", 1000.0)])
def test_physical_units_and_slope_are_read_once(tmp_path, unit, scale):
    affine = np.diag([0.2, 0.2, 0.2, 1.0])
    stored = affine.copy()
    stored[:3] *= scale
    array = np.zeros((7, 9, 11), np.int16)
    array[2, 4, 6] = 2
    nii = nib.Nifti1Image(array, stored)
    nii.header.set_xyzt_units(unit)
    nii.header.set_slope_inter(2.0, 10.0)
    path = tmp_path / "scaled.nii.gz"
    nib.save(nii, path)
    data, meta = read_nifti(path)
    assert data[6, 4, 2] == pytest.approx(14.0)
    np.testing.assert_allclose(meta.affine_mm, affine, atol=1e-6)
    assert meta.unit == unit
    io = FDKNiftiIO()
    images, properties = io.read_images([str(path)])
    out = tmp_path / "prob.nii.gz"
    io.write_probability(np.full_like(images[0], 0.25), str(out), properties)
    assert nib.load(out).get_fdata()[2, 4, 6] == pytest.approx(0.25)
    assert nib.load(out).header.get_xyzt_units()[0] == unit


@pytest.mark.parametrize("damage", ["label2", "nan", "qform", "spacing", "unit", "affine"])
def test_invalid_label_or_image_grid_rejected(tmp_path, damage):
    mp, _ = synthetic_dataset(tmp_path)
    case = load_manifest(mp, require_labels=True).cases[0]
    path = case.label if damage in ("label2", "affine") else case.image
    nii = nib.load(path)
    array = nii.get_fdata()
    affine = nii.affine.copy()
    header = nii.header.copy()
    if damage == "label2":
        array[0, 0, 0] = 2
    elif damage == "nan":
        array[0, 0, 0] = np.nan
    elif damage == "affine":
        affine[0, 3] += 1
    elif damage == "spacing":
        affine[0, 0] = 0.20011
    elif damage == "unit":
        header.set_xyzt_units("unknown")
    out = nib.Nifti1Image(array, affine, header)
    if damage == "qform":
        other = affine.copy()
        other[0, 3] += 2
        out.set_qform(other, 1)
        out.set_sform(affine, 1)
    nib.save(out, path)
    with pytest.raises(ValueError):
        load_volume(case, require_label=True)


def test_spacing_inside_tolerance_keeps_original_grid(tmp_path):
    affine = np.diag([0.20005, 0.2, 0.2, 1.0])
    path = tmp_path / "nearly_02.nii.gz"
    nii = nib.Nifti1Image(np.ones((5, 6, 7), np.float32), affine)
    nii.header.set_xyzt_units("mm")
    nib.save(nii, path)
    data, meta = read_nifti(path)
    assert data.shape == (7, 6, 5)
    assert meta.affine_mm[0, 0] == pytest.approx(0.20005, abs=1e-6)


def test_two_image_channels_with_different_affines_fail(tmp_path):
    paths = []
    for channel, offset in enumerate((0.0, 1.0)):
        affine = np.diag([0.2, 0.2, 0.2, 1.0])
        affine[0, 3] = offset
        nii = nib.Nifti1Image(np.ones((5, 6, 7), np.float32), affine)
        nii.header.set_xyzt_units("mm")
        path = tmp_path / f"channel_{channel}.nii.gz"
        nib.save(nii, path)
        paths.append(str(path))
    with pytest.raises(ValueError):
        FDKNiftiIO().read_images(paths)
