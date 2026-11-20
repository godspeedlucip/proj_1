"""Dataset601 input keeps its source grid, marker labels, and group split."""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.manifest import load_manifest, load_splits, validate_data
from nnunetv2.phase3.nifti_io import read_nifti
from nnunetv2.phase3.planning import make_phase3_plans
from .fixtures import make_config, synthetic_dataset


def dataset601_fixture(root: Path) -> Path:
    manifest_path, _ = synthetic_dataset(root / "original")
    original = json.loads(manifest_path.read_text())
    original["target_semantics"] = "stent_and_markers"
    dataset = root / "Dataset601_Phase3FDKUnique"
    for folder in ("imagesTr", "imagesTs", "labelsTr"):
        (dataset / folder).mkdir(parents=True)
    selected = []
    split_names = ("train", "train", "validation", "test")
    for record, split in zip(original["cases"], split_names):
        record["projection_source"] = "simulated_stent_projection"
        image = manifest_path.parent / record["image"]
        label = manifest_path.parent / record["label"]
        for path in (image, label):
            nii = nib.load(str(path))
            changed = nib.Nifti1Image(np.asarray(nii.dataobj),
                                      np.diag([0.188946] * 3 + [1]), nii.header)
            changed.header.set_xyzt_units("mm")
            nib.save(changed, str(path))
        image_folder = "imagesTs" if split == "test" else "imagesTr"
        os.link(image, dataset / image_folder / f"{record['case_id']}_0000.nii.gz")
        if split != "test":
            os.link(label, dataset / "labelsTr" / f"{record['case_id']}.nii.gz")
        selected.append({"case_id": record["case_id"], "group_id": record["group_id"],
                         "split": split, "angles_deg": record["angles_deg"],
                         "source_image": str(image), "source_label": str(label)})
    manifest_path.write_text(json.dumps(original))
    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "FDK"},
        "labels": {"background": 0, "stent_and_markers": 1},
        "numTraining": 3, "file_ending": ".nii.gz"}))
    (dataset / "splits_final.json").write_text(json.dumps([{
        "train": ["case_0", "case_1"], "val": ["case_2"]}]))
    (dataset / "source_mapping.json").write_text(json.dumps({
        "original_cases": 4, "unique_cases": 4,
        "split_counts": {"train": 2, "validation": 1, "test": 1},
        "source_manifest": str(manifest_path), "selected": selected}))
    return dataset


def test_non_point_two_source_spacing_is_preserved(tmp_path):
    source = dataset601_fixture(tmp_path)
    image = next((source / "imagesTr").glob("*.nii.gz"))
    _, meta = read_nifti(image)
    assert np.allclose(np.linalg.norm(meta.affine_mm[:3, :3], axis=0), 0.188946)
    raw = make_config(normalization__spacing_mm=[0.188946] * 3)
    config = resolve_config(raw)
    assert config.normalization.spacing_mm == (0.188946,) * 3


def test_dataset601_adapter_preserves_split_and_marker_semantics(tmp_path):
    from nnunetv2.phase3.dataset601 import import_dataset601

    source = dataset601_fixture(tmp_path)
    config = resolve_config(make_config())
    output = tmp_path / "adapter"
    manifest_path, splits_path, config_path = import_dataset601(source, output, config)
    manifest = load_manifest(manifest_path, require_labels=True)
    splits = load_splits(splits_path)
    assert manifest.target_semantics == "stent_and_markers"
    assert {case.projection_source for case in manifest.cases} == {"simulated_stent_projection"}
    assert (len(splits[0].train), len(splits[0].validation), len(splits[0].test)) == (2, 1, 1)
    assert validate_data(manifest, splits, formal=True)["case_count"] == 4
    adapted = resolve_config(json.loads(config_path.read_text()))
    assert adapted.normalization.spacing_mm == (0.188946,) * 3
    prepared = prepare_raw_fold(manifest, splits, source_fold=0, dataset_id=901,
                                output=tmp_path / "prepared")
    dataset_json = json.loads((prepared.raw_dir / "dataset.json").read_text())
    assert dataset_json["labels"] == {"background": 0, "stent_and_markers": 1, "ignore": 2}
    plans = make_phase3_plans(prepared, adapted)
    assert plans["configurations"]["3d_fullres"]["spacing"] == [0.188946] * 3
    assert plans["phase3"]["target_semantics"] == "stent_and_markers"
    assert set(prepared.train_ids) == {"case_0", "case_1"}
    assert prepared.validation_ids == ("case_2",)
    assert prepared.test_ids == ("case_3",)


def test_dataset601_adapter_rejects_split_mismatch(tmp_path):
    from nnunetv2.phase3.dataset601 import import_dataset601

    source = dataset601_fixture(tmp_path)
    split_path = source / "splits_final.json"
    split_path.write_text(json.dumps([{"train": ["case_0", "case_2"], "val": ["case_1"]}]))
    with pytest.raises(ValueError, match="split"):
        import_dataset601(source, tmp_path / "adapter", resolve_config(make_config()))


def test_all_case_export_is_accepted_without_deduplicating_case_ids(tmp_path):
    from nnunetv2.phase3.dataset601 import import_dataset601

    source = dataset601_fixture(tmp_path)
    source.rename(tmp_path / "Dataset602_Phase3All")
    source = tmp_path / "Dataset602_Phase3All"
    mapping_path = source / "source_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["exported_cases"] = 4
    mapping_path.write_text(json.dumps(mapping))
    manifest_path, splits_path, _ = import_dataset601(source, tmp_path / "all_adapter",
                                                      resolve_config(make_config()))
    assert len(load_manifest(manifest_path, require_labels=True).cases) == 4
    assert len(load_splits(splits_path)[0].train) == 2


def test_duplicate_case_ids_share_prepared_arrays_without_losing_cases(tmp_path):
    from nnunetv2.phase3.dataset601 import import_dataset601
    from nnunetv2.phase3.planning import prepare_training_fold

    source = dataset601_fixture(tmp_path)
    source.rename(tmp_path / "Dataset602_Phase3All")
    source = tmp_path / "Dataset602_Phase3All"
    original_manifest = tmp_path / "original/manifest.json"
    original = json.loads(original_manifest.read_text())
    repeated = dict(original["cases"][0], case_id="case_4",
                    image="case_4_0000.nii.gz", label="case_4.nii.gz")
    os.link(tmp_path / "original/case_0_0000.nii.gz", tmp_path / "original/case_4_0000.nii.gz")
    os.link(tmp_path / "original/case_0.nii.gz", tmp_path / "original/case_4.nii.gz")
    original["cases"].append(repeated)
    original_manifest.write_text(json.dumps(original))
    os.link(source / "imagesTr/case_0_0000.nii.gz", source / "imagesTr/case_4_0000.nii.gz")
    os.link(source / "labelsTr/case_0.nii.gz", source / "labelsTr/case_4.nii.gz")
    mapping_path = source / "source_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["exported_cases"] = 5
    mapping["split_counts"]["train"] = 3
    mapping["selected"].append(dict(mapping["selected"][0], case_id="case_4",
                                    source_image=str(tmp_path / "original/case_4_0000.nii.gz"),
                                    source_label=str(tmp_path / "original/case_4.nii.gz")))
    mapping_path.write_text(json.dumps(mapping))
    dataset_json_path = source / "dataset.json"
    dataset_json = json.loads(dataset_json_path.read_text())
    dataset_json["numTraining"] = 4
    dataset_json_path.write_text(json.dumps(dataset_json))
    splits_path = source / "splits_final.json"
    splits = json.loads(splits_path.read_text())
    splits[0]["train"].append("case_4")
    splits_path.write_text(json.dumps(splits))

    manifest_path, split_path, config_path = import_dataset601(
        source, tmp_path / "adapter", resolve_config(make_config()))
    manifest = load_manifest(manifest_path, require_labels=True)
    prepared = prepare_raw_fold(manifest, load_splits(split_path), source_fold=0,
                                dataset_id=902, output=tmp_path / "prepared")
    assert len(prepared.train_ids) == 3
    assert (prepared.raw_dir / "imagesTr/case_0_0000.nii.gz").samefile(
        prepared.raw_dir / "imagesTr/case_4_0000.nii.gz")
    prepared = prepare_training_fold(prepared, resolve_config(json.loads(config_path.read_text())))
    data_dir = prepared.preprocessed_dir / (prepared.plans_name + "_3d_fullres")
    assert (data_dir / "case_0.b2nd").samefile(data_dir / "case_4.b2nd")
    assert (data_dir / "case_0_seg.b2nd").samefile(data_dir / "case_4_seg.b2nd")
