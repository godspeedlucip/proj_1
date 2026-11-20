import copy
import json

from nnunetv2.phase3 import manifest as manifest_module
from nnunetv2.phase3 import checkpointing


def test_relocated_manifest_preserves_checkpoint_identity(tmp_path):
    old_raw = tmp_path / "old_raw"
    new_raw = tmp_path / "Dataset603" / "raw"
    old_test_labels = tmp_path / "old_test_labels"
    original = {
        "schema_version": "fdk_stent_manifest_v1",
        "cases": [
            {
                "case_id": "train_case", "group_id": "train_stent",
                "source_model_ids": ["stent_model:train"], "patient_id": None,
                "image": str(old_raw / "imagesTr/train_case_0000.nii.gz"),
                "label": str(old_raw / "labelsTr/train_case.nii.gz"),
                "angles_deg": [0.0, 30.0, 60.0],
                "projection_source": "simulated_stent_projection",
                "upstream_run_id": None, "input_kind": "fdk_3view",
            },
            {
                "case_id": "test_case", "group_id": "test_stent",
                "source_model_ids": ["stent_model:test"], "patient_id": None,
                "image": str(old_raw / "imagesTs/test_case_0000.nii.gz"),
                "label": str(old_test_labels / "test_case_stent.nii.gz"),
                "angles_deg": [0.0, 30.0, 60.0],
                "projection_source": "simulated_stent_projection",
                "upstream_run_id": None, "input_kind": "fdk_3view",
            },
        ],
        "upstream_registry": {}, "target_semantics": "stent_and_markers",
        "source_directory": str(tmp_path / "source"), "group_provenance": {},
    }
    expected = manifest_module.json_digest(original)
    relocated = copy.deepcopy(original)
    relocated["cases"][0]["image"] = str(new_raw / "imagesTr/train_case_0000.nii.gz")
    relocated["cases"][0]["label"] = str(new_raw / "labelsTr/train_case.nii.gz")
    relocated["cases"][1]["image"] = str(new_raw / "imagesTs/test_case_0000.nii.gz")
    relocated["cases"][1]["label"] = str(new_raw / "labelsTs/test_case.nii.gz")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    manifest_path = adapter / "manifest.json"
    manifest_path.write_text(json.dumps(relocated))
    (adapter / "relocation.json").write_text(json.dumps({
        "schema_version": "phase3_dataset603_relocation_v1",
        "original_manifest_digest": expected,
        "relocated_manifest_digest": manifest_module.json_digest(relocated),
        "old_raw_prefix": str(old_raw), "new_raw_prefix": str(new_raw),
        "old_test_label_prefix": str(old_test_labels),
        "new_test_label_prefix": str(new_raw / "labelsTs"),
    }))

    assert hasattr(manifest_module, "manifest_digest_matches")
    loaded = manifest_module.manifest_from_dict(relocated, adapter)
    assert manifest_module.manifest_digest_matches(manifest_path, loaded, expected)
    changed = copy.deepcopy(relocated)
    changed["cases"][1]["group_id"] = "another_group"
    changed_manifest = manifest_module.manifest_from_dict(changed, adapter)
    assert not manifest_module.manifest_digest_matches(manifest_path, changed_manifest, expected)


def test_checkpoint_log_follows_moved_result_folder(tmp_path):
    old_log = tmp_path / "old" / "Model" / "fold_0" / "training_log.txt"
    new_fold = tmp_path / "new" / "Model" / "fold_0"
    new_fold.mkdir(parents=True)
    new_log = new_fold / old_log.name
    new_log.write_bytes(b"completed epoch\n")

    assert hasattr(checkpointing, "resolve_saved_log_path")
    assert checkpointing.resolve_saved_log_path(old_log, new_fold, 10) == new_log

    mismatched = tmp_path / "old" / "AnotherModel" / "fold_0" / old_log.name
    try:
        checkpointing.resolve_saved_log_path(mismatched, new_fold, 10)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("A different model's log must not be accepted")
