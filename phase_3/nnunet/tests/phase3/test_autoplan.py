import pytest
import json
import os
import numpy as np
from types import SimpleNamespace

from nnunetv2.phase3 import autoplan
from nnunetv2.phase3.autoplan import apply_nnunet_patch_plan
from .fixtures import make_config


def test_native_planner_uses_positive_thread_count_and_restores_zero(tmp_path, monkeypatch):
    from nnunetv2.phase3 import planning
    from nnunetv2.experiment_planning.experiment_planners import default_experiment_planner as native

    prepared = SimpleNamespace(preprocessed_dir=tmp_path / "preprocessed",
                               raw_root=tmp_path / "raw", preprocessed_root=tmp_path,
                               dataset_name="Dataset999_Test", train_ids=["train"],
                               validation_ids=["validation"])
    monkeypatch.setattr(planning, "_source_geometry",
                        lambda _: ([64, 64, 64], [1.0, 1.0, 1.0], {"shapes_after_crop": []}))
    monkeypatch.setenv("nnUNet_n_proc_DA", "0")

    class Planner:
        def __init__(self, *args, **kwargs):
            pass

        def get_plans_for_configuration(self, *args):
            assert os.environ["nnUNet_n_proc_DA"] == "1"
            return {"patch_size": (64, 64, 64), "batch_size": 1}

    monkeypatch.setattr(native, "ExperimentPlanner", Planner)
    plan, shape, spacing = autoplan.plan_for_prepared(prepared, 24)
    assert plan["patch_size"] == (64, 64, 64)
    assert shape == [64, 64, 64]
    assert spacing == [1.0, 1.0, 1.0]
    assert os.environ["nnUNet_n_proc_DA"] == "0"
    assert not (prepared.preprocessed_dir / "dataset_fingerprint.json").exists()


def test_apply_nnunet_patch_plan_uses_planned_size_and_source_spacing():
    raw = make_config()
    planned = {"patch_size": (192, 160, 160), "batch_size": 2}
    result = apply_nnunet_patch_plan(raw, planned, (0.188946, 0.188946, 0.188946))
    assert result.cnn.patch_size == (192, 160, 160)
    assert result.cnn.batch_size == 2
    assert result.normalization.spacing_mm == (0.188946,) * 3
    assert raw["cnn"]["patch_size"] == [128, 128, 128]


def test_apply_nnunet_patch_plan_can_raise_sparse_target_sampling():
    result = apply_nnunet_patch_plan(make_config(),
                                     {"patch_size": (192, 192, 192), "batch_size": 2},
                                     (0.188946, 0.188946, 0.188946),
                                     foreground_probability=0.9)
    assert result.training.foreground_probability == 0.9


def test_apply_nnunet_patch_plan_rejects_incompatible_swin_edges():
    with pytest.raises(ValueError, match="divisible by 32"):
        apply_nnunet_patch_plan(make_config(), {"patch_size": (96, 128, 120), "batch_size": 2},
                                (0.188946, 0.188946, 0.188946))


def test_autoplan_report_serializes_native_numpy_patch(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared.json"
    source = tmp_path / "source.json"
    output = tmp_path / "generated.json"
    prepared.write_text("{}", encoding="utf-8")
    source.write_text(json.dumps(make_config()), encoding="utf-8")
    monkeypatch.setattr(autoplan.PreparedFold, "from_dict", lambda _: object())
    monkeypatch.setattr(autoplan, "plan_for_prepared",
                        lambda *_: ({"patch_size": (np.int64(192), np.int64(160), np.int64(160)),
                                     "batch_size": np.int64(2)}, [533, 512, 512], [0.188946] * 3))
    assert autoplan.main(["--prepared", str(prepared), "--config", str(source),
                          "--gpu-memory-target-gb", "24", "--output", str(output)]) == 0
    assert json.loads(output.with_suffix(".plan.json").read_text())["native_patch_size_dhw"] == [192, 160, 160]
