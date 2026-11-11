import json

import numpy as np
import pytest
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.experiment_planning.experiment_planners.fdk_planner import FDKnnUNetPlanner
from nnunetv2.preprocessing.preprocessors.fdk_preprocessor import FDKPreprocessor
from nnunetv2.phase3.storage import FDKDatasetBlosc2
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.phase3.config import config_digest, resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.planning import make_phase3_plans, prepare_training_fold, validate_phase3_plans
from nnunetv2.phase3.resampling import identity_resample
from .fixtures import make_config, synthetic_dataset


@pytest.fixture
def prepared(tmp_path):
    mp, sp = synthetic_dataset(tmp_path / "input")
    return prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                            source_fold=0, dataset_id=902, output=tmp_path / "fold")


@pytest.mark.parametrize("smoke", [False, True])
def test_real_plans_manager_loads_only_fixed_fullres(prepared, smoke):
    raw = make_config()
    if smoke:
        raw.update(run_kind="smoke")
        raw["cnn"].update(patch_size=[32, 32, 32], batch_size=1)
        raw["training"].update(max_optimizer_steps=4, updates_per_epoch=2)
    cfg = resolve_config(raw)
    plans = make_phase3_plans(prepared, cfg)
    validate_phase3_plans(plans)
    manager = PlansManager(plans)
    configuration = manager.get_configuration("3d_fullres")
    dataset = json.loads((prepared.raw_dir / "dataset.json").read_text(encoding="utf-8"))
    label_manager = manager.get_label_manager(dataset)
    assert manager.available_configurations == ["3d_fullres"]
    assert label_manager.num_segmentation_heads == 2
    assert manager.image_reader_writer_class.__name__ == "FDKNiftiIO"
    assert manager.experiment_planner_class is FDKnnUNetPlanner
    assert manager.transpose_forward == manager.transpose_backward == [0, 1, 2]
    assert configuration.patch_size == ([32, 32, 32] if smoke else [128, 128, 128])
    assert configuration.normalization_schemes == ["NoNormalization", "NoNormalization"]
    assert configuration.preprocessor_class is FDKPreprocessor
    assert configuration.network_arch_class_name.endswith("PlainConvUNet")
    arch = configuration.network_arch_init_kwargs
    assert arch["features_per_stage"] == [32, 64, 128, 256, 320]
    assert arch["strides"] == [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    assert arch["n_conv_per_stage"] == [2] * 5
    assert arch["n_conv_per_stage_decoder"] == [2] * 4
    assert plans["phase3"]["config_digest"] == config_digest(cfg)
    assert FDKnnUNetPlanner(prepared, cfg).plan_experiment() == plans


def test_identity_resample_preserves_dtype_and_rejects_grid_changes():
    array = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    result = identity_resample(array, (2, 3, 4), (0.20005, 0.2, 0.2), (0.2, 0.2, 0.2))
    assert result is array
    tensor = torch.from_numpy(array)
    assert identity_resample(tensor, (2, 3, 4), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2)) is tensor
    with pytest.raises(ValueError):
        identity_resample(array, (2, 3, 5), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2))
    with pytest.raises(ValueError):
        identity_resample(array, (2, 3, 4), (0.20011, 0.2, 0.2), (0.2, 0.2, 0.2))


def test_preprocessor_saves_uncropped_blosc2_and_predictor_metadata(prepared):
    cfg = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                     cnn__batch_size=1, training__max_optimizer_steps=4,
                                     training__updates_per_epoch=2))
    completed = prepare_training_fold(prepared, cfg)
    assert completed.preprocessed_complete
    assert completed.plans_name == "FDKPhase3_" + config_digest(cfg)[:12]
    plans_path = completed.preprocessed_dir / f"{completed.plans_name}.json"
    manager = PlansManager(str(plans_path))
    configuration = manager.get_configuration("3d_fullres")
    assert configuration.resampling_fn_probabilities.func.__name__ == "identity_resample"
    data_folder = completed.preprocessed_dir / configuration.data_identifier
    dataset = FDKDatasetBlosc2(str(data_folder))
    assert set(dataset.identifiers) == {"case_0", "case_1", "case_2"}
    data, seg, previous, properties = dataset.load_case("case_0")
    assert previous is None
    patch = crop_and_pad_nd(data, [[0, 8], [0, 8], [0, 8]])
    assert patch.shape == (2, 8, 8, 8)
    assert np.isfinite(patch).all()
    data, seg = np.asarray(data), np.asarray(seg)
    assert data.shape == (2, 20, 18, 16)
    assert seg.shape == (1, 20, 18, 16)
    assert np.isfinite(data).all() and set(np.unique(seg)) == {0, 1}
    assert properties["shape_before_cropping"] == (20, 18, 16)
    assert properties["shape_after_cropping_and_before_resampling"] == (20, 18, 16)
    assert properties["bbox_used_for_cropping"] == [[0, 20], [0, 18], [0, 16]]
    assert 1 in properties["class_locations"]
    assert (-1, 0, 1) in properties["class_locations"]
    assert properties["fdk_meta"].original_shape == (16, 18, 20)
    fingerprint = json.loads((completed.preprocessed_dir / "dataset_fingerprint.json").read_text(encoding="utf-8"))
    assert set(fingerprint["case_ids"]) == {"case_0", "case_1", "case_2"}
    assert "case_3" not in json.dumps(fingerprint)


def test_invalid_plan_strides_are_rejected(prepared):
    plans = make_phase3_plans(prepared, resolve_config(make_config()))
    plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["strides"][-1] = [1, 2, 2]
    with pytest.raises(ValueError):
        validate_phase3_plans(plans)
