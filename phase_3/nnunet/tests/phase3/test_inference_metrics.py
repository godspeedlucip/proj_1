import nibabel as nib
from dataclasses import asdict
import numpy as np
import pytest
import torch

from phase_3.fdk_unet.evaluate import compute_case_metrics as old_metrics
from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import make_views, prepare_raw_fold
from nnunetv2.phase3.evaluation import aggregate_group_metrics, compute_case_metrics
from nnunetv2.phase3.inference import predict_volume, write_prediction
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.nifti_io import VolumeCase, read_nifti
from nnunetv2.phase3.planning import make_phase3_plans
from nnunetv2.phase3.provenance import file_sha256, validate_prediction_provenance
from nnunetv2.phase3.manifest import json_digest, select_cases
from .fixtures import make_config, synthetic_dataset


class ConstantLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.))
        self.decoder = type("Decoder", (), {"deep_supervision": True})()
        self.observed = None

    def set_deep_supervision(self, value):
        self.decoder.deep_supervision = value

    def forward(self, data):
        self.observed = data.detach().cpu()
        return torch.zeros((data.shape[0], 2, *data.shape[2:]), device=data.device) + self.scale


def _setup(tmp_path):
    mp, sp = synthetic_dataset(tmp_path / "input")
    manifest, splits = load_manifest(mp, require_labels=True), load_splits(sp)
    prepared = prepare_raw_fold(manifest, splits, source_fold=0, dataset_id=911,
                                output=tmp_path / "fold")
    config = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                        cnn__batch_size=1, training__max_optimizer_steps=4,
                                        training__updates_per_epoch=2, swin__enabled=False,
                                        swin__execute_through="layers2", fusion__enabled=False,
                                        fusion__bottleneck_enabled=False, tubular__enabled=False,
                                        projection__enabled=False, projection__weight=0.0))
    return make_phase3_plans(prepared, config), config, prepared.stats


@pytest.mark.parametrize("shape", [(97, 113, 65), (17, 19, 21)])
def test_native_sliding_window_restores_shape_grid_and_model_state(tmp_path, shape):
    torch.set_num_threads(2)
    plans, config, stats = _setup(tmp_path)
    image = np.random.default_rng(2).normal(size=shape[::-1]).astype(np.float32)
    affine = np.diag([.2, .2, .2, 1.])
    affine[:3, 3] = (1., 2., 3.)
    nii = nib.Nifti1Image(image, affine)
    nii.header.set_xyzt_units("mm")
    source = tmp_path / "new volume.nii.gz"
    nib.save(nii, source)
    raw, meta = read_nifti(source)
    volume = VolumeCase("new", "group", raw, None, meta)
    network = ConstantLogits().train()
    result = predict_volume(network, volume, plans=plans, config=config, stats=stats, device="cpu")
    assert network.training and network.decoder.deep_supervision
    assert result.probability.shape == shape and result.mask.shape == shape
    assert np.allclose(result.probability, .5, atol=1e-6)
    assert np.count_nonzero(result.mask) == 0
    assert np.allclose(result.meta.affine, affine, atol=1e-7, rtol=0)
    prob_path, mask_path = write_prediction(result, case_id="new", output=tmp_path / "predictions")
    saved_prob, saved_meta = read_nifti(prob_path)
    saved_mask, _ = read_nifti(mask_path)
    assert np.allclose(saved_prob, result.probability)
    assert np.array_equal(saved_mask, result.mask)
    assert np.array_equal(saved_meta.affine, result.meta.affine)
    assert network.observed is not None and network.observed.shape[1] == 2


def test_metrics_match_legacy_case_and_group_definitions(tmp_path):
    from nnunetv2.phase3.inference import Prediction
    _, meta = read_nifti(synthetic_dataset(tmp_path / "input")[0].parent / "case_0_0000.nii.gz")
    mask = np.zeros((20, 18, 16), dtype=np.uint8)
    target = mask.copy()
    mask[2:5, 3:6, 4:6] = 1
    target[3:6, 3:6, 4:6] = 1
    prediction = Prediction(mask.astype(np.float32), mask, meta, {})
    old = old_metrics(prediction, target, case_id="c", group_id="g")
    current = compute_case_metrics(prediction, target, case_id="c", group_id="g")
    assert asdict(current) == asdict(old)
    assert aggregate_group_metrics([current])["mean"]["dice"] == current.dice


def test_predictor_failure_restores_training_and_ds_state(tmp_path, monkeypatch):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    plans, config, stats = _setup(tmp_path)
    mp = tmp_path / "input" / "case_0_0000.nii.gz"
    raw, meta = read_nifti(mp)
    volume = VolumeCase("case_0", "g0", raw, None, meta)
    network = ConstantLogits().train()
    def explode(*args, **kwargs):
        raise RuntimeError("injected")
    monkeypatch.setattr(nnUNetPredictor, "predict_sliding_window_return_logits", explode)
    with pytest.raises(RuntimeError, match="injected"):
        predict_volume(network, volume, plans=plans, config=config, stats=stats, device="cpu")
    assert network.training and network.decoder.deep_supervision


def test_experimental_prediction_provenance_rejects_tampering_and_external(tmp_path):
    from nnunetv2.phase3.inference import Prediction
    mp, sp = synthetic_dataset(tmp_path / "input")
    manifest, split = load_manifest(mp, require_labels=True), load_splits(sp)[0]
    case = select_cases(manifest, split, "test")[0]
    image, meta = read_nifti(case.image)
    probability = np.zeros_like(image)
    prob_path, mask_path = write_prediction(
        Prediction(probability, probability.astype(np.uint8), meta, {}),
        case_id=case.case_id, output=tmp_path / "prediction")
    checkpoint = tmp_path / "checkpoint_best.pth"
    checkpoint.write_bytes(b"checkpoint")
    record = {
        "schema_version": "phase3_predictions_v1", "mode": "experimental",
        "source_fold": 0, "split_name": "test", "split_digest": json_digest(split.to_dict()),
        "manifest_digest": json_digest(manifest.to_dict()),
        "case_groups": {case.case_id: case.group_id},
        "projection_sources": {case.case_id: case.projection_source},
        "image_sha256": {case.case_id: file_sha256(case.image)},
        "label_sha256": {case.case_id: file_sha256(case.label)},
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "outputs": {case.case_id: {"probability_path": str(prob_path),
                                   "mask_path": str(mask_path),
                                   "probability_sha256": file_sha256(prob_path),
                                   "mask_sha256": file_sha256(mask_path)}},
    }
    validate_prediction_provenance(record, manifest, split, source_fold=0, split_name="test")
    external = dict(record, mode="external")
    with pytest.raises(ValueError):
        validate_prediction_provenance(external, manifest, split, source_fold=0, split_name="test")
    original_mask_bytes = mask_path.read_bytes()
    mask_path.write_bytes(original_mask_bytes + b"tampered")
    with pytest.raises(ValueError, match="output hash"):
        validate_prediction_provenance(record, manifest, split, source_fold=0, split_name="test")
    mask_path.write_bytes(original_mask_bytes)
    shifted = nib.load(str(case.label))
    affine = shifted.affine.copy()
    affine[0, 3] += 1.
    nib.save(nib.Nifti1Image(np.asanyarray(shifted.dataobj), affine,
                            header=shifted.header.copy()), case.label)
    record["label_sha256"][case.case_id] = file_sha256(case.label)
    with pytest.raises(ValueError, match="grid"):
        validate_prediction_provenance(record, manifest, split, source_fold=0, split_name="test")
