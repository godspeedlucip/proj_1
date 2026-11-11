"""Public-command smoke flow on isolated synthetic source groups."""

import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import torch

from nnunetv2.phase3.data_conversion import PreparedFold
from nnunetv2.phase3.launch import model_folder
from nnunetv2.phase3.nifti_io import read_nifti
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.provenance import file_sha256, validate_prediction_provenance
import pytest
from .fixtures import synthetic_dataset


ROOT = Path(__file__).resolve().parents[4]


def _cli(*args):
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / "phase_3" / "nnunet"),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-m", "nnunetv2.phase3.cli", *map(str, args)],
                            cwd=ROOT, env=environment, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result.stdout


def test_public_cli_prepare_train_resume_predict_evaluate(tmp_path):
    source = tmp_path / "中文 空格" / "source"
    manifest, splits = synthetic_dataset(source, shape=(17, 19, 21))
    # One held-out source exceeds the 32^3 smoke patch and remains outside training statistics.
    large_shape = (33, 37, 35)
    image = np.zeros(large_shape, np.float32)
    label = np.zeros(large_shape, np.uint8)
    label[3:25, 15:17, 18:20] = 1
    image += label
    for name, array in (("case_3_0000", image), ("case_3", label)):
        nii = nib.Nifti1Image(array, np.diag([.2, .2, .2, 1.]))
        nii.header.set_xyzt_units("mm")
        nib.save(nii, source / f"{name}.nii.gz")
    output = tmp_path / "中文 空格" / "prepared"
    _cli("validate-data", "--manifest", manifest, "--splits", splits, "--run-kind", "smoke")
    _cli("prepare", "--manifest", manifest, "--splits", splits, "--source-fold", 0,
         "--dataset-id", 914, "--config", "smoke", "--output", output)
    record = PreparedFold.from_dict(json.loads((output / "prepared.json").read_text(encoding="utf-8")))
    assert record.nnunet_fold == 0 and set(record.source_cases) == set(record.train_ids + record.validation_ids)
    fingerprint = json.loads((record.preprocessed_dir / "dataset_fingerprint.json").read_text(encoding="utf-8"))
    assert "case_3" not in fingerprint["case_ids"]
    prepared_path = output / "prepared.json"
    _cli("train", "--prepared", prepared_path, "--config", "smoke", "--device", "cpu")
    fold = model_folder(record)
    best = fold / "checkpoint_best.pth"
    assert best.is_file() and (fold / "checkpoint_latest.pth").is_file()
    before = torch.load(best, map_location="cpu", weights_only=False)
    assert before["phase3_state"]["data_digest"] == record.data_digest
    _cli("train", "--prepared", prepared_path, "--config", "smoke", "--device", "cpu", "--resume")
    after = torch.load(fold / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    assert after["current_epoch"] == 2 and after["phase3_state"]["global_step"] == 4
    model_dir = fold.parent
    predictions = tmp_path / "中文 空格" / "predictions"
    _cli("predict", "--model-dir", model_dir, "--mode", "experimental",
         "--manifest", manifest, "--splits", splits, "--source-fold", 0,
         "--split-name", "test", "--device", "cpu", "--output", predictions)
    probability, prob_meta = read_nifti(predictions / "case_3_prob.nii.gz")
    mask, mask_meta = read_nifti(predictions / "case_3_mask.nii.gz")
    assert probability.shape == (35, 37, 33) and mask.shape == probability.shape
    assert np.array_equal(prob_meta.affine, mask_meta.affine)
    assert (predictions / "prediction_run.json").is_file()
    metrics = tmp_path / "中文 空格" / "metrics"
    _cli("evaluate", "--predictions", predictions, "--manifest", manifest,
         "--splits", splits, "--source-fold", 0, "--split-name", "test",
         "--output", metrics)
    summary = json.loads((metrics / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 1 and summary["group_count"] == 1
    assert (metrics / "metrics_per_case.csv").is_file()
    external = tmp_path / "中文 空格" / "external"
    external_image = source / "case_0_0000.nii.gz"
    _cli("predict", "--model-dir", model_dir, "--mode", "external",
         "--image", external_image, "--case-id", "external_0", "--device", "cpu",
         "--output", external)
    external_record = json.loads((external / "prediction_run.json").read_text(encoding="utf-8"))
    assert external_record["mode"] == "external"
    assert external_record["input_sha256"] == file_sha256(external_image)
    assert external_record["checkpoint_sha256"] == file_sha256(best)
    assert external_record["outputs"]["probability_sha256"] == file_sha256(external / "external_0_prob.nii.gz")
    with pytest.raises(ValueError):
        validate_prediction_provenance(external_record, load_manifest(manifest, require_labels=True),
                                       load_splits(splits)[0], source_fold=0, split_name="test")
