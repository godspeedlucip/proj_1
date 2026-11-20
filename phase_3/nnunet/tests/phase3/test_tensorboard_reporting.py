"""Epoch metrics and representative full-volume validation slices."""

import json

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nnunetv2.phase3.inference import Prediction
from nnunetv2.phase3.nifti_io import VolumeCase
from nnunetv2.phase3.tensorboard_reporting import Phase3TensorBoard, ValidationPreviewCollector


def test_three_distinct_validation_volumes_write_images_and_full_metrics(tmp_path):
    collector = ValidationPreviewCollector(limit=3)
    for case_number in range(3):
        image = np.arange(6 * 8 * 9, dtype=np.float32).reshape(6, 8, 9)
        target = np.zeros_like(image, dtype=np.uint8)
        target[case_number + 1, 3:5, 4:6] = 1
        mask = target.copy()
        probability = mask.astype(np.float32)
        volume = VolumeCase(f"case_{case_number}", "stent2", image, target, None)
        prediction = Prediction(probability, mask, None, {})
        collector.add(volume.case_id, volume, prediction)

    previews = collector.selected()
    assert len(previews) == 3
    assert [preview.case_id for preview in previews] == ["case_0", "case_1", "case_2"]
    assert [preview.z_index for preview in previews] == [1, 2, 3]

    summary = {
        "case_count": 3, "group_count": 1,
        "mean": {"dice": 1.0, "iou": 1.0, "mse": 0.0, "mae": 0.0},
        "std": {"dice": 0.0, "iou": 0.0, "mse": 0.0, "mae": 0.0},
        "groups": {"stent2": {"dice": 1.0, "iou": 1.0, "mse": 0.0, "mae": 0.0}},
        "per_case": [],
    }
    logger = Phase3TensorBoard(tmp_path, purge_step=1)
    logger.log_step(1, {"train/step/segmentation": 0.3, "train/step/projection": 0.1})
    logger.log_epoch(1, summary, previews,
                     training_scalars={"train/loss": 0.4, "validation/patch_loss": 0.2,
                                       "train/learning_rate": 0.01})
    logger.close()

    files = sorted((tmp_path / "validation_previews/epoch_0001").glob("*.png"))
    assert len(files) == 3
    report = json.loads((tmp_path / "validation_metrics/epoch_0001.json").read_text())
    assert report["mean"]["mse"] == 0.0
    assert len(report["previews"]) == 3

    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    assert len(events.Tags()["images"]) == 3
    assert events.Scalars("validation/full/dice")[0].value == 1.0
    assert events.Scalars("validation/full/mse")[0].value == 0.0
    assert events.Scalars("train/loss")[0].value > 0
    assert events.Scalars("train/step/projection")[0].value > 0


def test_preview_prefers_labeled_slice_over_large_false_positive_slice():
    collector = ValidationPreviewCollector(limit=1)
    image = np.zeros((6, 8, 9), dtype=np.float32)
    target = np.zeros_like(image, dtype=np.uint8)
    target[3, 3:5, 4:6] = 1
    predicted = np.zeros_like(target)
    predicted[0] = 1
    predicted[3, 3:5, 4:6] = 1
    volume = VolumeCase("case", "stent2", image, target, None)
    collector.add("case", volume, Prediction(predicted.astype(np.float32), predicted, None, {}))
    assert collector.selected()[0].z_index == 3
    assert collector.selected()[0].target_voxels == 4


def test_resume_preserves_completed_epoch_and_step_curves(tmp_path):
    summary = {"mean": {"dice": 0.5}, "std": {}, "groups": {}, "per_case": []}
    logger = Phase3TensorBoard(tmp_path, purge_step=1)
    logger.log_step(1, {"train/step/total_loss": 0.9})
    logger.log_step(250, {"train/step/total_loss": 0.7})
    logger.log_epoch(1, summary, [], step=250)
    logger.close()

    logger = Phase3TensorBoard(tmp_path, purge_step=251)
    logger.log_step(251, {"train/step/total_loss": 0.6})
    logger.log_epoch(2, summary, [], step=500)
    logger.close()

    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    assert [event.step for event in events.Scalars("train/step/total_loss")] == [1, 250, 251]
    assert [event.step for event in events.Scalars("validation/full/dice")] == [250, 500]
