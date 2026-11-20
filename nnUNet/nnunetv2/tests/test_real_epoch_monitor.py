import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from nnunetv2.training.nnUNetTrainer.variants.marker_guided.real_epoch_monitor import (
    RealEpochMonitor,
)


def _fixture(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    records = []
    for number in range(1, 4):
        image = np.full((16, 16), number, dtype=np.float32)
        source = frames / f"frame_{number:04d}.tif"
        tifffile.imwrite(source, image)
        records.append({
            "source": str(source),
            "marker_xy": [[3.0, 4.0], [12.0, 11.0]] if number != 3 else None,
            "status": "two_candidates" if number != 3 else "no_candidates",
            "edge_candidate": number == 2,
        })
    package = tmp_path / "predictions.json"
    package.write_text(json.dumps({"image_shape": [16, 16], "records": records}))
    return frames, package


def test_real_monitor_builds_matched_four_channel_inputs_and_epoch_outputs(tmp_path):
    frames, package = _fixture(tmp_path)
    monitor = RealEpochMonitor(frames, package, tmp_path / "monitor", preview_names=("frame_0001", "frame_0003"))
    prepared = monitor.prepare()
    assert len(prepared) == 3
    assert tifffile.imread(tmp_path / "monitor/inputs/frame_0001_0000.tif")[0, 0] == 1
    assert tifffile.imread(tmp_path / "monitor/inputs/frame_0001_0001.tif")[4, 3] > 0.9
    assert np.all(tifffile.imread(tmp_path / "monitor/inputs/frame_0002_0003.tif") == 1)
    assert np.all(tifffile.imread(tmp_path / "monitor/inputs/frame_0003_0003.tif") == 0)

    seen = []

    def segment(paths):
        seen.append([str(path) for path in paths])
        return np.eye(16, dtype=np.uint8)

    with SummaryWriter(str(tmp_path / "tensorboard")) as writer:
        summary = monitor.run_epoch(4, segment, writer)
    assert len(seen) == 3
    assert summary["frame_count"] == 3
    assert summary["valid_marker_count"] == 2
    assert summary["mean_foreground_fraction"] == pytest.approx(1 / 16)
    assert len(list((tmp_path / "monitor/epoch_0004").glob("frame_*.tif"))) == 3
    assert tifffile.imread(tmp_path / "monitor/epoch_0004/frame_0002.tif").shape == (16, 16)
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("real/mean_foreground_fraction")[0].step == 4
    assert events.Scalars("real/valid_marker_fraction")[0].value == pytest.approx(2 / 3)
    assert events.Images("real/frame_0001/overlay")[0].step == 4


def test_real_monitor_rejects_missing_prediction(tmp_path):
    frames, package = _fixture(tmp_path)
    data = json.loads(package.read_text())
    data["records"].pop()
    package.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing"):
        RealEpochMonitor(frames, package, tmp_path / "monitor").prepare()


def test_real_monitor_can_segment_one_named_frame_per_epoch(tmp_path):
    frames, package = _fixture(tmp_path)
    monitor = RealEpochMonitor(frames, package, tmp_path / "monitor",
                               selected_names=("frame_0002",), preview_names=("frame_0002",))
    prepared = monitor.prepare()
    assert [frame.name for frame in prepared] == ["frame_0002"]
    assert len(list((tmp_path / "monitor/inputs").glob("*.tif"))) == 4
    with SummaryWriter(str(tmp_path / "tensorboard")) as writer:
        summary = monitor.run_epoch(1, lambda _: np.ones((16, 16), dtype=np.uint8), writer)
    assert summary["frame_count"] == 1
    assert len(list((tmp_path / "monitor/epoch_0001").glob("frame_*.tif"))) == 1


def test_new_training_rejects_real_predictions_from_a_previous_run(tmp_path):
    frames, package = _fixture(tmp_path)
    previous = tmp_path / "monitor/epoch_0001"
    previous.mkdir(parents=True)
    (previous / "summary.json").write_text(json.dumps({"epoch": 1}))

    with pytest.raises(FileExistsError, match="previous run"):
        RealEpochMonitor(frames, package, tmp_path / "monitor",
                         require_empty_epochs=True).prepare()
