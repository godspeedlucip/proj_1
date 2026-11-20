"""Epoch-level TensorBoard metrics and representative validation slices."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class PreviewSlice:
    case_id: str
    group_id: str
    z_index: int
    image: np.ndarray
    target: np.ndarray
    predicted: np.ndarray
    target_voxels: int
    predicted_voxels: int


class ValidationPreviewCollector:
    """Choose one informative slice per distinct volume before taking extras."""

    def __init__(self, *, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("Preview limit must be positive")
        self.limit = limit
        self._cases: list[list[PreviewSlice]] = []

    def add(self, case_id, volume, prediction) -> None:
        image, target, predicted = volume.image_raw, volume.target, prediction.mask
        if (target is None or image.shape != target.shape or target.shape != predicted.shape
                or image.ndim != 3):
            raise ValueError("Validation preview requires aligned 3D image, target and prediction")
        target_counts = np.count_nonzero(target, axis=(1, 2))
        predicted_counts = np.count_nonzero(predicted, axis=(1, 2))
        midpoint = (image.shape[0] - 1) / 2
        # An early model may predict nearly the whole image. Keep the preview
        # on labeled anatomy even when false positives dominate other slices.
        indices = np.lexsort((np.abs(np.arange(image.shape[0]) - midpoint),
                              -predicted_counts, -target_counts))[:self.limit]
        case_slices = []
        for z in indices:
            z = int(z)
            case_slices.append(PreviewSlice(
                case_id, volume.group_id, z,
                np.array(image[z], copy=True), np.array(target[z], copy=True),
                np.array(predicted[z], copy=True),
                int(target_counts[z]), int(predicted_counts[z]),
            ))
        self._cases.append(case_slices)

    def selected(self) -> list[PreviewSlice]:
        selected = []
        for rank in range(self.limit):
            for case_slices in self._cases:
                if rank < len(case_slices):
                    selected.append(case_slices[rank])
                    if len(selected) == self.limit:
                        return selected
        return selected


def _render_preview(preview: PreviewSlice) -> np.ndarray:
    image = preview.image.astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError("Validation preview image contains nonfinite values")
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo:
        grayscale = np.zeros(image.shape, dtype=np.uint8)
    else:
        grayscale = np.rint(np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    input_panel = np.repeat(grayscale[..., None], 3, axis=-1)
    target_panel = np.zeros_like(input_panel)
    target_panel[..., 0] = np.asarray(preview.target != 0, dtype=np.uint8) * 255
    predicted_panel = np.zeros_like(input_panel)
    predicted_panel[..., 1] = np.asarray(preview.predicted != 0, dtype=np.uint8) * 255
    overlay = input_panel.copy()
    target = preview.target != 0
    predicted = preview.predicted != 0
    overlay[target, 0] = 255
    overlay[target, 2] = 0
    overlay[predicted, 1] = 255
    overlay[predicted, 2] = 0
    return np.concatenate((input_panel, target_panel, predicted_panel, overlay), axis=1)


class Phase3TensorBoard:
    def __init__(self, output_folder: Path, *, purge_step: int) -> None:
        self.output_folder = Path(output_folder)
        event_dir = self.output_folder / "tensorboard"
        # TensorBoard reads event files by filename. A same-second restart can
        # reverse their order when the writer's numeric suffix gains a digit.
        timestamps = []
        for path in event_dir.glob("events.out.tfevents.*"):
            parts = path.name.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                timestamps.append(int(parts[3]))
        if timestamps and max(timestamps) >= int(time.time()):
            time.sleep(max(timestamps) + 1 - time.time())
        self.writer = SummaryWriter(log_dir=str(event_dir),
                                    purge_step=purge_step)

    def log_step(self, step: int, values: dict[str, float]) -> None:
        for tag, value in values.items():
            if value is not None and np.isfinite(value):
                self.writer.add_scalar(tag, float(value), step)

    def log_epoch(self, epoch: int, summary: dict, previews: list[PreviewSlice],
                  *, training_scalars: dict[str, float] | None = None,
                  step: int | None = None) -> Path:
        if epoch < 1:
            raise ValueError("Epoch number must start at 1")
        event_step = epoch if step is None else step
        if event_step < 1:
            raise ValueError("TensorBoard step must be positive")
        for tag, value in (training_scalars or {}).items():
            if value is not None and np.isfinite(value):
                self.writer.add_scalar(tag, float(value), event_step)
        for section, prefix in (("mean", "validation/full"), ("std", "validation/std")):
            for metric, value in summary.get(section, {}).items():
                if value is not None and np.isfinite(value):
                    self.writer.add_scalar(f"{prefix}/{metric}", float(value), event_step)
        for group, metrics in summary.get("groups", {}).items():
            for metric, value in metrics.items():
                if value is not None and np.isfinite(value):
                    self.writer.add_scalar(f"validation/group/{group}/{metric}", float(value), event_step)

        preview_dir = self.output_folder / "validation_previews" / f"epoch_{epoch:04d}"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_rows = []
        for index, preview in enumerate(previews[:3], start=1):
            rgb = _render_preview(preview)
            filename = f"{index}_{preview.case_id}_z{preview.z_index:04d}.png"
            Image.fromarray(rgb).save(preview_dir / filename)
            self.writer.add_image(f"validation/preview/{index}_{preview.case_id}",
                                  np.moveaxis(rgb, -1, 0), event_step, dataformats="CHW")
            preview_rows.append({"case_id": preview.case_id, "group_id": preview.group_id,
                                 "z_index": preview.z_index,
                                 "target_voxels": preview.target_voxels,
                                 "predicted_voxels": preview.predicted_voxels,
                                 "path": str(preview_dir / filename)})
        report = dict(summary, epoch=epoch, previews=preview_rows)
        report_dir = self.output_folder / "validation_metrics"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"epoch_{epoch:04d}.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        self.writer.flush()
        return report_path

    def close(self) -> None:
        self.writer.close()
