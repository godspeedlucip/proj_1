"""Per-epoch segmentation of unlabeled real frames using frozen Phase 1 marker predictions."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

import numpy as np
import tifffile

from tools.marker_guided.build_prior_channels import PriorConfig, build_marker_channels


@dataclass(frozen=True)
class RealFrame:
    name: str
    source: Path
    inputs: tuple[Path, Path, Path, Path]
    marker_xy: tuple[tuple[float, float], tuple[float, float]] | None
    edge_candidate: bool


class RealEpochMonitor:
    def __init__(self, frames_dir: str | Path, predictions_json: str | Path, output_dir: str | Path,
                 *, prior_config: PriorConfig | None = None, preview_names: tuple[str, ...] | None = None,
                 selected_names: tuple[str, ...] | None = None,
                 require_empty_epochs: bool = False):
        self.frames_dir = Path(frames_dir).resolve()
        self.predictions_json = Path(predictions_json).resolve()
        self.output_dir = Path(output_dir)
        self.prior_config = prior_config or PriorConfig()
        self.preview_names = preview_names
        self.selected_names = selected_names
        self.require_empty_epochs = require_empty_epochs
        self.frames: list[RealFrame] = []

    def prepare(self) -> list[RealFrame]:
        if self.require_empty_epochs and any(self.output_dir.glob("epoch_*")):
            raise FileExistsError(
                f"real predictions from a previous run exist in {self.output_dir}; "
                "archive them or use a new results directory before fresh training")
        if not self.frames_dir.is_dir() or not self.predictions_json.is_file():
            raise FileNotFoundError("real frames directory and Phase 1 predictions.json must exist")
        source_files = sorted(self.frames_dir.glob("frame_*.tif"))
        package_bytes = self.predictions_json.read_bytes()
        package = json.loads(package_bytes)
        records = package.get("records")
        if not source_files or not isinstance(records, list):
            raise ValueError("real frames and prediction records must be nonempty")
        by_source = {}
        for record in records:
            source = Path(record["source"]).resolve()
            if source in by_source:
                raise ValueError(f"duplicate marker prediction for {source}")
            by_source[source] = record
        sources = {path.resolve() for path in source_files}
        if sources != set(by_source):
            missing = sorted(str(path) for path in sources - set(by_source))
            extra = sorted(str(path) for path in set(by_source) - sources)
            raise ValueError(f"missing predictions: {missing[:3]}; extra predictions: {extra[:3]}")
        if self.selected_names is not None:
            selected = set(self.selected_names)
            if not selected or len(selected) != len(self.selected_names):
                raise ValueError("selected_names must contain distinct frame names")
            if not selected.issubset({path.stem for path in source_files}):
                raise ValueError("selected_names contains an unknown real frame")
            source_files = [path for path in source_files if path.stem in selected]

        inputs_dir = self.output_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(package_bytes).hexdigest()
        manifest_path = inputs_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("predictions_sha256") != digest or manifest.get("frames_dir") != str(self.frames_dir):
                raise ValueError("cached real inputs belong to a different marker package or frame directory")

        expected_shape = tuple(package.get("image_shape", ()))
        frames = []
        for source in source_files:
            name = source.stem
            record = by_source[source.resolve()]
            with tifffile.TiffFile(source) as image_file:
                shape = tuple(image_file.series[0].shape)
            if len(shape) != 2 or (expected_shape and shape != expected_shape):
                raise ValueError(f"real image shape mismatch: {source}: {shape}")
            marker_xy = record.get("marker_xy")
            if record.get("status") == "two_candidates":
                if marker_xy is None:
                    raise ValueError(f"two_candidates has no marker_xy: {source}")
                pair_status = "uncertain" if record.get("edge_candidate") else "reliable"
                view = {"pair_status": pair_status, "marker_xy": marker_xy, "error_radius_px": [0.0, 0.0]}
                points = tuple(tuple(float(value) for value in point) for point in marker_xy)
            else:
                if marker_xy is not None:
                    raise ValueError(f"invalid pair unexpectedly has marker_xy: {source}")
                view = {"pair_status": "invalid"}
                points = None
            files = tuple(inputs_dir / f"{name}_{channel:04d}.tif" for channel in range(4))
            if not files[0].exists():
                files[0].symlink_to(source)
            elif files[0].resolve() != source.resolve():
                raise ValueError(f"cached projection points to wrong source: {files[0]}")
            if not all(path.is_file() for path in files[1:]):
                channels = build_marker_channels(view, shape, self.prior_config)
                for path, channel in zip(files[1:], channels):
                    tifffile.imwrite(path, channel, compression="zlib")
            frames.append(RealFrame(name, source, files, points, bool(record.get("edge_candidate"))))

        available = {frame.name for frame in frames}
        if self.preview_names is None:
            self.preview_names = tuple(frame.name for frame in frames[::max(1, len(frames) // 6)][:6])
        if not set(self.preview_names).issubset(available):
            raise ValueError("preview_names must be names of real frames")
        manifest_path.write_text(json.dumps({"frames_dir": str(self.frames_dir),
                                             "predictions_json": str(self.predictions_json),
                                             "predictions_sha256": digest,
                                             "frame_count": len(frames),
                                             "valid_marker_count": sum(frame.marker_xy is not None for frame in frames),
                                             "edge_candidate_count": sum(frame.edge_candidate for frame in frames),
                                             "prior_config": vars(self.prior_config)}, indent=2))
        self.frames = frames
        return frames

    @staticmethod
    def _preview(frame: RealFrame, mask: np.ndarray) -> np.ndarray:
        image = tifffile.imread(frame.source).astype(np.float32)
        low, high = np.percentile(image, (1, 99))
        gray = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
        rgb = np.repeat(gray[None], 3, axis=0)
        rgb[0, mask > 0] = 1
        rgb[1, mask > 0] *= 0.35
        rgb[2, mask > 0] *= 0.35
        if frame.marker_xy is not None:
            yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
            for x, y in frame.marker_xy:
                circle = (xx - x) ** 2 + (yy - y) ** 2 <= 8 ** 2
                rgb[:, circle] = np.array([0, 1, 1])[:, None]
        step = max(1, int(np.ceil(max(image.shape) / 384)))
        return rgb[:, ::step, ::step]

    def run_epoch(self, epoch: int, segment: Callable[[tuple[Path, Path, Path, Path]], np.ndarray], writer) -> dict:
        if epoch < 1:
            raise ValueError("epoch must be one-based")
        if not self.frames:
            self.prepare()
        final = self.output_dir / f"epoch_{epoch:04d}"
        if final.is_dir() and (final / "summary.json").is_file():
            return json.loads((final / "summary.json").read_text())
        partial = self.output_dir / f"epoch_{epoch:04d}.incomplete"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        started = monotonic()
        pixel_counts = []
        total_pixels = 0
        previews = []
        for frame in self.frames:
            with tifffile.TiffFile(frame.source) as image_file:
                shape = tuple(image_file.series[0].shape)
            total_pixels += int(np.prod(shape))
            mask = np.asarray(segment(frame.inputs))
            if mask.shape != shape or not np.isin(mask, (0, 1)).all():
                raise ValueError(f"segmentation for {frame.name} must be binary and have shape {shape}")
            mask = mask.astype(np.uint8, copy=False)
            tifffile.imwrite(partial / f"{frame.name}.tif", mask, compression="zlib")
            pixel_counts.append(int(mask.sum()))
            if frame.name in self.preview_names:
                previews.append((frame, self._preview(frame, mask)))
        summary = {
            "epoch": epoch,
            "frame_count": len(self.frames),
            "valid_marker_count": sum(frame.marker_xy is not None for frame in self.frames),
            "edge_candidate_count": sum(frame.edge_candidate for frame in self.frames),
            "mean_foreground_fraction": sum(pixel_counts) / total_pixels,
            "positive_frame_fraction": sum(count > 0 for count in pixel_counts) / len(pixel_counts),
            "mean_foreground_pixels": float(np.mean(pixel_counts)),
            "seconds": monotonic() - started,
            "source_predictions": str(self.predictions_json),
        }
        (partial / "summary.json").write_text(json.dumps(summary, indent=2))
        partial.replace(final)
        writer.add_scalar("real/mean_foreground_fraction", summary["mean_foreground_fraction"], epoch)
        writer.add_scalar("real/positive_frame_fraction", summary["positive_frame_fraction"], epoch)
        writer.add_scalar("real/valid_marker_fraction", summary["valid_marker_count"] / len(self.frames), epoch)
        writer.add_scalar("real/edge_candidate_fraction", summary["edge_candidate_count"] / len(self.frames), epoch)
        writer.add_scalar("real/inference_seconds", summary["seconds"], epoch)
        for frame, preview in previews:
            writer.add_image(f"real/{frame.name}/overlay", preview, epoch)
        writer.flush()
        return summary
