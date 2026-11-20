"""Qualitative, full-frame regression checks using frozen Phase 1 marker points."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import tifffile
from tqdm import tqdm
from .predict import predict_maps


@dataclass(frozen=True)
class RealFrame:
    name: str
    source: Path
    marker_xy: tuple[tuple[float, float], tuple[float, float]]


class RealRegressionMonitor:
    def __init__(self, frames_dir: str | Path, predictions_json: str | Path,
                 output_dir: str | Path, selected_names: tuple[str, ...]):
        if not selected_names:
            raise ValueError("select at least one real frame")
        self.frames_dir = Path(frames_dir).resolve()
        self.predictions_json = Path(predictions_json).resolve()
        self.selected_names = selected_names
        self.output_dir = Path(output_dir)
        self.frames = []

    def prepare(self) -> None:
        if not self.frames_dir.is_dir() or not self.predictions_json.is_file():
            raise FileNotFoundError("real frames and Phase 1 marker JSON must exist")
        package = json.loads(self.predictions_json.read_text(encoding="utf-8"))
        records = {Path(record["source"]).resolve(): record
                   for record in package.get("records", [])}
        if len(records) != len(package.get("records", [])):
            raise ValueError("duplicate real frame marker records")
        frames = []
        for name in self.selected_names:
            source = (self.frames_dir / f"{name}.tif").resolve()
            if not source.is_file() or source not in records:
                raise FileNotFoundError(f"real frame or marker record missing: {name}")
            record = records[source]
            marker_xy = record.get("marker_xy")
            if record.get("status") != "two_candidates" or marker_xy is None or len(marker_xy) != 2:
                raise ValueError(f"selected real frame has no marker pair: {name}")
            points = tuple(tuple(float(value) for value in point) for point in marker_xy)
            if any(len(point) != 2 for point in points):
                raise ValueError(f"invalid marker coordinates: {name}")
            frames.append(RealFrame(name, source, points))
        self.frames = frames

    @staticmethod
    def _preview(frame, image: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        low, high = np.percentile(image, (1, 99))
        gray = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
        rgb = np.repeat(gray[None], 3, axis=0)
        alpha = 0.75 * np.clip(prediction, 0, 1)
        rgb[0] = rgb[0] * (1 - alpha) + alpha
        rgb[1:] *= (1 - alpha)[None]
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        for x, y in frame.marker_xy:
            circle = (xx - x) ** 2 + (yy - y) ** 2 <= 8 ** 2
            rgb[:, circle] = np.array([0, 1, 1])[:, None]
        step = max(1, int(np.ceil(max(image.shape) / 512)))
        return rgb[:, ::step, ::step].astype(np.float32)

    def _write_tensorboard(self, epoch: int, summary: dict, writer) -> None:
        writer.add_scalar("real/inference_seconds", summary["seconds"], epoch)
        for frame in self.frames:
            epoch_dir = self.output_dir / f"epoch_{epoch:04d}"
            prediction = tifffile.imread(epoch_dir / f"{frame.name}.tif")
            image = tifffile.imread(frame.source).astype(np.float32)
            writer.add_image(f"real/{frame.name}/overlay",
                             self._preview(frame, image, prediction), epoch)
            writer.add_image(f"real/{frame.name}/prediction",
                             prediction[None, ::max(1, image.shape[0] // 512),
                                        ::max(1, image.shape[1] // 512)], epoch)
            stats = summary["frames"][frame.name]
            writer.add_scalar(f"real/{frame.name}/mean_intensity", stats["mean"], epoch)
            writer.add_scalar(f"real/{frame.name}/max_intensity", stats["max"], epoch)
            mask_path = epoch_dir / "masks" / f"{frame.name}.tif"
            if mask_path.is_file():
                mask = tifffile.imread(mask_path).astype(np.float32)
                writer.add_image(f"real/{frame.name}/mask", mask[None, ::max(1, image.shape[0] // 512),
                                                          ::max(1, image.shape[1] // 512)], epoch)
                writer.add_image(f"real/{frame.name}/mask_overlay",
                                 self._preview(frame, image, mask), epoch)
                depth = tifffile.imread(epoch_dir / "conditional_depth" / f"{frame.name}.tif")
                writer.add_image(f"real/{frame.name}/conditional_depth",
                                 depth[None, ::max(1, image.shape[0] // 512),
                                             ::max(1, image.shape[1] // 512)], epoch)
                writer.add_scalar(f"real/{frame.name}/mask_fraction", stats["mask_fraction"], epoch)
        writer.flush()

    def run_epoch(self, epoch: int, model, writer, device: str) -> dict:
        if not self.frames:
            self.prepare()
        final = self.output_dir / f"epoch_{epoch:04d}"
        summary_path = final / "summary.json"
        tensorboard_done = final / ".tensorboard_complete"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if {path.stem for path in final.glob("frame_*.tif")} != {frame.name for frame in self.frames}:
                raise ValueError(f"incomplete saved real predictions: {final}")
            if not tensorboard_done.is_file():
                self._write_tensorboard(epoch, summary, writer)
                tensorboard_done.write_text("complete\n", encoding="utf-8")
            return summary
        partial = self.output_dir / f"epoch_{epoch:04d}.incomplete"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        started = monotonic()
        stats = {}
        try:
            for frame in tqdm(self.frames, desc=f"Real epoch {epoch}", unit="frame",
                              file=sys.stdout, dynamic_ncols=True, leave=True,
                              mininterval=0.5 if sys.stdout.isatty() else 5.0):
                image = tifffile.imread(frame.source).astype(np.float32)
                maps = predict_maps(model, image, frame.marker_xy, device=device)
                prediction = maps["projection"]
                if prediction.shape != image.shape or not np.isfinite(prediction).all():
                    raise ValueError(f"invalid full-frame prediction for {frame.name}")
                tifffile.imwrite(partial / f"{frame.name}.tif", prediction,
                                 compression="zlib", compressionargs={"level": 1})
                stats[frame.name] = {"mean": float(prediction.mean()),
                                     "max": float(prediction.max()),
                                     "fraction_above_0_1": float((prediction > 0.1).mean()),
                                     "marker_xy": frame.marker_xy,
                                     "source": str(frame.source)}
                if "mask_probability" in maps:
                    mask = (maps["mask_probability"] >= 0.5).astype(np.uint8)
                    for folder in ("masks", "mask_probabilities", "conditional_depth"):
                        (partial / folder).mkdir(exist_ok=True)
                    tifffile.imwrite(partial / "masks" / f"{frame.name}.tif", mask,
                                     compression="zlib", compressionargs={"level": 1})
                    tifffile.imwrite(partial / "mask_probabilities" / f"{frame.name}.tif",
                                     maps["mask_probability"], compression="zlib",
                                     compressionargs={"level": 1})
                    tifffile.imwrite(partial / "conditional_depth" / f"{frame.name}.tif",
                                     maps["conditional_depth"], compression="zlib",
                                     compressionargs={"level": 1})
                    stats[frame.name]["mask_fraction"] = float(mask.mean())
            summary = {"epoch": epoch, "frame_count": len(self.frames),
                       "seconds": monotonic() - started,
                       "source_predictions": str(self.predictions_json),
                       "frames": stats,
                       "note": "real frames have no projection ground truth; outputs are qualitative"}
            (partial / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            partial.rename(final)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        self._write_tensorboard(epoch, summary, writer)
        tensorboard_done.write_text("complete\n", encoding="utf-8")
        return summary
