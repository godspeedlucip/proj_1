"""Lazy image/implant TIFF pairs from the Phase 1 physical manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from tools.marker_guided.build_prior_channels import build_marker_channels


@dataclass(frozen=True)
class ProjectionSample:
    image_path: Path
    target_path: Path
    marker_xy: tuple[tuple[float, float], tuple[float, float]]
    group_id: str
    case_id: str
    prior_paths: tuple[Path, Path, Path] | None = None
    pre_normalized_target: bool = False
    source_scale: float | None = None
    mask_path: Path | None = None


def load_samples(manifest_path: str | Path, split: str, fold: str = "fold_0") -> list[ProjectionSample]:
    """Resolve image and original implant projection without copying either."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unsupported split {split!r}")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (None, "phase1_physical_v1"):
        raise ValueError("expected a phase1 physical manifest")
    groups = manifest["folds"][fold]
    group_sets = {name: set(groups[name]) for name in ("train", "val", "test")}
    if any(group_sets[a] & group_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("stent group overlap between train, val or test")
    data_root = path.parent.parent
    samples = []
    for case in manifest["cases"]:
        series = case["group_id"]
        if series not in group_sets[split]:
            continue
        variant = str(case["metadata"]["stent_variant"])
        for view in case["views"]:
            image_path = (path.parent / view["projection_path"]).resolve()
            try:
                filename_angle = float(image_path.stem)
            except ValueError as exc:
                raise ValueError(f"image filename has no numeric angle: {image_path}") from exc
            if not np.isclose(filename_angle, float(view["angle_deg"])):
                raise ValueError(f"image filename angle disagrees with manifest: {image_path}")
            target_path = (data_root / "stent_projections" / series / variant /
                           "projections" / "implant" / image_path.name).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"missing stacked image: {image_path}")
            if not target_path.is_file():
                raise FileNotFoundError(f"missing original implant projection: {target_path}")
            points = view["marker_xy"]
            if len(points) != 2 or any(len(point) != 2 for point in points):
                raise ValueError(f"expected two marker points for {case['case_id']}")
            samples.append(ProjectionSample(image_path, target_path,
                                            (tuple(points[0]), tuple(points[1])),
                                            series, case["case_id"]))
    return samples


def load_exported_samples(directory: str | Path, split: str) -> list[ProjectionSample]:
    """Read only relative paths inside an owned regression data package."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unsupported split {split!r}")
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in ("phase2_projection_regression_v1",
                                              "phase2_projection_regression_v2",
                                              "phase2_mask_depth_v1"):
        raise ValueError("invalid regression dataset schema")
    scale = float(manifest["target_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid target scale in dataset manifest")
    groups = {name: {entry["group_id"] for entry in manifest["samples"] if entry["split"] == name}
              for name in ("train", "val", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("stent group overlap between dataset splits")

    def local_path(value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"nonlocal dataset path: {value}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"dataset path escapes package: {value}")
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset file: {path}")
        return path

    result = []
    for entry in manifest["samples"]:
        if entry["split"] != split:
            continue
        result.append(ProjectionSample(
            local_path(entry["image"]), local_path(entry["target"]),
            (tuple(entry["marker_xy"][0]), tuple(entry["marker_xy"][1])),
            entry["group_id"], entry["case_id"],
            (local_path(entry["point"]), local_path(entry["band"]),
             local_path(entry["validity"])), True, scale,
            local_path(entry["mask"]) if "mask" in entry else None))
    if len(result) != manifest["counts"][split]:
        raise ValueError("dataset manifest count mismatch")
    return result


def compute_target_scale(train_samples: list[ProjectionSample]) -> float:
    """Find one fixed positive scale from unique training projections only."""
    if not train_samples:
        raise ValueError("training samples are empty")
    scales = {sample.source_scale for sample in train_samples}
    if scales != {None}:
        if len(scales) != 1 or None in scales:
            raise ValueError("mixed target scales in training samples")
        return float(next(iter(scales)))
    maximum = 0.0
    for path in {sample.target_path for sample in train_samples}:
        target = tifffile.imread(path)
        if target.ndim != 2 or not np.isfinite(target).all() or np.any(target < 0):
            raise ValueError(f"invalid training target: {path}")
        maximum = max(maximum, float(np.max(target)))
    if maximum <= 0:
        raise ValueError("training projections have no positive intensity")
    return maximum


def _crop_and_pad(image: np.ndarray, y0: int, x0: int, height: int, width: int) -> np.ndarray:
    h, w = image.shape[-2:]
    src_y0, src_x0 = max(0, y0), max(0, x0)
    src_y1, src_x1 = min(h, y0 + height), min(w, x0 + width)
    result = np.zeros((*image.shape[:-2], height, width), dtype=np.float32)
    if src_y1 > src_y0 and src_x1 > src_x0:
        result[..., src_y0 - y0:src_y1 - y0, src_x0 - x0:src_x1 - x0] = (
            image[..., src_y0:src_y1, src_x0:src_x1])
    return result


class ProjectionDataset(Dataset):
    def __init__(self, samples: list[ProjectionSample], patch_size: tuple[int, int],
                 training: bool, foreground_crop_fraction: float = 0.33,
                 target_scale: float = 1.0, full_image: bool = False,
                 return_mask: bool = False):
        if len(patch_size) != 2 or min(patch_size) <= 0:
            raise ValueError("patch_size must contain two positive dimensions")
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("target_scale must be finite and positive")
        self.samples = samples
        self.patch_size = tuple(map(int, patch_size))
        self.training = training
        self.foreground_crop_fraction = foreground_crop_fraction
        self.target_scale = float(target_scale)
        self.full_image = full_image
        self.return_mask = return_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = tifffile.imread(sample.image_path).astype(np.float32)
        target = tifffile.imread(sample.target_path).astype(np.float32)
        if image.ndim != 2 or target.shape != image.shape:
            raise ValueError(f"image/target shape mismatch: {sample.image_path}, {sample.target_path}")
        if not np.isfinite(image).all() or not np.isfinite(target).all() or np.any(target < 0):
            raise ValueError(f"nonfinite image or invalid target: {sample.case_id}")
        if sample.pre_normalized_target:
            if np.any(target > 1):
                raise ValueError(f"pre-normalized target exceeds one: {sample.target_path}")
        else:
            target = np.clip(target / self.target_scale, 0.0, 1.0)
        mask = None
        if self.return_mask:
            if sample.mask_path is None:
                raise ValueError(f"cascade sample has no binary mask: {sample.case_id}")
            mask = tifffile.imread(sample.mask_path)
            if mask.shape != target.shape or not np.array_equal(mask, (target > 0).astype(mask.dtype)):
                raise ValueError(f"binary mask differs from continuous support: {sample.mask_path}")
            mask = mask.astype(np.float32)
        image = (image - image.mean()) / max(float(image.std()), 1e-8)
        if sample.prior_paths is None:
            point, band, validity = build_marker_channels(
                {"pair_status": "reliable", "marker_xy": sample.marker_xy,
                 "error_radius_px": [0.0, 0.0]}, image.shape)
        else:
            point, band, validity = (tifffile.imread(path).astype(np.float32)
                                     for path in sample.prior_paths)
            if any(channel.shape != image.shape for channel in (point, band, validity)):
                raise ValueError(f"image/prior shape mismatch: {sample.image_path}")
        channels = np.stack((image, point, band, validity), axis=0)
        height, width = self.patch_size
        h, w = image.shape
        if self.full_image:
            if h > height or w > width:
                raise ValueError(f"full image exceeds padded canvas: {sample.image_path}")
            y0, x0 = 0, 0
        elif self.training and np.random.random() >= self.foreground_crop_fraction:
            y0 = int(np.random.randint(min(0, h - height), max(0, h - height) + 1))
            x0 = int(np.random.randint(min(0, w - width), max(0, w - width) + 1))
        else:
            midpoint = np.mean(sample.marker_xy, axis=0)
            x0, y0 = int(round(midpoint[0] - width / 2)), int(round(midpoint[1] - height / 2))
            if self.training:
                x0 += int(np.random.randint(-width // 8, width // 8 + 1))
                y0 += int(np.random.randint(-height // 8, height // 8 + 1))
            if width <= w:
                x0 = min(max(x0, 0), w - width)
            if height <= h:
                y0 = min(max(y0, 0), h - height)
        channels = _crop_and_pad(channels, y0, x0, height, width)
        target = _crop_and_pad(target[None], y0, x0, height, width)
        if mask is not None:
            mask = _crop_and_pad(mask[None], y0, x0, height, width)
        if self.training:
            if np.random.random() < 0.5:
                channels, target = channels[..., ::-1], target[..., ::-1]
                if mask is not None:
                    mask = mask[..., ::-1]
            if np.random.random() < 0.5:
                channels, target = channels[..., ::-1, :], target[..., ::-1, :]
                if mask is not None:
                    mask = mask[..., ::-1, :]
        if mask is not None:
            return (torch.from_numpy(channels.copy()), torch.from_numpy(target.copy()),
                    torch.from_numpy(mask.copy()))
        return torch.from_numpy(channels.copy()), torch.from_numpy(target.copy())
