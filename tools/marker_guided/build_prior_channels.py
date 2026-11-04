"""Build the fixed E5-U input channels and nnU-Net dataset."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .validate_dataset import DatasetValidationError, validate_prediction_package


@dataclass(frozen=True)
class PriorConfig:
    sigma_0: float = 2.0
    k_sigma: float = 1.0
    w_0: float = 3.0
    k_w: float = 1.0
    endpoint_taper_px: float = 4.0
    d_min: float = 1.0
    version: str = "marker_prior_v1"

    def validate(self) -> None:
        if not self.version:
            raise ValueError("prior config version must be non-empty")
        values = (self.sigma_0, self.k_sigma, self.w_0, self.k_w, self.endpoint_taper_px, self.d_min)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in values):
            raise ValueError("prior parameters must be finite numeric values")
        if self.sigma_0 <= 0 or self.w_0 <= 0 or self.endpoint_taper_px <= 0 or self.d_min <= 0:
            raise ValueError("sigma_0, w_0, endpoint_taper_px, and d_min must be positive")
        if self.k_sigma < 0 or self.k_w < 0:
            raise ValueError("k_sigma and k_w must be non-negative")


def build_marker_channels(
    view_record: dict[str, Any], shape: tuple[int, int], prior_config: PriorConfig | dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point, finite-segment band, and E5-U validity maps in [0, 1]."""
    cfg = prior_config if isinstance(prior_config, PriorConfig) else PriorConfig(**(prior_config or {}))
    cfg.validate()
    if len(shape) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in shape):
        raise ValueError("shape must be positive 2D dimensions")
    h, w = map(int, shape)
    zeros = np.zeros((h, w), dtype=np.float32)
    if view_record.get("pair_status") == "invalid":
        return zeros.copy(), zeros.copy(), zeros.copy()
    if view_record.get("pair_status") not in ("reliable", "uncertain"):
        raise ValueError("pair_status must be reliable, uncertain, or invalid")
    points = np.asarray(view_record.get("marker_xy"), dtype=np.float64)
    radii = np.asarray(view_record.get("error_radius_px"), dtype=np.float64)
    if points.shape != (2, 2) or radii.shape != (2,) or not np.isfinite(points).all() or not np.isfinite(radii).all():
        raise ValueError("valid pair requires two finite points and two finite error radii")
    if np.any(radii < 0) or np.any(points[:, 0] < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= h):
        raise ValueError("coordinates must be in image and error radii non-negative")
    yy, xx = np.mgrid[:h, :w]
    point = zeros.copy()
    for (x, y), radius in zip(points, radii):
        sigma = float(np.sqrt(cfg.sigma_0**2 + (cfg.k_sigma * radius) ** 2))
        point = np.maximum(point, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))
    p1, p2 = points
    delta = p2 - p1
    length = float(np.linalg.norm(delta))
    if length < cfg.d_min:
        band = zeros.copy()
    else:
        t = ((xx - p1[0]) * delta[0] + (yy - p1[1]) * delta[1]) / (length**2)
        t_segment = np.clip(t, 0, 1)
        projection = np.stack((p1[0] + t_segment * delta[0], p1[1] + t_segment * delta[1]), axis=-1)
        distance = np.sqrt((xx - projection[..., 0]) ** 2 + (yy - projection[..., 1]) ** 2)
        overshoot = length * np.maximum(0, np.maximum(-t, t - 1))
        taper = np.exp(-(overshoot**2) / (2 * cfg.endpoint_taper_px**2))
        width = cfg.w_0 + cfg.k_w * float(np.max(radii))
        band = (np.exp(-(distance**2) / (2 * width**2)) * taper).astype(np.float32)
    validity = np.ones((h, w), dtype=np.float32)
    return point.astype(np.float32), band.astype(np.float32), validity


def build_e5u_dataset(
    manifest: str | Path | dict[str, Any],
    package: str | Path | dict[str, Any],
    phase2_fold_id: str,
    output_dir: str | Path,
    prior_config: PriorConfig | dict[str, Any] | None = None,
    dataset_id: int = 501,
    dataset_name: str | None = None,
) -> Path:
    """Write four-channel E5-U TIFF data, group-wise splits, and audit metadata."""
    cfg = prior_config if isinstance(prior_config, PriorConfig) else PriorConfig(**(prior_config or {}))
    cfg.validate()
    validated = validate_prediction_package(manifest, package, phase2_fold_id)
    manifest_obj, package_obj, joined = validated["manifest"], validated["package"], validated["joined"]
    manifest_root = validated["manifest_root"]
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to mix E5-U export with existing files in {out}")
    train_images, train_labels, test_images = out / "imagesTr", out / "labelsTr", out / "imagesTs"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    test_images.mkdir(parents=True, exist_ok=True)
    mapping: list[dict[str, Any]] = []
    group_sample_ids: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    for sample in manifest_obj["samples"]:
        key = (sample["case_id"], sample["view_id"])
        case, view = joined[key]
        nn_id, split = sample["nnunet_case_id"], sample["split"]
        image = tifffile.imread(Path(sample["projection_path"]) if Path(sample["projection_path"]).is_absolute() else manifest_root / sample["projection_path"])
        point, band, validity = build_marker_channels(view, image.shape, cfg)
        destination = test_images if split == "test" else train_images
        channels = (np.asarray(image), point, band, validity)
        for index, channel in enumerate(channels):
            tifffile.imwrite(destination / f"{nn_id}_{index:04d}.tif", channel)
        if split != "test":
            mask_path = Path(sample["mask_path"])
            if not mask_path.is_absolute():
                mask_path = manifest_root / mask_path
            tifffile.imwrite(train_labels / f"{nn_id}.tif", tifffile.imread(mask_path))
        group_sample_ids[split].append(nn_id)
        mapping.append({**{k: sample[k] for k in ("case_id", "view_id", "nnunet_case_id", "group_id", "split", "geometry_id")},
                        **{k: sample[k] for k in ("angle_deg", "geometry_manifest_key") if k in sample},
                        "prediction_source": case["prediction_source"], "model_id": case["model_id"],
                        "prior_config_version": cfg.version})
    ds_name = dataset_name or f"Dataset{dataset_id:03d}_MarkerGuided_{phase2_fold_id}"
    dataset_json = {
        "channel_names": {"0": "zscore", "1": "nonorm", "2": "nonorm", "3": "nonorm"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(group_sample_ids["train"]) + len(group_sample_ids["val"]),
        "file_ending": ".tif",
        "name": ds_name,
        "description": "E5-U projection, point, band, validity channels",
        "marker_guided": {"experiment": "E5-U", "phase2_fold_id": phase2_fold_id,
                           "channel_order": ["projection", "point", "band", "validity"],
                           "prior_config": asdict(cfg)},
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    split_entry = [{"train": group_sample_ids["train"], "val": group_sample_ids["val"]}]
    (out / "splits_final.json").write_text(json.dumps(split_entry, indent=2), encoding="utf-8")
    (out / "case_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (out / "source_registry.json").write_text(json.dumps(package_obj["source_registry"], indent=2), encoding="utf-8")
    return out
