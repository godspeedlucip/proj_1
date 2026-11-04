"""Validation for the Phase 1 to Phase 2 handoff package.

Dataset manifest format::

    {"phase2_fold_id": "fold_0", "splits": {"train": ["g1"], ...},
     "samples": [{"case_id", "view_id", "nnunet_case_id", "group_id",
                  "split", "projection_path", "mask_path", "geometry_id"}]}

Prediction package format::

    {"phase2_fold_id": "fold_0", "source_registry": [{"model_id",
      "fit_group_ids", "selection_group_ids"}], "cases": [<v1 case record>]}

Paths may be relative to their respective JSON files/directories. The package
may instead be a directory containing ``package.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


class DatasetValidationError(ValueError):
    """Raised when the handoff package is inconsistent or unsafe to consume."""


def _load_json(value: str | Path | dict[str, Any], name: str) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, dict):
        return value, None
    path = Path(value)
    if path.is_dir():
        path = path / ("manifest.json" if name == "manifest" else "package.json")
    try:
        return json.loads(path.read_text(encoding="utf-8")), path.parent
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"cannot read {name} JSON at {path}: {exc}") from exc


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DatasetValidationError(f"{label} must be a finite number")
    return float(value)


def _resolved(base: Path | None, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() or base is None else base / p


def validate_prediction_package(
    manifest: str | Path | dict[str, Any],
    package: str | Path | dict[str, Any],
    phase2_fold_id: str,
) -> dict[str, Any]:
    """Validate exact (case_id, view_id) joins, geometry, splits, and sources.

    Returns normalized manifest/package objects plus the joined sample records.
    """
    manifest_obj, manifest_root = _load_json(manifest, "manifest")
    package_obj, package_root = _load_json(package, "package")
    for obj, name in ((manifest_obj, "manifest"), (package_obj, "package")):
        if obj.get("phase2_fold_id") != phase2_fold_id:
            raise DatasetValidationError(f"{name} phase2_fold_id does not match {phase2_fold_id!r}")

    splits = manifest_obj.get("splits")
    if not isinstance(splits, dict) or not {"train", "val", "test"}.issubset(splits):
        raise DatasetValidationError("manifest.splits must define train, val, and test group lists")
    group_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        groups = splits[split]
        if not isinstance(groups, list) or any(not isinstance(g, str) or not g for g in groups):
            raise DatasetValidationError(f"manifest.splits.{split} must be a list of group IDs")
        for group in groups:
            if group in group_to_split:
                raise DatasetValidationError(f"group {group!r} appears in multiple splits")
            group_to_split[group] = split

    samples = manifest_obj.get("samples")
    cases = package_obj.get("cases")
    if not isinstance(samples, list) or not samples or not isinstance(cases, list):
        raise DatasetValidationError("manifest.samples and package.cases must be lists (samples non-empty)")
    sample_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    nnunet_ids: set[str] = set()
    group_views: dict[str, dict[str, set[int]]] = {}
    for sample in samples:
        try:
            key = (sample["case_id"], sample["view_id"])
            case_id, view_id = key
            group_id, split, nn_id = sample["group_id"], sample["split"], sample["nnunet_case_id"]
            projection, mask = sample["projection_path"], sample.get("mask_path")
            geometry_id = sample["geometry_id"]
        except (KeyError, TypeError) as exc:
            raise DatasetValidationError(f"manifest sample missing required field: {exc}") from exc
        if not isinstance(case_id, str) or not case_id or isinstance(view_id, bool) or not isinstance(view_id, int):
            raise DatasetValidationError("sample case_id/view_id invalid")
        if key in sample_by_key:
            raise DatasetValidationError(f"duplicate sample view {key}")
        if not all(isinstance(v, str) and v for v in (group_id, split, nn_id, projection, geometry_id)):
            raise DatasetValidationError(f"sample {key} has empty/invalid identifiers or path")
        if split not in ("train", "val", "test") or group_to_split.get(group_id) != split:
            raise DatasetValidationError(f"sample {key} group/split mismatch")
        if split != "test" and (not isinstance(mask, str) or not mask):
            raise DatasetValidationError(f"labeled {split} sample {key} requires mask_path")
        if not (isinstance(sample.get("geometry_manifest_key"), str) and sample["geometry_manifest_key"]
                or "angle_deg" in sample):
            raise DatasetValidationError(f"sample {key} requires angle_deg or geometry_manifest_key")
        if "angle_deg" in sample:
            _finite_number(sample["angle_deg"], f"angle_deg for {key}")
        if nn_id in nnunet_ids:
            raise DatasetValidationError(f"duplicate nnunet_case_id {nn_id!r}")
        nnunet_ids.add(nn_id)
        sample_by_key[key] = sample
        group_views.setdefault(case_id, {}).setdefault(group_id, set()).add(view_id)
        image_path = _resolved(manifest_root, projection)
        if not image_path.is_file():
            raise DatasetValidationError(f"projection does not exist for {key}: {image_path}")
        try:
            image = tifffile.imread(image_path)
        except Exception as exc:
            raise DatasetValidationError(f"cannot read projection for {key}: {exc}") from exc
        if image.ndim != 2 or not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise DatasetValidationError(f"projection for {key} must be a finite 2D numeric image")
        if mask is not None:
            mask_path = _resolved(manifest_root, mask)
            if not mask_path.is_file():
                raise DatasetValidationError(f"mask does not exist for {key}: {mask_path}")
            label = tifffile.imread(mask_path)
            if label.shape != image.shape:
                raise DatasetValidationError(f"projection/mask shape mismatch for {key}")
            if not np.isfinite(label).all() or not np.isin(label, (0, 1)).all():
                raise DatasetValidationError(f"mask for {key} must contain only binary labels 0 and 1")
    for case_id, group_map in group_views.items():
        if len(group_map) != 1:
            raise DatasetValidationError(f"case {case_id!r} maps to multiple groups")
        views = next(iter(group_map.values()))
        if len(views) != 3:
            raise DatasetValidationError(f"case {case_id!r} must have exactly three distinct views")

    records: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, dict) or case.get("schema_version") != "phase1_marker_case_v1":
            raise DatasetValidationError("each package case must use phase1_marker_case_v1")
        case_id = case.get("case_id")
        if not all(isinstance(case.get(k), str) and case[k] for k in ("case_id", "group_id", "model_id", "geometry_id")):
            raise DatasetValidationError("case metadata requires case_id, group_id, model_id, geometry_id")
        if case.get("calibration_id") is not None and (not isinstance(case["calibration_id"], str) or not case["calibration_id"]):
            raise DatasetValidationError(f"case {case_id!r} has invalid calibration_id")
        if case.get("prediction_source") not in ("group_oof", "held_out"):
            raise DatasetValidationError(f"case {case_id!r} has unsupported prediction_source")
        if case["prediction_source"] == "group_oof" and not case.get("fold_id"):
            raise DatasetValidationError(f"group_oof case {case_id!r} requires fold_id")
        views = case.get("views")
        if not isinstance(views, list) or len(views) != 3:
            raise DatasetValidationError(f"case {case_id!r} must contain exactly three views")
        seen: set[int] = set()
        for view in views:
            if not isinstance(view, dict) or isinstance(view.get("view_id"), bool) or not isinstance(view.get("view_id"), int):
                raise DatasetValidationError(f"case {case_id!r} has invalid view_id")
            view_id = view["view_id"]
            key = (case_id, view_id)
            if view_id in seen or key in records:
                raise DatasetValidationError(f"duplicate prediction view {key}")
            seen.add(view_id)
            sample = sample_by_key.get(key)
            if sample is None:
                raise DatasetValidationError(f"prediction has no exact manifest match for {key}")
            if sample["group_id"] != case["group_id"] or sample["geometry_id"] != case["geometry_id"]:
                raise DatasetValidationError(f"group/geometry mismatch for {key}")
            image_path = _resolved(manifest_root, sample["projection_path"])
            image = tifffile.imread(image_path)
            size = view.get("image_size")
            if size != [int(image.shape[0]), int(image.shape[1])]:
                raise DatasetValidationError(f"image_size mismatch for {key}")
            status = view.get("pair_status")
            if status not in ("reliable", "uncertain", "invalid"):
                raise DatasetValidationError(f"invalid pair_status for {key}")
            if "q_pair" not in view:
                raise DatasetValidationError(f"view {key} must explicitly include q_pair (null is allowed for E5-U)")
            if status == "invalid":
                if view.get("marker_xy") is not None or view.get("error_radius_px") is not None:
                    raise DatasetValidationError(f"invalid view {key} must have null coordinates and error radii")
                if view.get("q_pair") != 0:
                    raise DatasetValidationError(f"invalid view {key} must have q_pair=0")
            else:
                points, radii = view.get("marker_xy"), view.get("error_radius_px")
                if not isinstance(points, list) or len(points) != 2 or not isinstance(radii, list) or len(radii) != 2:
                    raise DatasetValidationError(f"valid view {key} requires two points and error radii")
                h, w = image.shape
                for point in points:
                    if not isinstance(point, list) or len(point) != 2:
                        raise DatasetValidationError(f"invalid point for {key}")
                    x, y = (_finite_number(v, f"coordinate for {key}") for v in point)
                    if not (0 <= x < w and 0 <= y < h):
                        raise DatasetValidationError(f"coordinate outside image for {key}")
                for radius in radii:
                    if _finite_number(radius, f"error radius for {key}") < 0:
                        raise DatasetValidationError(f"negative error radius for {key}")
                q = view.get("q_pair")
                if q is not None:
                    if not 0 <= _finite_number(q, f"q_pair for {key}") <= 1:
                        raise DatasetValidationError(f"q_pair outside [0,1] for {key}")
                    tolerance = view.get("q_tolerance_px", case.get("q_tolerance_px"))
                    if case.get("calibration_id") is None or _finite_number(tolerance, f"q_tolerance_px for {key}") <= 0:
                        raise DatasetValidationError(f"numeric q_pair for {key} requires calibration_id and positive tolerance")
            records[key] = (case, view)
        valid_q_values = [view.get("q_pair") for view in views if view.get("pair_status") != "invalid"]
        if case.get("calibration_id") is None and any(value is not None for value in valid_q_values):
            raise DatasetValidationError(f"case {case_id!r} has q_pair values without calibration_id")
        if case.get("calibration_id") is not None and any(value is None for value in valid_q_values):
            raise DatasetValidationError(f"case {case_id!r} mixes calibrated and uncalibrated valid q_pair values")
    if set(records) != set(sample_by_key):
        missing = sorted(set(sample_by_key) - set(records))
        raise DatasetValidationError(f"prediction package missing manifest views: {missing}")

    registry = package_obj.get("source_registry")
    if not isinstance(registry, list):
        raise DatasetValidationError("package.source_registry must be a list")
    sources = {}
    for entry in registry:
        if not isinstance(entry, dict) or not isinstance(entry.get("model_id"), str) or not entry["model_id"]:
            raise DatasetValidationError("source_registry entries require model_id")
        if entry["model_id"] in sources:
            raise DatasetValidationError(f"duplicate source registry model_id {entry['model_id']!r}")
        if not all(isinstance(entry.get(k), str) and entry[k] for k in ("model_artifact_id", "pipeline_config_id")):
            raise DatasetValidationError(f"model {entry['model_id']!r} requires artifact and pipeline config IDs")
        if not isinstance(entry.get("fit_group_ids"), list) or not isinstance(entry.get("selection_group_ids"), list):
            raise DatasetValidationError(f"model {entry['model_id']!r} requires fit and selection group lists")
        if any(not isinstance(group, str) or not group for group in entry["fit_group_ids"] + entry["selection_group_ids"]):
            raise DatasetValidationError(f"model {entry['model_id']!r} has invalid group IDs")
        sources[entry["model_id"]] = entry
    calibration_registry = package_obj.get("calibration_registry", [])
    if not isinstance(calibration_registry, list):
        raise DatasetValidationError("package.calibration_registry must be a list")
    calibrations = {}
    for entry in calibration_registry:
        if not isinstance(entry, dict) or not isinstance(entry.get("calibration_id"), str) or not entry["calibration_id"]:
            raise DatasetValidationError("calibration_registry entries require calibration_id")
        if entry["calibration_id"] in calibrations:
            raise DatasetValidationError(f"duplicate calibration_id {entry['calibration_id']!r}")
        if not isinstance(entry.get("fit_group_ids"), list):
            raise DatasetValidationError(f"calibration {entry['calibration_id']!r} requires fit_group_ids")
        if any(not isinstance(group, str) or not group for group in entry["fit_group_ids"]):
            raise DatasetValidationError(f"calibration {entry['calibration_id']!r} has invalid group IDs")
        calibrations[entry["calibration_id"]] = entry
    for key, (case, _) in records.items():
        source = sources.get(case["model_id"])
        if source is None:
            raise DatasetValidationError(f"missing source registry entry for model {case['model_id']!r}")
        fit_groups = set(source["fit_group_ids"])
        selection_groups = set(source["selection_group_ids"])
        train_groups = set(splits["train"])
        if not (fit_groups | selection_groups) <= train_groups:
            raise DatasetValidationError(f"model {case['model_id']!r} used groups outside this fold's train split")
        sample = sample_by_key[key]
        if sample["split"] == "train":
            if case["prediction_source"] != "group_oof":
                raise DatasetValidationError(f"training sample {key} must use group_oof prediction")
            if sample["group_id"] in fit_groups | selection_groups:
                raise DatasetValidationError(f"training group leaked into its prediction model for {key}")
        elif case["prediction_source"] != "held_out":
            raise DatasetValidationError(f"held-out sample {key} must use held_out prediction")
        view = records[key][1]
        if view.get("pair_status") != "invalid" and view.get("q_pair") is not None:
            calibration = calibrations.get(case["calibration_id"])
            if calibration is None:
                raise DatasetValidationError(f"missing calibration registry entry for {case['calibration_id']!r}")
            calibration_groups = set(calibration["fit_group_ids"])
            if not calibration_groups <= train_groups or sample["group_id"] in calibration_groups:
                raise DatasetValidationError(f"calibration leakage or out-of-fold group for {key}")
    return {"manifest": manifest_obj, "package": package_obj, "joined": records,
            "manifest_root": manifest_root, "package_root": package_root}
