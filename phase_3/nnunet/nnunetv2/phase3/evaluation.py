"""Original-grid case metrics and equal-source-group aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .inference import Prediction


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    group_id: str
    dice: float
    precision: float
    recall: float
    false_positive_mm3: float
    pred_components: int
    target_components: int
    skeleton_distance_mm: float | None
    empty_status: str
    iou: float
    mse: float
    mae: float


def compute_case_metrics(prediction: Prediction, target: np.ndarray, *, case_id: str,
                         group_id: str) -> CaseMetrics:
    predicted, truth = prediction.mask, target
    probability = np.asarray(prediction.probability)
    if (predicted.shape != truth.shape or not np.isin(predicted, (0, 1)).all()
            or not np.isin(truth, (0, 1)).all() or probability.shape != truth.shape
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("Metrics require aligned binary masks and finite foreground probabilities")
    squared_error = 0.0
    absolute_error = 0.0
    for start in range(0, truth.shape[0], 16):
        difference = probability[start:start + 16].astype(np.float32, copy=False) - truth[start:start + 16]
        squared_error += float(np.square(difference, dtype=np.float64).sum())
        absolute_error += float(np.abs(difference, dtype=np.float64).sum())
    mse = squared_error / truth.size
    mae = absolute_error / truth.size
    predicted, truth = predicted.astype(bool), truth.astype(bool)
    n_pred, n_target = int(predicted.sum()), int(truth.sum())
    overlap = int((predicted & truth).sum())
    union = n_pred + n_target - overlap
    iou = overlap / union if union else 1.0
    status, distance = "none", None
    if not n_pred and not n_target:
        dice = precision = recall = 1.
        distance, status = 0., "both_empty"
    elif not n_pred:
        dice = precision = recall = 0.
        status = "empty_prediction"
    elif not n_target:
        dice = precision = 0.
        recall, status = 1., "empty_target"
    else:
        dice = 2 * overlap / (n_pred + n_target)
        precision, recall = overlap / n_pred, overlap / n_target
        points = []
        for mask in (predicted, truth):
            ijk = np.argwhere(skeletonize(mask, method="lee"))[:, ::-1]
            points.append(ijk @ prediction.meta.affine_mm[:3, :3].T + prediction.meta.affine_mm[:3, 3])
        if not all(len(part) for part in points):
            status = "empty_skeleton"
        else:
            distance = float((cKDTree(points[0]).query(points[1])[0].mean()
                              + cKDTree(points[1]).query(points[0])[0].mean()) / 2)
    structure = np.ones((3, 3, 3), bool)
    return CaseMetrics(case_id, group_id, float(dice), float(precision), float(recall),
                       float((n_pred - overlap) * abs(np.linalg.det(prediction.meta.affine_mm[:3, :3]))),
                       int(label(predicted, structure)[1]), int(label(truth, structure)[1]),
                       distance, status, float(iou), float(mse), float(mae))


def aggregate_group_metrics(rows) -> dict:
    rows = list(rows)
    if not rows or len({row.case_id for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate case metrics")
    keys = ("dice", "iou", "mse", "mae", "precision", "recall",
            "false_positive_mm3", "pred_components", "target_components",
            "skeleton_distance_mm")
    groups = {}
    for group_id in sorted({row.group_id for row in rows}):
        members = [row for row in rows if row.group_id == group_id]
        values = {}
        for key in keys:
            valid = [getattr(row, key) for row in members if getattr(row, key) is not None]
            values[key] = float(np.mean(valid)) if valid else None
        values.update(case_count=len(members),
                      distance_valid_cases=sum(row.skeleton_distance_mm is not None for row in members),
                      distance_failures=sum(row.skeleton_distance_mm is None for row in members))
        groups[group_id] = values
    mean, std, denominator = {}, {}, {}
    for key in keys:
        valid = [group[key] for group in groups.values() if group[key] is not None]
        mean[key] = float(np.mean(valid)) if valid else None
        std[key] = float(np.std(valid, ddof=0)) if valid else None
        denominator[key] = len(valid)
    return {"groups": groups, "mean": mean, "std": std,
            "valid_group_denominators": denominator, "case_count": len(rows),
            "group_count": len(groups), "std_ddof": 0,
            "distance_valid_cases": sum(row.skeleton_distance_mm is not None for row in rows),
            "distance_failures": sum(row.skeleton_distance_mm is None for row in rows),
            "distance_failure_cases": [row.case_id for row in rows if row.skeleton_distance_mm is None]}
