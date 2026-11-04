import json
from pathlib import Path
import numpy as np
import tifffile
from .schema import Case, View


def _circular_span(angles):
    values = np.sort(np.mod(np.asarray(angles, dtype=float), 360.0))
    gaps = np.diff(np.r_[values, values[0] + 360.0])
    return float(360.0 - np.max(gaps))


def load_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("manifest must contain a cases list")
    cases = []
    ids = set()
    for record in records:
        if not isinstance(record, dict) or not record.get("case_id") or not record.get("group_id"):
            raise ValueError("case_id and group_id are required")
        if record["case_id"] in ids:
            raise ValueError("duplicate case_id")
        ids.add(record["case_id"])
        views = record.get("views", [])
        if len(views) != 3:
            raise ValueError("each case must contain exactly three views")
        view_ids = [v.get("view_id") for v in views]
        if len(set(view_ids)) != 3:
            raise ValueError("view_id values must be distinct")
        angles = [v.get("angle_deg") for v in views]
        if not np.isfinite(angles).all() or _circular_span(angles) > 60.0 + 1e-6:
            raise ValueError("view angles must fit one circular 60 degree window")
        parsed = []
        image_shape = None
        for v in views:
            projection = (path.parent / v["projection_path"]).resolve()
            if not projection.is_file():
                raise ValueError(f"image projection does not exist: {projection}")
            image = tifffile.imread(projection)
            if image.ndim != 2 or not np.isfinite(image).all():
                raise ValueError("image must be a finite 2D projection")
            if image_shape is None:
                image_shape = image.shape
            elif image.shape != image_shape:
                raise ValueError("image dimensions must match across views")
            xy = np.asarray(v.get("marker_xy"), dtype=float)
            if xy.shape != (2, 2) or not np.isfinite(xy).all():
                raise ValueError("marker coordinates must be finite 2x2")
            h, w = image.shape
            if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] >= h):
                raise ValueError("marker coordinates out of bounds")
            P = np.asarray(v.get("P"), dtype=float)
            if P.shape != (3, 4) or not np.isfinite(P).all() or np.linalg.matrix_rank(P) < 3:
                raise ValueError("P must be finite full-rank 3x4")
            mask = None
            mask_path = v.get("mask_path")
            if mask_path:
                mpath = (path.parent / mask_path).resolve()
                if not mpath.is_file():
                    raise ValueError(f"mask image does not exist: {mpath}")
                mask = tifffile.imread(mpath)
                if mask.shape != image.shape:
                    raise ValueError("mask dimensions must match image")
            parsed.append(View(int(v["view_id"]), float(v["angle_deg"]), str(projection), image, xy, P, mask, mask_path))
        cases.append(Case(str(record["case_id"]), str(record["group_id"]), parsed, record.get("difficulty", "normal"), record.get("metadata", {})))
    return cases


def validate_group_split(cases, fold_spec):
    splits = {name: set(fold_spec.get(name, [])) for name in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if splits[left] & splits[right]:
            raise ValueError(f"group split overlap between {left} and {right}")
    known = {c.group_id for c in cases}
    assigned = set.union(*splits.values()) if splits else set()
    if known - assigned:
        raise ValueError(f"groups missing from split: {sorted(known-assigned)}")
    if assigned - known:
        raise ValueError(f"split references unknown groups: {sorted(assigned-known)}")
    return True
