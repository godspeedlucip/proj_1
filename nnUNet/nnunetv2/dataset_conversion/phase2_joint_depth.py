"""Prepare raw, matched mask and line-integral targets from the Phase 1 physical cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
from pipelines.medical_image import MedicalImage
from pipelines.project import _new_projection_setup, read_marker_mask, to_tigre_volume
from pipelines.stent_fov import apply_saved_stent_geometry, validate_stent_projection

DEFAULT_MANIFEST = PROJECT / "training_data/training/generated/phase1_200/phase1/physical_manifest.json"
DEFAULT_OUTPUT = PROJECT / "training_data/training/nnunet/phase2/Dataset504_MarkerDepth"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_length_units(source, series, variant):
    """Check central detector rays through an analytically specified solid at two angles."""
    import tigre

    path = source / "stent_projections" / series / variant / "projection.json"
    saved = json.loads(path.read_text())["geometry"]
    shape = np.asarray(saved["nVoxel"])
    spacing = np.asarray(saved["dVoxel_mm"])
    geo, _, _, _ = _new_projection_setup(shape[::-1], spacing[::-1], n_projections=2, angle_span_deg=90)
    apply_saved_stent_geometry(geo, saved)
    if np.any(geo.offOrigin) or np.any(geo.offDetector):
        raise ValueError("length calibration requires a centered reference geometry")
    # Center a 40(Z) x 32(Y) x 16(X) occupied voxel solid in the exact source geometry.
    lengths = np.array([40, 32, 16])
    low = (shape - lengths) // 2
    solid = np.zeros(tuple(shape), dtype=np.float32)
    solid[tuple(slice(int(i), int(i + n)) for i, n in zip(low, lengths))] = 1.
    radians = np.deg2rad(np.array([0, 90], dtype=np.float32))
    projected = tigre.Ax(solid, geo, radians, projection_type="Siddon")
    pixel_rc = np.asarray(geo.nDetector) // 2
    uv_mm = (pixel_rc - (np.asarray(geo.nDetector) - 1) / 2) * geo.dDetector
    obliquity = np.sqrt(1 + np.dot(uv_mm, uv_mm) / geo.DSD**2)
    expected = (lengths * spacing)[[2, 1]] * obliquity
    measured = projected[:, pixel_rc[0], pixel_rc[1]].astype(float)
    error = np.abs(measured - expected)
    if not np.isfinite(error).all() or np.max(error) > 0.001:
        raise ValueError(f"known-thickness TIGRE calibration failed: {measured} mm vs {expected} mm")
    return {"reference_geometry": str(path), "occupied_voxels_zyx": lengths.tolist(),
            "voxel_spacing_zyx_mm": spacing.tolist(), "angles_deg": [0, 90],
            "detector_pixel_rc": pixel_rc.tolist(), "analytical_ray_length_mm": expected.tolist(),
            "measured_ray_length_mm": measured.tolist(), "max_absolute_error_mm": float(error.max())}


def project_model(source, output, series, variant, views):
    import tigre

    model_dir = source / "stent_models" / series / variant
    volume_path = model_dir / "stent_real_grid.nii.gz"
    geometry_path = source / "stent_projections" / series / variant / "projection.json"
    model_path, marker_path = model_dir / "stent.json", model_dir / "marker.json"
    info = json.loads(model_path.read_text())
    report = json.loads(geometry_path.read_text())
    angles = report["angles_deg"]
    if (report["stent_series"] != series or report["stent_variant"] != variant
            or report["output_mode"] != "2D" or report["n_projections"] != len(angles)
            or list(zip(report["view_indices"], angles)) != views):
        raise ValueError(f"model and physical views disagree for {series}/{variant}")
    model = MedicalImage.read(volume_path)
    if (tuple(info["volume_shape"]) != model.data.shape
            or not np.allclose(info["voxel_spacing_mm"], model.spacing)
            or not np.all((model.data == 0) | (model.data == 1))):
        raise ValueError(f"invalid occupied-voxel model {volume_path}")
    marker = read_marker_mask(marker_path, model.data)
    if np.count_nonzero(marker) != info["marker_voxels"] or not marker.any():
        raise ValueError(f"marker occupancy mismatch in {model_dir}")
    geo, _, _, _ = _new_projection_setup(model.data.shape, info["voxel_spacing_mm"],
                                          n_projections=len(angles), angle_span_deg=report["angle_span_deg"])
    apply_saved_stent_geometry(geo, report["geometry"])
    volume = to_tigre_volume(model.data)
    del marker, model
    depth_dir = output / "depth" / series / variant
    mask_dir = output / "mask" / series / variant
    depth_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    masks = {}
    stats = {}
    for start in range(0, len(views), 12):
        batch = views[start:start + 12]
        radians = np.deg2rad(np.asarray([angle for _, angle in batch], dtype=np.float32))
        depth = tigre.Ax(volume, geo, radians, projection_type="Siddon")
        if (depth.dtype != np.float32 or depth.shape != (len(batch), *geo.nDetector)
                or not np.isfinite(depth).all() or depth.min() < 0):
            raise ValueError(f"invalid TIGRE output for {series}/{variant}")
        validate_stent_projection(depth, margin_pixels=report["geometry"]["margin_pixels"])
        for (view_id, angle), frame in zip(batch, depth):
            name = f"{view_id:03d}.tif"
            mask = (frame > 0).astype(np.uint8)
            for directory, data in ((depth_dir, frame), (mask_dir, mask)):
                tifffile.imwrite(directory / name, data, photometric="minisblack",
                                 compression="zlib", compressionargs={"level": 1}, metadata=None)
                if not np.array_equal(tifffile.imread(directory / name), data):
                    raise ValueError(f"TIFF round-trip failed for {directory / name}")
            masks[view_id] = mask
            stats[view_id] = {"angle_deg": angle, "foreground_pixels": int(mask.sum()),
                              "raw_max_mm": float(frame.max())}
    provenance = {"occupancy_volume": str(volume_path), "projection_report": str(geometry_path),
                  "occupancy_sha256": digest(volume_path), "marker_sha256": digest(marker_path),
                  "model_metadata_sha256": digest(model_path), "geometry_sha256": digest(geometry_path),
                  "marker_voxels": info["marker_voxels"], "geometry": report["geometry"], "views": stats}
    return masks, provenance


def get_records(manifest_path, physical):
    if physical["schema_version"] != "phase1_physical_v1":
        raise ValueError("expected phase1_physical_v1")
    folds = physical["folds"]["fold_0"]
    group_split = {}
    for split in ("train", "val", "test"):
        for group in folds[split]:
            if group in group_split:
                raise ValueError(f"duplicate group {group} in fold_0")
            group_split[group] = split
    cases = set()
    records = []
    model_views = {}
    source_labels = {}
    for case in physical["cases"]:
        case_id, series = case["case_id"], case["group_id"]
        variant = case["metadata"]["stent_variant"]
        split = group_split[series]
        views = sorted((v["view_id"], v["angle_deg"]) for v in case["views"])
        if (not views or len({v for v, _ in views}) != len(views)
                or len({a for _, a in views}) != len(views)
                or model_views.setdefault((series, variant), views) != views
                or case_id in cases):
            raise ValueError(f"duplicate or inconsistent case/views in {case_id}")
        cases.add(case_id)
        for v in case["views"]:
            view_id = v["view_id"]
            model_view = Path(series) / variant / f"{view_id:03d}.tif"
            image = (manifest_path.parent / v["projection_path"]).resolve()
            label = (manifest_path.parent / v["mask_path"]).resolve()
            if image.stem != label.stem or float(image.stem) != v["angle_deg"]:
                raise ValueError(f"image/label angle mismatch in {case_id} view {view_id}")
            sample_id = f"{case_id}_v{view_id:03d}"
            source_labels[sample_id] = label
            records.append({"sample_id": sample_id, "split": split, "case_id": case_id,
                            "group_id": series, "stent_variant": variant, "view_id": view_id,
                            "angle_deg": v["angle_deg"], "image": str(image),
                            "depth": str(Path("depth") / model_view), "mask": str(Path("mask") / model_view),
                            "marker_xy": v["marker_xy"]})
    if len(source_labels) != len(records) or len({r["image"] for r in records}) != len(records):
        raise ValueError("duplicate sample ID or image path")
    return records, model_views, source_labels


def validate_record(record, masks, source_labels):
    image = tifffile.imread(record["image"], maxworkers=1)
    label = tifffile.imread(source_labels[record["sample_id"]], maxworkers=1)
    mask = masks[(record["group_id"], record["stent_variant"])][record["view_id"]]
    if (image.shape != mask.shape or image.ndim != 2 or not np.isfinite(image).all()
            or label.shape != mask.shape or not np.all((label == 0) | (label == 1))):
        raise ValueError(f"invalid stacked image or physical label for {record['sample_id']}")
    physical_fg = label > 0
    iou = float(np.count_nonzero(physical_fg & mask) / np.count_nonzero(physical_fg | mask))
    if iou < 0.99:
        raise ValueError(f"projection support disagrees with physical label for {record['sample_id']}: {iou}")
    xy = np.asarray(record["marker_xy"])
    if xy.shape != (2, 2) or not np.isfinite(xy).all():
        raise ValueError(f"invalid physical marker coordinates for {record['sample_id']}")
    points = np.rint(xy).astype(int)
    if (np.any(points < 0) or np.any(points >= mask.shape[::-1])
            or not np.all(mask[points[:, 1], points[:, 0]])):
        raise ValueError(f"marker off ray-integral support in {record['sample_id']}")
    return iou


def prepare(manifest_path, destination, gpu, workers):
    manifest_path, destination = manifest_path.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if workers < 1:
        raise ValueError("workers must be positive")
    # Select by UUID, which is stable across CUDA and nvidia-smi device orderings.
    gpu_uuid = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
        raise ValueError(f"expected one physical GPU for index {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    from tigre.utilities.gpu import GpuIds
    if list(GpuIds().devices) != [0]:
        raise RuntimeError("TIGRE did not resolve the selected physical GPU as its sole device")
    physical = json.loads(manifest_path.read_text())
    records, model_views, source_labels = get_records(manifest_path, physical)
    counts = Counter(r["split"] for r in records)
    if counts != {"train": 5040, "val": 720, "test": 60}:
        raise ValueError(f"current physical cohort counts differ from Dataset504: {counts}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        output = Path(temporary)
        masks, provenance = {}, {}
        for (series, variant), views in sorted(model_views.items()):
            masks[series, variant], provenance[f"{series}/{variant}"] = project_model(
                manifest_path.parent.parent, output, series, variant, views)
            print(f"Projected {series}/{variant} ({len(views)} raw Siddon depth views)", flush=True)
        calibration = verify_length_units(manifest_path.parent.parent, *sorted(model_views)[0])
        train_keys = {(r["group_id"], r["stent_variant"], r["view_id"]) for r in records
                      if r["split"] == "train"}
        scale = max(provenance[f"{s}/{v}"]["views"][i]["raw_max_mm"] for s, v, i in train_keys)
        print(f"Validating {len(records)} full projected images against physical labels", flush=True)
        min_iou = 1.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, iou in enumerate(pool.map(lambda r: validate_record(r, masks, source_labels), records), 1):
                min_iou = min(min_iou, iou)
                if i % 500 == 0:
                    print(f"Validated {i}/{len(records)}", flush=True)
        payload = {"schema_version": "phase2_joint_marker_depth_v1", "counts": dict(counts),
                   "target_scale": scale, "target_scale_definition": "max raw depth in training views (mm)",
                   "depth_definition": "Siddon line integral of 0/1 stent occupancy including both markers (mm)",
                   "mask_definition": "uint8(raw_depth_mm > 0)", "marker_source": "physical geometry truth, x/y",
                   "source_manifest": str(manifest_path), "source_manifest_sha256": digest(manifest_path),
                   "tigre_version": version("pytigre"), "gpu_uuid": gpu_uuid,
                   "fold": physical["folds"]["fold_0"], "projection_provenance": provenance,
                   "validation": {"full_images_checked": len(records),
                                  "minimum_physical_label_iou": min_iou,
                                  "known_thickness_calibration": calibration,
                                  "unique_training_depths_for_scale": len(train_keys)},
                   "samples": records}
        (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        output.rename(destination)
    print(f"Prepared {destination}: {dict(counts)}, raw depth scale {scale}, minimum label IoU {min_iou}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi index for TIGRE")
    parser.add_argument("--workers", type=int, default=8, help="parallel full-image validation reads")
    a = parser.parse_args()
    prepare(a.manifest, a.output, a.gpu, a.workers)
