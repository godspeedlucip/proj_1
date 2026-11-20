"""Export continuous 2D targets projected from the Phase 1 stent-model volumes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile

from pipelines.medical_image import MedicalImage
from pipelines.project import _new_projection_setup, project_volume, to_tigre_volume
from pipelines.stent_fov import configure_safe_stent_geometry

from .export_data import verify_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = PROJECT_ROOT / "training_data/training/generated/phase2_marker_fusion_regression_v1"
DEFAULT_DESTINATION = PROJECT_ROOT / "training_data/training/generated/phase2_stent_model_regression_v2"
DEFAULT_SOURCE = PROJECT_ROOT / "training_data/training/generated/phase1_200"


def normalize_stent_targets(train_paths: set[Path], all_paths: set[Path]) -> float:
    """Scale ray integrals by the train-only maximum, retaining their float values."""
    if not train_paths or not all_paths or not train_paths <= all_paths:
        raise ValueError("training targets must be a nonempty subset of all targets")
    scale = 0.0
    for path in sorted(train_paths):
        image = tifffile.imread(path)
        if image.ndim != 2 or not np.isfinite(image).all() or np.any(image < 0):
            raise ValueError(f"invalid stent-model projection: {path}")
        scale = max(scale, float(image.max()))
    if scale <= 0:
        raise ValueError("training stent-model projections are empty")
    for path in sorted(all_paths):
        image = tifffile.imread(path).astype(np.float32)
        if image.ndim != 2 or not np.isfinite(image).all() or np.any(image < 0):
            raise ValueError(f"invalid stent-model projection: {path}")
        tifffile.imwrite(path, np.clip(image / scale, 0, 1).astype(np.float32),
                         compression="zlib", compressionargs={"level": 1})
    return scale


def _link_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _project_model(model_dir: Path, report_path: Path, destination: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    info = json.loads((model_dir / "stent.json").read_text(encoding="utf-8"))
    image_path = model_dir / "stent_real_grid.nii.gz"
    stent = MedicalImage.read(image_path)
    if stent.data.shape != tuple(info["volume_shape"]):
        raise ValueError(f"model shape disagrees with metadata: {image_path}")
    if not np.isin(stent.data, (0, 1)).all():
        raise ValueError(f"model must be a binary occupancy grid: {image_path}")
    angles = np.asarray(report["angles_deg"], dtype=np.float32)
    if report["output_mode"] != "2D" or len(angles) != report["n_projections"]:
        raise ValueError(f"incompatible projection metadata: {report_path}")
    geometry, _, _, view_indices = _new_projection_setup(
        info["volume_shape"], info["voxel_spacing_mm"],
        n_projections=len(angles), angle_span_deg=float(report["angle_span_deg"]))
    if not np.allclose(np.linspace(0, report["angle_span_deg"], len(angles)), angles):
        raise ValueError(f"projection angles differ from setup: {report_path}")
    configure_safe_stent_geometry(
        geometry, np.transpose(stent.data > 0, (2, 1, 0)),
        f"{report['stent_series']}:{report['stent_variant']}", randomize=False)
    project_volume(to_tigre_volume(stent.data), geometry, np.deg2rad(angles),
                   view_indices, destination, "stent", check_margin=True,
                   output_mode="2D", angles_deg=angles.tolist())
    return {"model_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "projection_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}


def export_stent_model_dataset(base_dir: str | Path, source_root: str | Path,
                               destination: str | Path) -> Path:
    """Retain image/prior files and replace target with projected model occupancy."""
    base_dir, source_root, destination = map(
        lambda p: Path(p).resolve(), (base_dir, source_root, destination))
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    base = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    if (base.get("schema_version") != "phase2_projection_regression_v1"
            or base.get("target_kind") != "normalized_continuous_implant_including_markers"):
        raise ValueError("expected the original marker-fusion data package")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"unfinished export exists: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        records = []
        shared_files: set[Path] = set()
        train_targets: set[Path] = set()
        all_targets: set[Path] = set()
        models: set[tuple[str, str]] = set()
        for old in base["samples"]:
            old_target = Path(old["target"])
            if (len(old_target.parts) != 6 or old_target.parts[0] != "targets"
                    or old_target.parts[-2] != "implant"):
                raise ValueError(f"unexpected target path: {old_target}")
            series, variant = old_target.parts[1:3]
            models.add((series, variant))
            new_target = old_target.parent.parent / "stent" / old_target.name
            record = {**old, "target": new_target.as_posix()}
            records.append(record)
            all_targets.add(temporary / new_target)
            if old["split"] == "train":
                train_targets.add(temporary / new_target)
            for key in ("image", "point", "band", "validity"):
                relative = Path(old[key])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"nonlocal data path: {relative}")
                shared_files.add(relative)
        for index, relative in enumerate(sorted(shared_files), 1):
            _link_immutable(base_dir / relative, temporary / relative)
            if index % 500 == 0 or index == len(shared_files):
                print(f"Linked {index}/{len(shared_files)} image/prior files", flush=True)
        provenance = {}
        for index, (series, variant) in enumerate(sorted(models), 1):
            model_dir = source_root / "stent_models" / series / variant
            report_path = source_root / "stent_projections" / series / variant / "projection.json"
            provenance[f"{series}/{variant}"] = _project_model(
                model_dir, report_path, temporary / "targets" / series / variant)
            print(f"Projected {index}/{len(models)} stent models: {series}/{variant}", flush=True)
        generated = set(temporary.glob("targets/*/*/projections/stent/*.tif"))
        if generated != all_targets:
            raise ValueError(f"generated targets disagree with manifest: {len(generated)} vs {len(all_targets)}")
        scale = normalize_stent_targets(train_targets, all_targets)
        payload = {**base, "schema_version": "phase2_projection_regression_v2",
                   "target_kind": "normalized_continuous_stent_model_projection",
                   "target_scale": scale, "samples": records,
                   "source_model_root": str(source_root / "stent_models"),
                   "projection_provenance": provenance,
                   "base_dataset": str(base_dir),
                   "base_manifest_sha256": hashlib.sha256((base_dir / "manifest.json").read_bytes()).hexdigest()}
        (temporary / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_dataset(temporary)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    print(export_stent_model_dataset(args.base_dir, args.source_root, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
