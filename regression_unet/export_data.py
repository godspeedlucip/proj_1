"""Materialize the marker-guided regression dataset as an independent TIFF package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from tools.marker_guided.build_prior_channels import build_marker_channels

from .data import ProjectionSample, compute_target_scale, load_samples
from .train import DEFAULT_MANIFEST


DEFAULT_DESTINATION = (Path(__file__).resolve().parents[3]
                       / "training_data/training/generated/phase2_marker_fusion_regression_v1")


def _write_tiff(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(values, dtype=np.float32),
                     compression="zlib", compressionargs={"level": 1})


def verify_dataset(directory: str | Path) -> dict[str, int]:
    """Check every manifest reference and the numeric range of shared arrays."""
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in ("phase2_projection_regression_v1",
                                              "phase2_projection_regression_v2"):
        raise ValueError("invalid regression dataset schema")
    counts = {"train": 0, "val": 0, "test": 0}
    arrays = {"target": set(), "point": set(), "band": set(), "validity": set()}
    array_shapes = {}
    for sample in manifest["samples"]:
        split = sample["split"]
        counts[split] += 1
        paths = {}
        for key in ("image", "target", "point", "band", "validity"):
            relative = Path(sample[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"nonlocal {key} path in dataset manifest")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing {key}: {path}")
            paths[key] = path
            if key in arrays:
                arrays[key].add(path)
        with tifffile.TiffFile(paths["image"]) as image_tiff:
            image_page = image_tiff.pages[0]
            if len(image_page.shape) != 2 or image_page.dtype != np.dtype("float32"):
                raise ValueError(f"invalid image TIFF: {paths['image']}")
            shape = image_page.shape
        for key in ("target", "point", "band", "validity"):
            path = paths[key]
            if path not in array_shapes:
                with tifffile.TiffFile(path) as array_tiff:
                    array_shapes[path] = array_tiff.pages[0].shape
            if array_shapes[path] != shape:
                raise ValueError(f"image/{'prior' if key != 'target' else 'target'} shape mismatch: {path}")
    if counts != manifest["counts"]:
        raise ValueError("manifest counts disagree with samples")
    for key, paths in arrays.items():
        for path in paths:
            values = tifffile.imread(path)
            if (values.dtype != np.float32 or values.ndim != 2
                    or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
                raise ValueError(f"invalid {key} values: {path}")
    return counts


def export_dataset(source_manifest: str | Path, destination: str | Path) -> Path:
    """Copy inputs and write shared normalized targets/priors, then publish atomically."""
    source_manifest = Path(source_manifest).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"unfinished export already exists: {temporary}")
    by_split = {split: load_samples(source_manifest, split) for split in ("train", "val", "test")}
    samples = [(split, sample) for split, group in by_split.items() for sample in group]
    target_scale = compute_target_scale(by_split["train"])
    source_root = source_manifest.parent.parent
    image_root = source_root / "stacked"
    target_root = source_root / "stent_projections"
    image_bytes = sum(sample.image_path.stat().st_size for _, sample in samples)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < image_bytes + 1024**3:
        raise OSError("insufficient free space for owned image copies")
    priors_by_target: dict[Path, tuple[tuple[float, float], tuple[float, float]]] = {}
    for _, sample in samples:
        previous = priors_by_target.setdefault(sample.target_path, sample.marker_xy)
        if previous != sample.marker_xy:
            raise ValueError(f"shared target has inconsistent marker coordinates: {sample.target_path}")
    temporary.mkdir()
    try:
        shared: dict[Path, dict[str, str]] = {}
        validity_paths: dict[tuple[int, int], str] = {}
        for target_path, marker_xy in sorted(priors_by_target.items()):
            target_relative = target_path.relative_to(target_root)
            raw = tifffile.imread(target_path).astype(np.float32)
            if raw.ndim != 2 or not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError(f"invalid source implant projection: {target_path}")
            output_target = Path("targets") / target_relative
            _write_tiff(temporary / output_target, np.clip(raw / target_scale, 0, 1))
            point, band, validity = build_marker_channels(
                {"pair_status": "reliable", "marker_xy": marker_xy,
                 "error_radius_px": [0.0, 0.0]}, raw.shape)
            base = Path("priors") / target_relative.parts[0] / target_relative.parts[1]
            point_relative = base / f"{target_path.stem}.point.tif"
            band_relative = base / f"{target_path.stem}.band.tif"
            _write_tiff(temporary / point_relative, point)
            _write_tiff(temporary / band_relative, band)
            validity_relative = validity_paths.get(raw.shape)
            if validity_relative is None:
                validity_relative = str(Path("priors") / f"validity_{raw.shape[0]}x{raw.shape[1]}.tif")
                _write_tiff(temporary / validity_relative, validity)
                validity_paths[raw.shape] = validity_relative
            shared[target_path] = {"target": output_target.as_posix(),
                                   "point": point_relative.as_posix(),
                                   "band": band_relative.as_posix(),
                                   "validity": validity_relative}
        records = []
        split_ids = {name: [] for name in ("train", "val", "test")}
        for index, (split, sample) in enumerate(samples, 1):
            image_relative = Path("images") / sample.image_path.relative_to(image_root)
            output_image = temporary / image_relative
            output_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.image_path, output_image)
            if output_image.stat().st_size != sample.image_path.stat().st_size:
                raise IOError(f"image copy size mismatch: {sample.image_path}")
            sample_id = f"{sample.case_id}_a{sample.image_path.stem}"
            split_ids[split].append(sample_id)
            records.append({"sample_id": sample_id, "case_id": sample.case_id,
                            "group_id": sample.group_id, "split": split,
                            "marker_xy": sample.marker_xy,
                            "image": image_relative.as_posix(), **shared[sample.target_path]})
            if index % 500 == 0 or index == len(samples):
                print(f"Copied {index}/{len(samples)} images", flush=True)
        payload = {"schema_version": "phase2_projection_regression_v1",
                   "target_kind": "normalized_continuous_implant_including_markers",
                   "target_scale": target_scale,
                   "source_manifest": str(source_manifest),
                   "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                   "counts": {name: len(ids) for name, ids in split_ids.items()},
                   "splits": split_ids, "samples": records}
        (temporary / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_dataset(temporary)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    result = export_dataset(args.source_manifest, args.output_dir)
    print(f"Published {result}", flush=True)


if __name__ == "__main__":
    main()
