"""Create a Phase 2 E5-U dataset from original stacked TIFFs and marker truth.

The Phase 1 manifest is an index into the original stacked projections and
contains marker positions projected from the source stent geometry. No Phase 1
image or mask is copied into this dataset. The existing Phase 2 validator and
prior rasterizer provide the data contract and three prior channels.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile

PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PHASE2_ROOT, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pipelines.phase1_manifest import build_phase1_triplets, build_physical_manifest
from tools.marker_guided.build_prior_channels import PriorConfig, build_marker_channels
from tools.marker_guided.validate_dataset import validate_prediction_package


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_path(manifest_path: Path, relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def make_handoff(source_manifest: Path, fold_id: str,
                 progress: Callable[[str], None] | None = None) -> tuple[dict, dict]:
    """Translate Phase 1 geometry truth into the existing Phase 2 contract."""
    source_manifest = Path(source_manifest).resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("schema_version") != "phase1_real_v1":
        raise ValueError("source must be a three-view phase1_real_v1 manifest")
    try:
        splits = source["folds"][fold_id]
    except KeyError as exc:
        raise ValueError(f"missing Phase 1 fold {fold_id!r}") from exc
    group_to_split = {}
    for split in ("train", "val", "test"):
        for group in splits[split]:
            if group in group_to_split:
                raise ValueError(f"group {group!r} occurs in multiple splits")
            group_to_split[group] = split

    model_id = "proj1_marker_geometry_oracle_v1"
    samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    total_views = len(source["cases"]) * 3
    if progress:
        progress(f"Indexing {total_views} views from {source_manifest}")
    for case in source["cases"]:
        case_id, group_id = case["case_id"], case["group_id"]
        split = group_to_split.get(group_id)
        if split is None:
            raise ValueError(f"case {case_id!r} has no split")
        views = case["views"]
        if len(views) != 3 or len({v["view_id"] for v in views}) != 3:
            raise ValueError(f"case {case_id!r} needs exactly three distinct views")
        geometry_id = case["metadata"]["geometry_id"]
        prediction_views = []
        for view in views:
            view_id = view["view_id"]
            if view.get("marker_xy") is None:
                raise ValueError(f"missing marker_xy truth for {(case_id, view_id)}")
            projection = _source_path(source_manifest, view["projection_path"])
            mask = _source_path(source_manifest, view["mask_path"])
            with tifffile.TiffFile(projection) as tiff:
                image_shape = tiff.pages[0].shape
            if len(image_shape) != 2:
                raise ValueError(f"projection must be 2D: {projection}")
            sample = {"case_id": case_id, "view_id": view_id,
                      "nnunet_case_id": f"{case_id}_v{view_id}",
                      "group_id": group_id, "split": split,
                      "projection_path": str(projection),
                      "geometry_id": geometry_id, "angle_deg": view["angle_deg"]}
            if split != "test":
                sample["mask_path"] = str(mask)
            samples.append(sample)
            prediction_views.append({"view_id": view_id,
                                     "image_size": [int(image_shape[0]), int(image_shape[1])],
                                     "marker_xy": view["marker_xy"],
                                     "error_radius_px": [0.0, 0.0],
                                     "q_pair": None, "pair_status": "reliable"})
        records.append({"schema_version": "phase1_marker_case_v1",
                        "case_id": case_id, "group_id": group_id,
                        "model_id": model_id, "geometry_id": geometry_id,
                        "calibration_id": None,
                        "prediction_source": "group_oof" if split == "train" else "held_out",
                        "fold_id": f"{fold_id}_oracle_no_model_fit",
                        "oracle_source": "projected_stent_marker_geometry",
                        "views": prediction_views})
        if progress and (len(samples) % 300 == 0 or len(samples) == total_views):
            progress(f"Indexed {len(samples)}/{total_views} views")

    manifest = {"phase2_fold_id": fold_id, "splits": splits, "samples": samples}
    package = {"phase2_fold_id": fold_id,
               "oracle_source": "projected_stent_marker_geometry",
               "source_manifest": str(source_manifest),
               "source_registry": [{"model_id": model_id,
                                    "model_artifact_id": "proj1_stent_marker_json_geometry_v1",
                                    "pipeline_config_id": "proj1_phase1_manifest_projection_matrix_v1",
                                    "fit_group_ids": [], "selection_group_ids": [],
                                    "oracle_source": "projected_stent_marker_geometry"}],
               "calibration_registry": [], "cases": records}
    if progress:
        progress(f"Validating {len(samples)} views")
    validate_prediction_package(manifest, package, fold_id, progress=progress)
    return manifest, package


def _write_dataset(manifest: dict, package: dict, output_dir: Path,
                   *, fold_id: str, dataset_id: int, prior_config: PriorConfig) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty dataset directory: {output_dir}")
    prior_config.validate()
    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_ts = output_dir / "imagesTs"
    for directory in (images_tr, labels_tr, images_ts):
        directory.mkdir(parents=True, exist_ok=True)
    case_by_id = {case["case_id"]: case for case in package["cases"]}
    mapping = []
    ids = {"train": [], "val": [], "test": []}
    compress = {"compression": "zlib", "compressionargs": {"level": 1}}
    for index, sample in enumerate(manifest["samples"], 1):
        case = case_by_id[sample["case_id"]]
        view = next(v for v in case["views"] if v["view_id"] == sample["view_id"])
        image = tifffile.imread(sample["projection_path"])
        point, band, validity = build_marker_channels(view, image.shape, prior_config)
        nn_id, split = sample["nnunet_case_id"], sample["split"]
        target = images_ts if split == "test" else images_tr
        for channel, pixels in enumerate((image, point, band, validity)):
            tifffile.imwrite(target / f"{nn_id}_{channel:04d}.tif", pixels, **compress)
        if split != "test":
            label = tifffile.imread(sample["mask_path"])
            tifffile.imwrite(labels_tr / f"{nn_id}.tif", label, **compress)
        ids[split].append(nn_id)
        mapping.append({k: sample[k] for k in ("case_id", "view_id", "nnunet_case_id",
                                               "group_id", "split", "geometry_id", "angle_deg")})
        if index % 30 == 0 or index == len(manifest["samples"]):
            print(f"Wrote {index}/{len(manifest['samples'])} views", flush=True)
    name = f"Dataset{dataset_id:03d}_MarkerGuided_{fold_id}"
    _json(output_dir / "dataset.json", {
        "channel_names": {"0": "zscore", "1": "nonorm", "2": "nonorm", "3": "nonorm"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(ids["train"]) + len(ids["val"]),
        "file_ending": ".tif", "name": name,
        "description": "E5-U original projection and oracle marker prior",
        "marker_guided": {"experiment": "E5-U", "phase2_fold_id": fold_id,
                          "channel_order": ["projection", "point", "band", "validity"],
                          "prior_config": asdict(prior_config),
                          "oracle_source": "projected_stent_marker_geometry"}})
    _json(output_dir / "splits_final.json", [{"train": ids["train"], "val": ids["val"]}])
    _json(output_dir / "case_mapping.json", mapping)
    _json(output_dir / "source_registry.json", package["source_registry"])
    _json(output_dir / "ground_truth_provenance.json", {
        "source_manifest": package["source_manifest"],
        "oracle_source": package["oracle_source"],
        "note": "Coordinates are exact projected geometry truth, not Phase 1 model predictions.",
        "fit_group_ids": [], "selection_group_ids": [],
        "error_radius_px": [0.0, 0.0], "q_pair": None})


def build_from_phase1_manifest(source_manifest: Path, handoff_dir: Path, output_dir: Path,
                               *, dataset_id: int = 501, fold_id: str = "fold_0",
                               prior_config: PriorConfig | None = None,
                               metadata_only: bool = False) -> dict:
    """Index original TIFFs, validate oracle truth, and write compressed E5-U data."""
    if dataset_id < 1:
        raise ValueError("dataset_id must be positive")
    handoff_dir, output_dir = Path(handoff_dir), Path(output_dir)
    if handoff_dir.exists() and any(handoff_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty handoff directory: {handoff_dir}")
    if not metadata_only and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty dataset directory: {output_dir}")
    print(f"Loading Phase 1 manifest: {source_manifest}", flush=True)
    manifest, package = make_handoff(Path(source_manifest), fold_id,
                                     progress=lambda message: print(message, flush=True))
    handoff_dir.mkdir(parents=True, exist_ok=True)
    _json(handoff_dir / "manifest.json", manifest)
    _json(handoff_dir / "package.json", package)
    if not metadata_only:
        _write_dataset(manifest, package, output_dir, fold_id=fold_id,
                       dataset_id=dataset_id, prior_config=prior_config or PriorConfig())
    return {"source_manifest": str(Path(source_manifest).resolve()),
            "handoff_dir": str(handoff_dir.resolve()),
            "dataset_dir": None if metadata_only else str(output_dir.resolve()),
            "case_count": len(package["cases"]), "sample_count": len(manifest["samples"]),
            "split_views": {split: sum(s["split"] == split for s in manifest["samples"])
                            for split in ("train", "val", "test")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (PROJECT_ROOT / "training_data" / "training" / "nnunet" / "phase2"
                      / "Dataset503_rebuild_stage")
    parser.add_argument("--phase1-manifest", type=Path,
                        default=PROJECT_ROOT / "training_data" / "training" / "generated" / "phase1_200" / "phase1" / "manifest.json")
    parser.add_argument("--handoff-dir", type=Path, default=default_output / "ground_truth_fold_0_v2")
    parser.add_argument("--output-dir", type=Path, default=default_output / "raw" / "Dataset503_MarkerGuided_fold_0")
    parser.add_argument("--fold-id", default="fold_0")
    parser.add_argument("--dataset-id", type=int, default=503)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--regenerate-source", action="store_true",
                        help="recreate the Phase 1 index from original stacked TIFFs and stent geometry")
    parser.add_argument("--stacked-root", type=Path, default=PROJECT_ROOT / "training_data" / "training" / "generated" / "phase1_200" / "stacked")
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "training_data" / "training" / "generated" / "phase1_200" / "stent_models")
    parser.add_argument("--val-series", default="stent2")
    parser.add_argument("--nuns", type=int, default=10)
    parser.add_argument("--sigma-0", type=float, default=2.0)
    parser.add_argument("--k-sigma", type=float, default=1.0)
    parser.add_argument("--w-0", type=float, default=3.0)
    parser.add_argument("--k-w", type=float, default=1.0)
    parser.add_argument("--endpoint-taper-px", type=float, default=4.0)
    parser.add_argument("--d-min", type=float, default=1.0)
    args = parser.parse_args(argv)
    source_manifest = args.phase1_manifest
    if args.regenerate_source:
        if args.handoff_dir.exists() and any(args.handoff_dir.iterdir()):
            raise FileExistsError(f"handoff directory is nonempty: {args.handoff_dir}")
        source_dir = args.handoff_dir.parent / f"phase1_regenerated_{args.fold_id}"
        if source_dir.exists() and any(source_dir.iterdir()):
            raise FileExistsError(f"regenerated source directory is nonempty: {source_dir}")
        physical = source_dir / "physical_manifest.json"
        source_manifest = source_dir / "manifest.json"
        build_physical_manifest(args.stacked_root, args.models_root, physical,
                                val_series=args.val_series)
        build_phase1_triplets(physical, source_manifest, nuns=args.nuns)
    cfg = PriorConfig(args.sigma_0, args.k_sigma, args.w_0, args.k_w,
                      args.endpoint_taper_px, args.d_min)
    result = build_from_phase1_manifest(source_manifest, args.handoff_dir, args.output_dir,
                                        dataset_id=args.dataset_id, fold_id=args.fold_id,
                                        prior_config=cfg, metadata_only=args.metadata_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
