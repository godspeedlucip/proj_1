"""Adapt a Phase 3 nnU-Net raw export without changing its NIfTI grids."""

from __future__ import annotations

import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from .config import Phase3Config, resolve_config
from .manifest import load_manifest, manifest_from_dict, load_splits


def import_dataset601(source: Path, output: Path,
                      config: Phase3Config) -> tuple[Path, Path, Path]:
    source, output = Path(source).resolve(), Path(output).resolve()
    if not re.fullmatch(r"Dataset[0-9]{3}_Phase3[A-Za-z0-9_]+", source.name) or not source.is_dir():
        raise ValueError("Expected a Phase 3 nnU-Net raw DatasetNNN directory")
    if output.exists():
        raise FileExistsError(output)
    dataset = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    if (dataset.get("channel_names") != {"0": "FDK"}
            or dataset.get("labels") != {"background": 0, "stent_and_markers": 1}
            or dataset.get("file_ending") != ".nii.gz"):
        raise ValueError("Phase 3 FDK channels or marker label semantics differ")
    mapping = json.loads((source / "source_mapping.json").read_text(encoding="utf-8"))
    selected = mapping.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Phase 3 FDK source mapping is empty")
    original = load_manifest(mapping["source_manifest"], require_labels=True)
    if original.target_semantics != "stent_and_markers":
        raise ValueError("Phase 3 FDK source manifest must include stent and markers")
    source_cases = {row["case_id"]: row for row in original.to_dict()["cases"]}
    original_split = json.loads((source / "splits_final.json").read_text(encoding="utf-8"))
    if (not isinstance(original_split, list) or len(original_split) != 1
            or set(original_split[0]) != {"train", "val"}):
        raise ValueError("Phase 3 FDK requires one explicit train/validation split")
    split_ids = {"train": set(original_split[0]["train"]),
                 "validation": set(original_split[0]["val"])}
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("Phase 3 FDK train/validation split overlaps")

    cases, group_splits = [], {key: set() for key in ("train", "validation", "test")}
    case_ids = {key: set() for key in group_splits}
    seen_ids = set()
    group_owner = {}
    measured_spacing = None
    unique_image_inodes = set()
    for item in selected:
        case_id, group, split = item["case_id"], item["group_id"], item["split"]
        if split not in case_ids or case_id not in source_cases or case_id in seen_ids:
            raise ValueError(f"Invalid or duplicate Phase 3 FDK case: {case_id}")
        if source_cases[case_id]["group_id"] != group or source_cases[case_id]["angles_deg"] != item["angles_deg"]:
            raise ValueError(f"Phase 3 FDK source identity differs: {case_id}")
        if group in group_owner and group_owner[group] != split:
            raise ValueError(f"Phase 3 FDK group crosses split: {group}")
        group_owner[group] = split
        case_ids[split].add(case_id)
        seen_ids.add(case_id)
        group_splits[split].add(group)
        image_folder = "imagesTs" if split == "test" else "imagesTr"
        image = source / image_folder / f"{case_id}_0000.nii.gz"
        label = (Path(item["source_label"]) if split == "test"
                 else source / "labelsTr" / f"{case_id}.nii.gz")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Phase 3 FDK image or label missing: {case_id}")
        if not image.samefile(item["source_image"]) or not label.samefile(item["source_label"]):
            raise ValueError(f"Phase 3 FDK source mapping differs from exported files: {case_id}")
        unique_image_inodes.add((image.stat().st_dev, image.stat().st_ino))
        nii = nib.load(str(image))
        label_nii = nib.load(str(label))
        if (len(nii.shape) != 3 or nii.shape != label_nii.shape
                or nii.header.get_xyzt_units()[0] != "mm"
                or label_nii.header.get_xyzt_units()[0] != "mm"
                or not np.allclose(nii.affine, label_nii.affine, atol=1e-4, rtol=0)):
            raise ValueError(f"Phase 3 FDK image/marker label grid differs: {case_id}")
        spacing = np.linalg.norm(nii.affine[:3, :3], axis=0)[::-1]
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError(f"Invalid Phase 3 FDK spacing: {case_id}")
        if measured_spacing is None:
            measured_spacing = spacing
        elif not np.allclose(spacing, measured_spacing, atol=1e-4, rtol=0):
            raise ValueError("Phase 3 FDK voxel spacing differs across cases")
        row = dict(source_cases[case_id])
        row.update(image=str(image), label=str(label))
        cases.append(row)

    if (case_ids["train"] != split_ids["train"]
            or case_ids["validation"] != split_ids["validation"]
            or len(case_ids["train"] | case_ids["validation"]) != dataset.get("numTraining")
            or case_ids["test"] != {p.name[:-12] for p in (source / "imagesTs").glob("*_0000.nii.gz")}
            or any(not values for values in group_splits.values())
            or len(cases) != mapping.get("exported_cases", mapping.get("unique_cases"))
            or len(unique_image_inodes) != mapping.get("unique_cases")
            or {key: len(values) for key, values in case_ids.items()} != mapping.get("split_counts")):
        raise ValueError("Phase 3 FDK split or case inventory differs from source mapping")
    if {p.name[:-12] for p in (source / "imagesTr").glob("*_0000.nii.gz")} != case_ids["train"] | case_ids["validation"]:
        raise ValueError("Phase 3 FDK training image inventory differs from split")
    if {p.name[:-7] for p in (source / "labelsTr").glob("*.nii.gz")} != case_ids["train"] | case_ids["validation"]:
        raise ValueError("Phase 3 FDK label inventory differs from split")

    manifest_raw = original.to_dict()
    manifest_raw["cases"] = sorted(cases, key=lambda row: row["case_id"])
    manifest_from_dict(manifest_raw, output)
    split_raw = {"schema_version": "fdk_stent_splits_v1", "folds": {"0": {
        key: sorted(groups) for key, groups in group_splits.items()}}}
    adapted_config = config.to_dict()
    adapted_config["normalization"]["spacing_mm"] = [round(float(axis), 6) for axis in measured_spacing]
    resolve_config(adapted_config)
    output.mkdir(parents=True, exist_ok=False)
    paths = (output / "manifest.json", output / "splits.json", output / "phase3_config.json")
    for path, value in zip(paths, (manifest_raw, split_raw, adapted_config)):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    load_splits(paths[1])
    return paths
