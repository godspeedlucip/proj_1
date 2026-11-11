"""Create fold-isolated paired views for nnU-Net without exposing test images."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from .manifest import Manifest, FoldSplit, input_digest, json_digest, load_splits, select_cases, validate_data
from .nifti_io import SpatialMeta, VolumeCase, load_volume
from .prior_stats import PriorStats, fit_prior_stats, validate_prior_stats
from .provenance import file_sha256


PREPARED_SCHEMA = "phase3_prepared_fold_v1"


@dataclass(frozen=True)
class PreparedFold:
    schema_version: str
    dataset_name: str
    source_fold: int
    nnunet_fold: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    raw_dir: Path
    preprocessed_dir: Path
    results_dir: Path
    case_groups: dict[str, str]
    manifest_digest: str
    split_digest: str
    data_digest: str
    stats_digest: str
    stats: PriorStats
    plans_name: str | None = None
    config_digest: str | None = None
    preprocessed_complete: bool = False
    source_cases: dict[str, dict[str, str]] = field(default_factory=dict)
    preprocessed_file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def raw_root(self) -> Path:
        return self.raw_dir.parent

    @property
    def preprocessed_root(self) -> Path:
        return self.preprocessed_dir.parent

    @property
    def results_root(self) -> Path:
        return self.results_dir.parent

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "source_fold": self.source_fold,
            "nnunet_fold": self.nnunet_fold,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "raw_dir": str(self.raw_dir),
            "preprocessed_dir": str(self.preprocessed_dir),
            "results_dir": str(self.results_dir),
            "case_groups": dict(self.case_groups),
            "manifest_digest": self.manifest_digest,
            "split_digest": self.split_digest,
            "data_digest": self.data_digest,
            "stats_digest": self.stats_digest,
            "stats": self.stats.to_dict(),
            "plans_name": self.plans_name,
            "config_digest": self.config_digest,
            "preprocessed_complete": self.preprocessed_complete,
            "source_cases": self.source_cases,
            "preprocessed_file_hashes": self.preprocessed_file_hashes,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> PreparedFold:
        if raw.get("schema_version") != PREPARED_SCHEMA:
            raise ValueError("Unknown prepared fold schema")
        converted = dict(raw)
        for key in ("train_ids", "validation_ids", "test_ids"):
            converted[key] = tuple(converted[key])
        for key in ("raw_dir", "preprocessed_dir", "results_dir"):
            converted[key] = Path(converted[key])
        converted["stats"] = PriorStats.from_dict(converted["stats"])
        converted.setdefault("source_cases", {})
        converted.setdefault("preprocessed_file_hashes", {})
        return cls(**converted)


def make_views(volume: VolumeCase, stats: PriorStats) -> np.ndarray:
    """Match the old case z-score and training-percentile intensity formulas."""
    x = volume.image_raw
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite FDK image")
    mean = x.mean(dtype=np.float64)
    std = x.std(dtype=np.float64, ddof=0)
    zscore = ((x.astype(np.float64) - mean) / max(std, 1e-6)).astype(np.float32)
    percentile = np.clip((x - stats.lo) / (stats.hi - stats.lo), 0, 1).astype(np.float32)
    return np.stack((zscore, percentile), axis=0)


def _write_view(array: np.ndarray, meta: SpatialMeta, path: Path) -> None:
    if array.shape != tuple(reversed(meta.original_shape)):
        raise ValueError("View shape differs from source image")
    header = meta.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    nii = nib.Nifti1Image(array.transpose(2, 1, 0), meta.affine, header)
    qform, qcode = meta.header.get_qform(coded=True)
    sform, scode = meta.header.get_sform(coded=True)
    if qcode:
        nii.set_qform(qform, int(qcode))
    if scode:
        nii.set_sform(sform, int(scode))
    nii.header.set_slope_inter(1.0, 0.0)
    nib.save(nii, path)


def prepare_raw_fold(manifest: Manifest, splits: dict[int, FoldSplit], *, source_fold: int,
                     dataset_id: int, output: Path, stats_seed: int = 2026) -> PreparedFold:
    """Write one fold's train/validation images; retain test only in provenance."""
    if type(source_fold) is not int or source_fold not in splits:
        raise ValueError(f"Unknown source fold: {source_fold}")
    if type(dataset_id) is not int or not 1 <= dataset_id <= 999:
        raise ValueError("dataset_id must be an integer from 1 to 999")
    if stats_seed != 2026:
        raise ValueError("Preparation statistics seed is fixed at 2026")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    validate_data(manifest, splits, formal=False)
    split = splits[source_fold]
    stats = fit_prior_stats(manifest, split, seed=stats_seed)
    validate_prior_stats(stats, manifest, split)
    dataset_name = f"Dataset{dataset_id:03d}_Phase3F{source_fold:02d}"
    raw_dir = output / "raw" / dataset_name
    preprocessed_dir = output / "preprocessed" / dataset_name
    results_dir = output / "results" / dataset_name
    train_cases = select_cases(manifest, split, "train")
    validation_cases = select_cases(manifest, split, "validation")
    test_cases = select_cases(manifest, split, "test")
    train_ids = tuple(sorted(c.case_id for c in train_cases))
    validation_ids = tuple(sorted(c.case_id for c in validation_cases))
    test_ids = tuple(sorted(c.case_id for c in test_cases))
    if not train_ids or not validation_ids or not test_ids:
        raise ValueError("Each fold partition requires at least one case")
    prepared = PreparedFold(
        PREPARED_SCHEMA, dataset_name, source_fold, 0, train_ids, validation_ids, test_ids,
        raw_dir, preprocessed_dir, results_dir,
        {c.case_id: c.group_id for c in manifest.cases},
        json_digest(manifest.to_dict()), json_digest(split.to_dict()),
        input_digest(manifest, set(split.train) | set(split.validation), include_labels=True),
        json_digest(stats.to_dict()), stats,
        source_cases={case.case_id: {
            "image": str(case.image), "label": str(case.label),
            "image_sha256": file_sha256(case.image),
            "label_sha256": file_sha256(case.label),
        } for case in (*train_cases, *validation_cases)},
    )
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=False)
    (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=False)
    preprocessed_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)
    for case in (*train_cases, *validation_cases):
        volume = load_volume(case, require_label=True)
        views = make_views(volume, stats)
        for channel, view in enumerate(views):
            _write_view(view, volume.meta, raw_dir / "imagesTr" / f"{case.case_id}_{channel:04d}.nii.gz")
        shutil.copy2(case.label, raw_dir / "labelsTr" / f"{case.case_id}.nii.gz")
    dataset_json = {
        "channel_names": {"0": "noNorm", "1": "noNorm"},
        "labels": {"background": 0, "stent_struts": 1, "ignore": 2},
        "numTraining": len(train_ids) + len(validation_ids),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "FDKNiftiIO",
        "phase3_channel_semantics": ["case_zscore", "train_group_percentile"],
    }
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    split_json = [{"train": list(train_ids), "val": list(validation_ids)}]
    (preprocessed_dir / "splits_final.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
    (output / "prepared.json").write_text(json.dumps(prepared.to_dict(), indent=2), encoding="utf-8")
    return prepared
