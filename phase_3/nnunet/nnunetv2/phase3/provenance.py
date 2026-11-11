"""Content hashes used to bind prepared data, predictions and evaluations."""

from pathlib import Path
import hashlib


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_provenance(record: dict, manifest, split, *, source_fold: int,
                                   split_name: str) -> None:
    """Bind experimental predictions to exact cases, source files, grid and checkpoint."""
    import numpy as np
    from .manifest import json_digest, select_cases
    from .nifti_io import read_nifti, same_grid

    if (record.get("schema_version") != "phase3_predictions_v1"
            or record.get("mode") != "experimental"
            or record.get("source_fold") != source_fold
            or record.get("split_name") != split_name
            or record.get("split_digest") != json_digest(split.to_dict())
            or record.get("manifest_digest") != json_digest(manifest.to_dict())):
        raise ValueError("Prediction provenance does not match experimental fold and split")
    cases = select_cases(manifest, split, split_name)
    ids = {case.case_id for case in cases}
    if (set(record.get("outputs", {})) != ids
            or set(record.get("image_sha256", {})) != ids
            or set(record.get("label_sha256", {})) != ids
            or record.get("case_groups") != {case.case_id: case.group_id for case in cases}
            or record.get("projection_sources") != {case.case_id: case.projection_source for case in cases}):
        raise ValueError("Prediction cases and source groups differ from requested evaluation")
    checkpoint = Path(record["checkpoint_path"])
    if file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise ValueError("Prediction checkpoint hash differs")
    for case in cases:
        if (case.label is None or file_sha256(case.image) != record["image_sha256"][case.case_id]
                or file_sha256(case.label) != record["label_sha256"][case.case_id]):
            raise ValueError(f"Prediction input or label hash differs: {case.case_id}")
        paths = record["outputs"][case.case_id]
        probability_path, mask_path = Path(paths["probability_path"]), Path(paths["mask_path"])
        if (file_sha256(probability_path) != paths["probability_sha256"]
                or file_sha256(mask_path) != paths["mask_sha256"]):
            raise ValueError(f"Prediction output hash differs: {case.case_id}")
        _, original_meta = read_nifti(case.image)
        _, label_meta = read_nifti(case.label)
        probability, probability_meta = read_nifti(probability_path)
        mask, mask_meta = read_nifti(mask_path)
        if (not same_grid(original_meta, label_meta)
                or not same_grid(original_meta, probability_meta)
                or not same_grid(original_meta, mask_meta)):
            raise ValueError(f"Prediction physical grid differs: {case.case_id}")
        if (not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
                or not np.isin(mask, (0, 1)).all() or not np.array_equal(mask, probability > .5)):
            raise ValueError(f"Prediction probability and mask disagree: {case.case_id}")
