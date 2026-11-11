"""Public Phase 3 FDK preparation, training, prediction and evaluation CLI."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re

import torch

from .config import Phase3Config, resolve_config
from .data_conversion import PreparedFold, prepare_raw_fold
from .evaluation import aggregate_group_metrics, compute_case_metrics
from .inference import Prediction, predict_volume, write_prediction
from .launch import launch_training
from .manifest import json_digest, load_manifest, load_splits, select_cases, validate_data
from .nifti_io import VolumeCase, load_volume, read_nifti
from .planning import prepare_training_fold
from .prior_stats import PriorStats
from .prior_weights import obtain_prior_weights, prepare_prior_artifact
from .provenance import file_sha256, validate_prediction_provenance
from .runtime import verify_runtime


PRESET_NAMES = ("baseline", "swin", "tubular", "projection", "swin_tubular",
                "swin_projection", "tubular_projection", "full", "random_prior",
                "h00", "h01", "h10", "h11", "no_ds", "smoke")
PRESET_DIR = Path(__file__).resolve().parents[2] / "configs" / "phase3"


def _merge_overrides(config: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if key not in config:
            raise ValueError(f"Unknown configuration override: {key}")
        if isinstance(value, dict):
            if not isinstance(config[key], dict):
                raise ValueError(f"Override changes configuration section type: {key}")
            _merge_overrides(config[key], value)
        else:
            config[key] = value


def load_preset(name: str, *, overrides: dict | None = None) -> Phase3Config:
    if name not in PRESET_NAMES:
        raise ValueError(f"Unknown preset: {name}")
    raw = json.loads((PRESET_DIR / f"{name}.json").read_text(encoding="utf-8"))
    if overrides:
        _merge_overrides(raw, overrides)
    return resolve_config(raw)


def _load_config(selector: str, seed: int | None) -> Phase3Config:
    overrides = {"seed": seed} if seed is not None else None
    path = Path(selector)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if overrides:
            _merge_overrides(raw, overrides)
        return resolve_config(raw)
    return load_preset(selector, overrides=overrides)


def _predict_command(args) -> int:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_dir = Path(args.model_dir).resolve()
    plans = json.loads((model_dir / "plans.json").read_text(encoding="utf-8"))
    config = resolve_config(plans["phase3"]["config"])
    stats = PriorStats.from_dict(plans["phase3"]["stats"])
    checkpoint_path = model_dir / "fold_0" / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(args.device)
    predictor = nnUNetPredictor(device=device, use_mirroring=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,),
                                                   checkpoint_name=args.checkpoint)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.mode == "external":
        if args.image is None or args.manifest or args.splits:
            raise ValueError("External prediction requires one --image and no experimental split")
        case_id = args.case_id or "external_0"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
            raise ValueError("Unsafe external case ID")
        image, meta = read_nifti(args.image)
        volume = VolumeCase(case_id, "external", image, None, meta)
        prediction = predict_volume(predictor.network, volume, plans=plans, config=config,
                                    stats=stats, device=device)
        probability_path, mask_path = write_prediction(prediction, case_id=case_id, output=output)
        record = {
            "schema_version": "phase3_predictions_v1", "mode": "external",
            "case_id": case_id,
            "input_image": str(Path(args.image).resolve()),
            "input_sha256": file_sha256(args.image),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "config_digest": plans["phase3"]["config_digest"],
            "outputs": {
                "probability_path": str(probability_path),
                "mask_path": str(mask_path),
                "probability_sha256": file_sha256(probability_path),
                "mask_sha256": file_sha256(mask_path),
            },
        }
        (output / "prediction_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return 0
    if args.image is not None or not all((args.manifest, args.splits,
                                           args.source_fold is not None, args.split_name)):
        raise ValueError("Experimental prediction requires manifest, splits, source fold and split name")
    manifest = load_manifest(args.manifest, require_labels=True)
    splits = load_splits(args.splits)
    split = splits[args.source_fold]
    if (args.source_fold != plans["phase3"]["source_fold"]
            or json_digest(manifest.to_dict()) != plans["phase3"]["manifest_digest"]
            or json_digest(split.to_dict()) != plans["phase3"]["split_digest"]):
        raise ValueError("Experimental prediction data differs from model fold")
    cases = select_cases(manifest, split, args.split_name)
    record = {
        "schema_version": "phase3_predictions_v1", "mode": "experimental",
        "source_fold": args.source_fold, "split_name": args.split_name,
        "split_digest": json_digest(split.to_dict()),
        "manifest_digest": json_digest(manifest.to_dict()),
        "case_groups": {case.case_id: case.group_id for case in cases},
        "projection_sources": {case.case_id: case.projection_source for case in cases},
        "image_sha256": {case.case_id: file_sha256(case.image) for case in cases},
        "label_sha256": {case.case_id: file_sha256(case.label) for case in cases},
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "outputs": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    for case in cases:
        volume = load_volume(case, require_label=False)
        prediction = predict_volume(predictor.network, volume, plans=plans, config=config,
                                    stats=stats, device=device)
        probability_path, mask_path = write_prediction(prediction, case_id=case.case_id, output=output)
        record["outputs"][case.case_id] = {
            "probability_path": str(probability_path), "mask_path": str(mask_path),
            "probability_sha256": file_sha256(probability_path),
            "mask_sha256": file_sha256(mask_path),
        }
    validate_prediction_provenance(record, manifest, split, source_fold=args.source_fold,
                                   split_name=args.split_name)
    (output / "prediction_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


def _evaluate_command(args) -> int:
    manifest = load_manifest(args.manifest, require_labels=True)
    split = load_splits(args.splits)[args.source_fold]
    predictions = Path(args.predictions).resolve()
    record = json.loads((predictions / "prediction_run.json").read_text(encoding="utf-8"))
    validate_prediction_provenance(record, manifest, split, source_fold=args.source_fold,
                                   split_name=args.split_name)
    rows = []
    for case in select_cases(manifest, split, args.split_name):
        paths = record["outputs"][case.case_id]
        probability, meta = read_nifti(paths["probability_path"])
        mask, _ = read_nifti(paths["mask_path"])
        truth, _ = read_nifti(case.label)
        rows.append(compute_case_metrics(Prediction(probability, mask, meta, {}), truth,
                                         case_id=case.case_id, group_id=case.group_id))
    summary = aggregate_group_metrics(rows)
    summary["provenance"] = {key: record[key] for key in
                             ("checkpoint_sha256", "source_fold", "split_digest", "split_name")}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics_per_case.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    (output / "metrics_per_group.json").write_text(json.dumps(summary["groups"], indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase3-nnunet")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--splits", required=True)
    validate.add_argument("--run-kind", choices=("smoke", "formal"), default="formal")
    prior = commands.add_parser("prepare-prior")
    source = prior.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights")
    source.add_argument("--download-official", action="store_true")
    prior.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--splits", required=True)
    prepare.add_argument("--source-fold", type=int, required=True)
    prepare.add_argument("--dataset-id", type=int, required=True)
    prepare.add_argument("--config", required=True, help="Preset name or full JSON config")
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--prepared", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--prior-artifact")
    train.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--checkpoint", default="checkpoint_best.pth")
    predict.add_argument("--mode", choices=("external", "experimental"), required=True)
    predict.add_argument("--image")
    predict.add_argument("--case-id")
    predict.add_argument("--manifest")
    predict.add_argument("--splits")
    predict.add_argument("--source-fold", type=int)
    predict.add_argument("--split-name", choices=("train", "validation", "test"))
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    predict.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--splits", required=True)
    evaluate.add_argument("--source-fold", type=int, required=True)
    evaluate.add_argument("--split-name", choices=("validation", "test"), required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    verify_runtime(Path(__file__).resolve().parents[2])
    if args.command == "validate-data":
        manifest = load_manifest(args.manifest, require_labels=True)
        report = validate_data(manifest, load_splits(args.splits), formal=args.run_kind == "formal")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "prepare-prior":
        artifact = Path(args.output).resolve()
        weights = obtain_prior_weights(weights=args.weights,
                                       download_official=args.download_official,
                                       output=artifact.parent)
        print(prepare_prior_artifact(weights, artifact))
        return 0
    if args.command == "prepare":
        config = _load_config(args.config, args.seed)
        manifest = load_manifest(args.manifest, require_labels=True)
        splits = load_splits(args.splits)
        validate_data(manifest, splits, formal=config.run_kind == "formal")
        prepared = prepare_raw_fold(manifest, splits, source_fold=args.source_fold,
                                    dataset_id=args.dataset_id, output=Path(args.output))
        prepared = prepare_training_fold(prepared, config)
        print(prepared.raw_dir.parent.parent / "prepared.json")
        return 0
    if args.command == "train":
        prepared = PreparedFold.from_dict(json.loads(Path(args.prepared).read_text(encoding="utf-8")))
        config = _load_config(args.config, args.seed)
        return launch_training(prepared, config, seed=config.seed, device=args.device,
                               prior_artifact=Path(args.prior_artifact) if args.prior_artifact else None,
                               resume=args.resume)
    if args.command == "predict":
        return _predict_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
