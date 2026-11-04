"""Opt-in real nnU-Net command smoke; requires an exported synthetic handoff package."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_marker_guided_real_nnunet_pipeline(tmp_path):
    if os.environ.get("MARKER_GUIDED_RUN_INTEGRATION") != "1":
        pytest.skip("set MARKER_GUIDED_RUN_INTEGRATION=1 and provide a synthetic Phase 1 handoff package")
    required = ("MARKER_GUIDED_MANIFEST", "MARKER_GUIDED_PACKAGE", "MARKER_GUIDED_FOLD",
                "MARKER_GUIDED_DATASET_ID", "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        pytest.fail(f"integration environment is missing: {', '.join(missing)}")
    commands = ("nnUNetv2_plan_and_preprocess", "nnUNetv2_train", "nnUNetv2_predict")
    absent = [command for command in commands if shutil.which(command) is None]
    if absent:
        pytest.fail(f"install phase_2/nnUNet in the active environment; missing console commands: {absent}")
    dataset_id = int(os.environ["MARKER_GUIDED_DATASET_ID"])
    fold = os.environ["MARKER_GUIDED_FOLD"]
    dataset_name = f"Dataset{dataset_id:03d}_MarkerGuided_{fold}"
    raw_dataset = Path(os.environ["nnUNet_raw"]) / dataset_name
    subprocess.run([sys.executable, "-m", "phase_2.tools.marker_guided.cli", "build-e5u",
                    "--manifest", os.environ["MARKER_GUIDED_MANIFEST"],
                    "--package", os.environ["MARKER_GUIDED_PACKAGE"],
                    "--phase2-fold-id", fold, "--dataset-id", str(dataset_id),
                    "--output-dir", str(raw_dataset)], check=True)
    env = os.environ.copy()
    env["nnUNet_compile"] = "false"
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"],
                   check=True, env=env)
    preprocessed_dataset = Path(env["nnUNet_preprocessed"]) / dataset_name
    split_file = preprocessed_dataset / "splits_final.json"
    if not split_file.is_file():
        pytest.fail(f"planner did not copy group-wise split file to {split_file}")
    subprocess.run(["nnUNetv2_train", str(dataset_id), "2d", "0", "-tr",
                    "nnUNetTrainerMarkerGuidedSmoke", "-device", "cpu"], check=True, env=env)
    results_root = Path(env["nnUNet_results"])
    checkpoints = list(results_root.rglob("checkpoint_final.pth"))
    if not checkpoints:
        pytest.fail(f"smoke training did not produce checkpoint_final.pth under {results_root}")
    output = tmp_path / "predictions"
    subprocess.run(["nnUNetv2_predict", "-i", str(raw_dataset / "imagesTs"), "-o", str(output),
                    "-d", str(dataset_id), "-c", "2d", "-f", "0", "-tr",
                    "nnUNetTrainerMarkerGuidedSmoke", "-device", "cpu"], check=True, env=env)
    predictions = list(output.glob("*.tif")) + list(output.glob("*.nii.gz"))
    if not predictions:
        pytest.fail(f"sliding-window inference produced no segmentation in {output}")
