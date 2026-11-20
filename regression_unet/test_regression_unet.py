import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import numpy as np
import tifffile
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from regression_unet.model import build_unet, load_plan
from regression_unet.data import ProjectionDataset, compute_target_scale, load_exported_samples, load_samples
from regression_unet.loss import weighted_smooth_l1, mask_depth_loss
from regression_unet.train import main as train_main, run_training
from regression_unet import train as train_module
from regression_unet.predict import load_checkpoint_model, predict_maps, predict_projection
from regression_unet.export_data import export_dataset, verify_dataset
from regression_unet.export_stent_model_data import normalize_stent_targets
from regression_unet.export_mask_depth_data import export_mask_depth_dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PLANS = (Path(__file__).resolve().parents[3] / "training_data/training/nnunet/phase2/preprocessed"
         / "Dataset501_MarkerGuided_fold_0/nnUNetPlans.json")


def test_resume_accepts_relocated_dataset_only_when_manifest_is_identical():
    previous = {"dataset_dir": "/old/data", "dataset_manifest_sha256": "same",
                "model_family": "mask_then_depth", "batch_size": 8}
    current = {**previous, "dataset_dir": "/new/data"}
    assert hasattr(train_module, "resume_config_mismatches")
    assert train_module.resume_config_mismatches(previous, current) == []
    assert "dataset_manifest_sha256" in train_module.resume_config_mismatches(
        previous, {**current, "dataset_manifest_sha256": "changed"})
    assert "batch_size" in train_module.resume_config_mismatches(
        previous, {**current, "batch_size": 4})


def test_launcher_commands_work_without_pythonpath_or_gpu_environment_variables(tmp_path):
    launcher = Path(__file__).with_name("launch.py")
    clean = os.environ.copy()
    clean.pop("PYTHONPATH", None)
    clean.pop("CUDA_VISIBLE_DEVICES", None)
    for command, expected in (("train", "--real-samples"),
                              ("tensorboard", "--logdir"),
                              ("predict", "--markers-json")):
        result = subprocess.run([sys.executable, str(launcher), command, "--help"],
                                cwd=tmp_path, env=clean, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert expected in result.stdout
    result = subprocess.run([sys.executable, str(launcher), "test", "-q", "-k",
                             "stent_model_targets_use_one_training_scale"],
                            cwd=tmp_path, env=clean, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_stent_model_targets_use_one_training_scale_and_preserve_continuous_values(tmp_path):
    train_path = tmp_path / "stent1.tif"
    val_path = tmp_path / "stent2.tif"
    tifffile.imwrite(train_path, np.array([[0.0, 1.0, 2.0]], dtype=np.float32))
    tifffile.imwrite(val_path, np.array([[0.0, 1.0, 4.0]], dtype=np.float32))
    scale = normalize_stent_targets({train_path}, {train_path, val_path})
    assert scale == pytest.approx(2.0)
    np.testing.assert_allclose(tifffile.imread(train_path), [[0.0, 0.5, 1.0]])
    np.testing.assert_allclose(tifffile.imread(val_path), [[0.0, 0.5, 1.0]])


def test_saved_plan_builds_marker_fusion_regression_unet():
    plan = load_plan(PLANS)
    network = build_unet(plan)
    assert network.encoder.stages[0][0].convs[0].conv.in_channels == 1
    assert network.prior_encoder.stem[0].in_channels == 2
    assert set(network.adapters) == {"0", "1"}
    assert network.gate_channel == 3
    assert network.decoder.seg_layers[0].out_channels == 1
    assert len(network.encoder.stages) == 9
    assert network.decoder.deep_supervision is False


def test_small_plan_forward_returns_float_map_at_input_resolution():
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage", "n_conv_per_stage_decoder"):
        architecture[key] = architecture[key][:3] if key != "n_conv_per_stage_decoder" else architecture[key][:2]
    model = build_unet(plan)
    result = model(torch.randn(1, 4, 32, 32))
    assert result.shape == (1, 1, 32, 32)
    assert result.dtype == torch.float32


def test_zero_validity_ignores_marker_prior_but_valid_prior_changes_output():
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    model = build_unet(plan).eval()
    for adapter in model.adapters.values():
        torch.nn.init.normal_(adapter.attention[-1].weight, std=0.1)
    inputs = torch.randn(1, 4, 32, 32)
    without_prior = inputs.clone()
    without_prior[:, 1:3] = 0
    inputs[:, 3] = 0
    without_prior[:, 3] = 0
    with torch.inference_mode():
        gated_a = model(inputs)
        gated_b = model(without_prior)
        inputs[:, 3] = 1
        without_prior[:, 3] = 1
        active_a = model(inputs)
        active_b = model(without_prior)
    assert torch.allclose(gated_a, gated_b, atol=1e-6)
    assert not torch.allclose(active_a, active_b, atol=1e-6)


def test_loader_keeps_older_plain_unet_checkpoints_readable(tmp_path):
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    plain = build_unet(plan, model_family="plain")
    path = tmp_path / "old_checkpoint.pt"
    torch.save({"plans": plan, "model": plain.state_dict(), "config": {}}, path)
    loaded = load_checkpoint_model(path)
    assert isinstance(loaded, PlainConvUNet)
    assert loaded(torch.zeros(1, 4, 32, 32)).shape == (1, 1, 32, 32)


def test_export_creates_owned_continuous_dataset_with_relative_paths(tmp_path):
    source = _paired_fixture(tmp_path / "source")
    original_image = tifffile.imread(source.parent.parent / "stacked/train/ct_001/stent1/001/projections/image/000.tif")
    destination = tmp_path / "output/data/phase2_marker_fusion_regression_v1"
    export_dataset(source, destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["target_scale"] == pytest.approx(0.75)
    assert manifest["counts"] == {"train": 1, "val": 0, "test": 0}
    sample = manifest["samples"][0]
    assert sample["split"] == "train"
    for key in ("image", "target", "point", "band", "validity"):
        assert not Path(sample[key]).is_absolute()
        assert (destination / sample[key]).is_file()
    np.testing.assert_array_equal(tifffile.imread(destination / sample["image"]), original_image)
    target = tifffile.imread(destination / sample["target"])
    assert target.dtype == np.float32
    assert float(target[12, 13]) == pytest.approx(1.0)
    assert float(target.min()) == 0.0
    assert float(target.max()) == 1.0
    assert float(tifffile.imread(destination / sample["point"]).max()) > 0.9
    assert float(tifffile.imread(destination / sample["band"]).max()) > 0.9
    assert np.all(tifffile.imread(destination / sample["validity"]) == 1)
    np.testing.assert_array_equal(tifffile.imread(source.parent.parent / "stacked/train/ct_001/stent1/001/projections/image/000.tif"), original_image)


def test_exported_loader_reads_precomputed_priors_without_rescaling_label_twice(tmp_path):
    source = _paired_fixture(tmp_path / "source")
    destination = export_dataset(source, tmp_path / "output/data/package")
    samples = load_exported_samples(destination, "train")
    assert len(samples) == 1
    assert samples[0].source_scale == pytest.approx(0.75)
    inputs, target = ProjectionDataset(samples, patch_size=(32, 32), training=False,
                                       target_scale=samples[0].source_scale)[0]
    assert float(target[0, 10, 11]) == pytest.approx(0.5)
    assert float(target[0, 12, 13]) == pytest.approx(1.0)
    assert float(inputs[1].max()) > 0.9
    assert float(inputs[2].max()) > 0.9
    assert torch.all(inputs[3] == 1)


def test_export_rejects_conflicting_marker_coordinates_for_shared_target(tmp_path):
    source = _paired_fixture(tmp_path / "source")
    manifest = json.loads(source.read_text())
    repeated = copy.deepcopy(manifest["cases"][0])
    repeated["case_id"] = "ct_002_stent1_001"
    repeated["views"][0]["marker_xy"][0][0] = 9.0
    repeated["views"][0]["projection_path"] = repeated["views"][0]["projection_path"].replace("ct_001", "ct_002")
    manifest["cases"].append(repeated)
    source.write_text(json.dumps(manifest))
    second_image = source.parent.parent / "stacked/train/ct_002/stent1/001/projections/image/000.tif"
    second_image.parent.mkdir(parents=True)
    tifffile.imwrite(second_image, np.zeros((32, 32), dtype=np.float32))
    destination = tmp_path / "output/data/package"
    with pytest.raises(ValueError, match="inconsistent marker coordinates"):
        export_dataset(source, destination)
    assert not destination.exists()
    assert not destination.with_name(".package.tmp").exists()


def test_export_verifier_rejects_prior_shape_mismatch(tmp_path):
    source = _paired_fixture(tmp_path / "source")
    destination = export_dataset(source, tmp_path / "output/data/package")
    sample = json.loads((destination / "manifest.json").read_text())["samples"][0]
    tifffile.imwrite(destination / sample["point"], np.zeros((31, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="prior shape"):
        verify_dataset(destination)


def test_train_cli_uses_owned_package(tmp_path):
    source = _paired_fixture(tmp_path / "source")
    manifest = json.loads(source.read_text())
    val_case = copy.deepcopy(manifest["cases"][0])
    val_case["case_id"] = "ct_001_stent2_001"
    val_case["group_id"] = "stent2"
    val_case["views"][0]["projection_path"] = val_case["views"][0]["projection_path"].replace("stent1", "stent2")
    manifest["cases"].append(val_case)
    source.write_text(json.dumps(manifest))
    root = source.parent.parent
    val_image = root / "stacked/train/ct_001/stent2/001/projections/image/000.tif"
    val_image.parent.mkdir(parents=True)
    tifffile.imwrite(val_image, tifffile.imread(root / "stacked/train/ct_001/stent1/001/projections/image/000.tif"))
    val_target = root / "stent_projections/stent2/001/projections/implant/000.tif"
    val_target.parent.mkdir(parents=True)
    tifffile.imwrite(val_target, np.full((32, 32), 0.25, dtype=np.float32))
    dataset = export_dataset(source, tmp_path / "output/data/package")
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    plans_path = tmp_path / "tiny_plans.json"
    plans_path.write_text(json.dumps(plan))
    output = tmp_path / "trained"
    train_main(["--dataset-dir", str(dataset), "--plans", str(plans_path),
                "--output-dir", str(output), "--epochs", "1",
                "--steps-per-epoch", "1", "--val-steps", "1", "--batch-size", "1",
                "--patch-size", "32", "32", "--num-workers", "0", "--device", "cpu"])
    config = json.loads((output / "run_config.json").read_text())
    assert config["dataset_dir"] == str(dataset)
    assert config["target_scale"] == pytest.approx(0.75)
    assert (output / "checkpoint_best.pt").is_file()


def test_load_plan_rejects_non_2d_architecture(tmp_path):
    plan = {"configurations": {"2d": {"architecture": {"network_class_name": "Other"}}}}
    path = tmp_path / "plans.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="PlainConvUNet"):
        load_plan(path)


def _paired_fixture(tmp_path, *, target_shape=(32, 32), include_target=True):
    root = tmp_path / "phase1_200"
    manifest_dir = root / "phase1"
    manifest_dir.mkdir(parents=True)
    image_path = root / "stacked/train/ct_001/stent1/001/projections/image/000.tif"
    image_path.parent.mkdir(parents=True)
    image = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    tifffile.imwrite(image_path, image)
    target_path = root / "stent_projections/stent1/001/projections/implant/000.tif"
    if include_target:
        target_path.parent.mkdir(parents=True)
        target = np.zeros(target_shape, dtype=np.float32)
        target[12, 13] = 0.75
        target[10, 11] = 0.375
        tifffile.imwrite(target_path, target)
    manifest = {"cases": [{"case_id": "ct_001_stent1_001", "group_id": "stent1",
                           "metadata": {"stent_variant": "001"},
                           "views": [{"view_id": 0, "angle_deg": 0.0,
                                      "projection_path": "../stacked/train/ct_001/stent1/001/projections/image/000.tif",
                                      "marker_xy": [[8.0, 8.0], [20.0, 20.0]]}]}],
                "folds": {"fold_0": {"train": ["stent1"], "val": ["stent2"], "test": []}}}
    path = manifest_dir / "physical_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_dataset_pairs_original_float_projection_and_keeps_marker_channels(tmp_path):
    samples = load_samples(_paired_fixture(tmp_path), "train")
    assert len(samples) == 1
    inputs, target = ProjectionDataset(samples, patch_size=(32, 32), training=False)[0]
    assert inputs.shape == (4, 32, 32)
    assert target.shape == (1, 32, 32)
    assert float(target[0, 12, 13]) == pytest.approx(0.75)
    assert float(inputs[0].mean()) == pytest.approx(0, abs=1e-6)
    assert float(inputs[0].std(unbiased=False)) == pytest.approx(1, abs=1e-6)
    assert float(inputs[1].max()) > 0.9
    assert float(inputs[2].max()) > 0.9
    assert torch.all(inputs[3] == 1)


def test_dataset_uses_one_training_scale_for_continuous_targets(tmp_path):
    samples = load_samples(_paired_fixture(tmp_path), "train")
    scale = compute_target_scale(samples)
    assert scale == pytest.approx(0.75)
    _, target = ProjectionDataset(samples, patch_size=(32, 32), training=False,
                                  target_scale=scale)[0]
    assert float(target[0, 12, 13]) == pytest.approx(1.0)
    assert float(target.min()) == 0.0
    assert float(target.max()) == 1.0


def test_full_image_mode_keeps_all_pixels_when_padding_for_unet(tmp_path):
    samples = load_samples(_paired_fixture(tmp_path), "train")
    target_path = samples[0].target_path
    target = tifffile.imread(target_path)
    target[0, 0] = 0.25
    target[-1, -1] = 0.5
    tifffile.imwrite(target_path, target)
    inputs, actual = ProjectionDataset(samples, patch_size=(64, 64), training=False,
                                       full_image=True)[0]
    assert inputs.shape == (4, 64, 64)
    assert actual.shape == (1, 64, 64)
    assert float(actual[0, 0, 0]) == pytest.approx(0.25)
    assert float(actual[0, 31, 31]) == pytest.approx(0.5)
    assert torch.all(actual[0, 32:] == 0)


def test_full_image_training_uses_padded_canvas_in_checkpoint(tmp_path):
    source = _paired_fixture(tmp_path)
    samples = load_samples(source, "train")
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    run_training(plan, samples, samples, tmp_path / "full", epochs=1,
                 steps_per_epoch=1, val_steps=1, batch_size=1,
                 device="cpu", full_image=True)
    config = json.loads((tmp_path / "full/run_config.json").read_text())
    assert config["full_image"] is True
    assert config["patch_size"] == [32, 32]


def test_dataset_rejects_missing_target_and_dimension_mismatch(tmp_path):
    with pytest.raises(FileNotFoundError, match="implant"):
        load_samples(_paired_fixture(tmp_path / "missing", include_target=False), "train")
    samples = load_samples(_paired_fixture(tmp_path / "mismatch", target_shape=(31, 32)), "train")
    with pytest.raises(ValueError, match="shape"):
        ProjectionDataset(samples, patch_size=(32, 32), training=False)[0]


def test_load_samples_rejects_overlapping_stent_groups(tmp_path):
    path = _paired_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["folds"]["fold_0"]["val"] = ["stent1"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        load_samples(path, "train")


def test_load_samples_rejects_angle_filename_mismatch(tmp_path):
    path = _paired_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["cases"][0]["views"][0]["angle_deg"] = 10.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="angle"):
        load_samples(path, "train")


def test_weighted_loss_prioritizes_equal_error_on_support():
    target = torch.tensor([[[[1.0, 0.0]]]])
    foreground_error = torch.tensor([[[[0.0, 0.0]]]])
    background_error = torch.tensor([[[[1.0, 1.0]]]])
    assert weighted_smooth_l1(foreground_error, target) > 50 * weighted_smooth_l1(background_error, target)


def test_cascade_uses_predicted_mask_to_gate_image_and_continuous_output():
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    model = build_unet(plan, model_family="mask_then_depth")
    assert model.mask_network.__class__.__module__ == "regression_unet.cascade_model"
    inputs = torch.randn(1, 4, 32, 32)
    details = model(inputs, return_aux=True)
    assert details["mask_logits"].shape == (1, 1, 32, 32)
    assert details["depth"].shape == (1, 1, 32, 32)
    assert torch.allclose(details["masked_image"], inputs[:, :1] * details["mask_probability"])
    assert torch.allclose(details["projection"], details["depth"] * details["mask_probability"])
    assert torch.allclose(model(inputs), details["projection"])
    loss, _ = mask_depth_loss(details, torch.rand(1, 1, 32, 32))
    loss.backward()
    assert model.mask_network.decoder.head.weight.grad is not None
    assert model.depth_network.decoder.head.weight.grad is not None
    changed_band = inputs.clone()
    changed_band[:, 2] += 100
    assert torch.allclose(model(inputs), model(changed_band))


def test_exported_cascade_data_has_two_aligned_labels(tmp_path):
    source = _paired_fixture(tmp_path / "source")
    regression_dir = export_dataset(source, tmp_path / "output/data/regression")
    regression_manifest = json.loads((regression_dir / "manifest.json").read_text())
    regression_manifest["schema_version"] = "phase2_projection_regression_v2"
    regression_manifest["target_kind"] = "normalized_continuous_stent_model_projection"
    (regression_dir / "manifest.json").write_text(json.dumps(regression_manifest))
    cascade_dir = export_mask_depth_dataset(regression_dir, tmp_path / "output/data/cascade")
    manifest = json.loads((cascade_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "phase2_mask_depth_v1"
    sample = manifest["samples"][0]
    assert (cascade_dir / sample["mask"]).is_file()
    continuous = tifffile.imread(cascade_dir / sample["target"])
    binary = tifffile.imread(cascade_dir / sample["mask"])
    np.testing.assert_array_equal(binary, (continuous > 0).astype(np.uint8))
    loaded = load_exported_samples(cascade_dir, "train")
    inputs, target, mask = ProjectionDataset(loaded, (32, 32), training=False,
                                             target_scale=loaded[0].source_scale,
                                             return_mask=True)[0]
    assert inputs.shape == (4, 32, 32)
    np.testing.assert_array_equal(mask.numpy(), (target.numpy() > 0).astype(np.float32))


def test_cascade_warms_up_mask_then_resumes_joint_training(tmp_path):
    source = _paired_fixture(tmp_path / "source")
    sample = load_samples(source, "train")[0]
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, (tifffile.imread(sample.target_path) > 0).astype(np.uint8))
    sample = replace(sample, mask_path=mask_path)
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    options = dict(steps_per_epoch=1, val_steps=1, batch_size=1, patch_size=(32, 32),
                   device="cpu", model_family="mask_then_depth", mask_lr_factor=0.1,
                   mask_warmup_epochs=1)
    run = tmp_path / "run"
    run_training(plan, [sample], [sample], run, epochs=1, **options)
    first = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert first["metrics"]["training_phase"] == "mask_warmup"
    assert first["metrics"]["depth_learning_rate"] == 0
    depth_parameter_ids = first["optimizer"]["param_groups"][1]["params"]
    assert not any(key in first["optimizer"]["state"] for key in depth_parameter_ids)
    run_training(plan, [sample], [sample], run, epochs=2, resume=True, **options)
    second = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert second["metrics"]["training_phase"] == "joint"
    assert second["metrics"]["depth_learning_rate"] > 0
    assert any(key in second["optimizer"]["state"] for key in depth_parameter_ids)
    restored = load_checkpoint_model(run / "checkpoint_latest.pt")
    image = tifffile.imread(sample.image_path)
    maps = predict_maps(restored, image, sample.marker_xy, device="cpu")
    assert set(maps) == {"projection", "mask_probability", "conditional_depth"}
    assert all(value.shape == image.shape for value in maps.values())
    assert all(np.isfinite(value).all() for value in maps.values())


def test_one_epoch_training_writes_reusable_regression_checkpoint(tmp_path):
    manifest_path = _paired_fixture(tmp_path)
    train_samples = load_samples(manifest_path, "train")
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    result = run_training(plan, train_samples, train_samples, tmp_path / "run", epochs=1,
                          steps_per_epoch=1, val_steps=1, batch_size=1,
                          patch_size=(32, 32), device="cpu")
    assert result["epochs_completed"] == 1
    assert (tmp_path / "run" / "checkpoint_latest.pt").is_file()
    checkpoint = torch.load(tmp_path / "run" / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["config"]["patch_size"] == [32, 32]
    assert checkpoint["config"]["target"] == "normalized_implant_projection_including_markers"
    assert checkpoint["config"]["model_family"] == "marker_fusion"
    assert checkpoint["config"]["target_scale"] == pytest.approx(0.75)
    model = load_checkpoint_model(tmp_path / "run" / "checkpoint_best.pt", device="cpu")
    image = tifffile.imread(train_samples[0].image_path)
    result = predict_projection(model, image[:31, :30], [[8.0, 8.0], [20.0, 20.0]], device="cpu")
    assert result.shape == (31, 30)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    assert (result <= 1).all()
    legacy = torch.load(tmp_path / "run" / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    legacy.pop("rng_state")
    for key in ("schedule_epochs", "dataset_manifest_sha256", "real_frames_dir",
                "real_markers_json", "real_markers_sha256", "real_frame_sha256", "real_samples"):
        legacy["config"].pop(key, None)
    torch.save(legacy, tmp_path / "run" / "checkpoint_latest.pt")
    resumed = run_training(plan, train_samples, train_samples, tmp_path / "run", epochs=2,
                           steps_per_epoch=1, val_steps=1, batch_size=1,
                           patch_size=(32, 32), device="cpu", resume=True)
    assert resumed["epochs_completed"] == 2
    checkpoint = torch.load(tmp_path / "run" / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2


def test_resume_records_tensorboard_and_real_frame_every_epoch(tmp_path):
    source = _paired_fixture(tmp_path / "synthetic")
    samples = load_samples(source, "train")
    plan = copy.deepcopy(load_plan(PLANS))
    architecture = plan["configurations"]["2d"]["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 3
    architecture["features_per_stage"] = [8, 16, 32]
    for key in ("kernel_sizes", "strides", "n_conv_per_stage"):
        architecture[key] = architecture[key][:3]
    architecture["n_conv_per_stage_decoder"] = architecture["n_conv_per_stage_decoder"][:2]
    frames = tmp_path / "real"
    frames.mkdir()
    frame = frames / "frame_0001.tif"
    tifffile.imwrite(frame, np.arange(32 * 32, dtype=np.float32).reshape(32, 32))
    markers = tmp_path / "predictions.json"
    markers.write_text(json.dumps({"image_shape": [32, 32], "records": [
        {"source": str(frame.resolve()), "status": "two_candidates",
         "marker_xy": [[8.0, 8.0], [20.0, 20.0]], "edge_candidate": False}]}))
    run = tmp_path / "run"
    options = dict(epochs=2, steps_per_epoch=1, val_steps=1, batch_size=1,
                   patch_size=(32, 32), device="cpu", real_frames_dir=frames,
                   real_markers_json=markers, real_samples=("frame_0001",))
    stopped = run_training(plan, samples, samples, run, stop_requested=lambda: True, **options)
    assert stopped["epochs_completed"] == 1
    assert (run / "real_predictions/epoch_0001/frame_0001.tif").is_file()
    first = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1 and "rng_state" in first
    with pytest.raises(ValueError, match="schedule_epochs"):
        run_training(plan, samples, samples, run, resume=True,
                     **{**options, "schedule_epochs": 999})
    frame_bytes = frame.read_bytes()
    frame.write_bytes(frame_bytes + b"changed")
    with pytest.raises(ValueError, match="real monitor source"):
        run_training(plan, samples, samples, run, resume=True, **options)
    frame.write_bytes(frame_bytes)
    (run / "metrics.jsonl").unlink()
    shutil.rmtree(run / "real_predictions/epoch_0001")
    shutil.rmtree(run / "tensorboard")
    run_training(plan, samples, samples, run, resume=True, **options)
    assert (run / "real_predictions/epoch_0001/summary.json").is_file()
    final = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert final["epoch"] == 2
    prediction = tifffile.imread(run / "real_predictions/epoch_0002/frame_0001.tif")
    assert prediction.shape == (32, 32) and np.isfinite(prediction).all()
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in metrics] == [1, 2]
    events = EventAccumulator(str(run / "tensorboard"))
    events.Reload()
    assert "train/loss" in events.Tags()["scalars"]
    assert [item.step for item in events.Scalars("train/loss")] == [1, 2]
    assert "valid/loss" in events.Tags()["scalars"]
    assert "real/frame_0001/overlay" in events.Tags()["images"]
    shutil.rmtree(run / "real_predictions/epoch_0002")
    run_training(plan, samples, samples, run, resume=True, **options)
    assert (run / "real_predictions/epoch_0002/summary.json").is_file()
    assert [json.loads(line)["epoch"] for line in (run / "metrics.jsonl").read_text().splitlines()] == [1, 2]
    (run / "real_predictions/epoch_0002/.tensorboard_complete").unlink()
    run_training(plan, samples, samples, run, resume=True, **options)
    assert (run / "real_predictions/epoch_0002/.tensorboard_complete").is_file()
