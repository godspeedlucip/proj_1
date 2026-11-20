import json
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from nnunetv2.tests.test_marker_guided_depth_network import small_configuration
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.fusion import MarkerFusion
from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.data import (
    FullFrameDataset, axial_distance_map, prepare_image,
)
from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.inference import (
    frame_metrics, predict_frame, restore_network,
)
from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.network import (
    build_joint_network, initialize_from_legacy, joint_loss,
)
from nnunetv2.utilities.checkpoints import save_checkpoint_file
from phase2_joint import configure_fusion


def test_axial_geometry_and_invalid_points():
    shape = (8, 11)
    points = [[2, 3], [6, 3]]
    fusion = MarkerFusion("encoder-axial", 2)
    axial = axial_distance_map(points, shape, 2)
    expected = [1, .5, 0, 0, 0, 0, 0, .5, 1, 1, 1]
    np.testing.assert_equal(axial, np.tile(expected, (shape[0], 1)))
    diagonal = [[1, 5], [7, 2]]
    np.testing.assert_equal(axial_distance_map(diagonal, shape, 2), axial_distance_map(diagonal[::-1], shape, 2))
    image = np.arange(np.prod(shape)).reshape(shape)
    for invalid in (None, [[1, 2], [1, 2]], [[-1, 0], [4, 4]], [[float("nan"), 2], [0, 0]]):
        channels = prepare_image(image, invalid, True, fusion=fusion)
        assert channels.shape == (5, *shape) and channels[0].any() and not channels[1:].any()


def test_loader_flips_all_five_planes_and_padding_together(tmp_path):
    image = np.arange(77, dtype=np.float32).reshape(7, 11)
    row = dict(image="image.tif", mask="mask.tif", depth="depth.tif", marker_xy=[[1, 2], [6, 4]], sample_id="s")
    for key, array in (("image", image), ("mask", image > 40), ("depth", image)):
        tifffile.imwrite(tmp_path / row[key], array)
    fusion = MarkerFusion("encoder-axial")
    data = FullFrameDataset(tmp_path, [row], 2, True, [(True, False)], fusion)[0]
    expected = prepare_image(image, row["marker_xy"], True, fusion=fusion)
    np.testing.assert_equal(data["data"][:, :7, :11].numpy(), expected[:, ::-1, :])
    assert not data["data"][:, 7:].any() and not data["data"][:, :, 11:].any()
    assert data["valid_pixels"].sum() == image.size


@pytest.mark.parametrize("mode", ["encoder", "encoder-axial"])
def test_cascade_uses_first_adapter_output_in_second_stage_and_bottleneck(mode):
    fusion = MarkerFusion(mode)
    net = build_joint_network(small_configuration(), fusion.channels, 2, True, fusion)
    backbone = net.backbone
    seen = {}
    def capture(name):
        def hook(module, args, out):
            seen[name] = out
        return hook
    def capture_input(module, args):
        seen["next_in"] = args[0]
    def capture_decoder(module, args):
        seen["bottleneck_in"] = args[0][-1]
    x = torch.randn(1, fusion.channels, 32, 48)
    x[:, 3] = 1
    with (backbone.adapters["0"].register_forward_hook(capture("first")),
          backbone.encoder.stages[1].register_forward_pre_hook(capture_input),
          backbone.adapters["2"].register_forward_hook(capture("last")),
          backbone.decoder.register_forward_pre_hook(capture_decoder)):
        outputs = net(x, return_depth=True)
    assert seen["next_in"] is seen["first"] and seen["bottleneck_in"] is seen["last"]
    assert len(outputs["seg_logits_ds"]) == 2
    h = outputs["predicted_mask"]
    loss, _ = joint_loss(outputs, torch.ones_like(h), torch.ones_like(h), torch.ones_like(h))
    loss.backward()
    assert backbone.adapters["2"].attention[-1].weight.grad.abs().sum() > 0
    if fusion.axial:
        # The axial path starts neutral. After the identity-initialized adapter opens,
        # the existing segmentation loss must be able to train the axial projection.
        assert backbone.axial_projections["2"].weight.count_nonzero() == 0
        with torch.no_grad():
            backbone.adapters["2"].attention[-1].weight.normal_(0, .1)
        net.zero_grad()
        net(x)[0][:, 1].square().mean().backward()
        assert backbone.axial_projections["2"].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("joint", [False, True])
def test_warm_start_and_gate_off_against_legacy(joint):
    config = small_configuration()
    legacy = build_joint_network(config, 4, 2, False).eval()
    fusion = MarkerFusion("encoder-axial")
    net = build_joint_network(config, 5, 2, False, fusion).eval()
    weights = (legacy if joint else legacy.backbone).state_dict()
    initialize_from_legacy(net, weights, joint=joint)
    x = torch.randn(1, 5, 32, 48)
    x[:, 3] = 0
    with torch.no_grad():
        torch.testing.assert_close(net(x), legacy(x[:, :4]), rtol=0, atol=0)
        if joint:
            torch.testing.assert_close(net(x, return_depth=True)["depth_normalized"],
                                       legacy(x[:, :4], return_depth=True)["depth_normalized"], rtol=0, atol=0)
    missing = dict(weights)
    missing.pop(next(key for key in missing if "adapters.0." in key))
    with pytest.raises(RuntimeError, match="Missing key"):
        initialize_from_legacy(net, missing, joint=joint)


def tiny_plans_architecture():
    config = small_configuration()
    init = config.network_arch_init_kwargs.copy()
    imports = ["conv_op", "norm_op", "nonlin"]
    for key in imports:
        init[key] = "torch.nn." + init[key].__name__
    return dict(network_class_name=config.network_arch_class_name, arch_kwargs=init, _kw_requires_import=imports)


@pytest.mark.parametrize(("mode", "scale"), [("skip", 32), ("encoder", 32), ("encoder-axial", 16), ("encoder-axial", 32)])
def test_plan_and_checkpoint_restore_use_matching_tiff_inference_channels(tmp_path, mode, scale):
    fusion = MarkerFusion(mode, scale)
    base = "DepthCascadePlans"
    root = tmp_path / "Dataset505_Test"
    folder = root / "preprocessed" / root.name
    folder.mkdir(parents=True)
    plans = dict(dataset_name=root.name, plans_name=base, configurations={"2d": {"architecture": tiny_plans_architecture()}},
                 joint_depth={"target_scale": 2.0})
    (folder / (base + ".json")).write_text(json.dumps(plans))
    if mode != "skip":
        configure_fusion(Namespace(dataset_root=root, base_plans=base, fusion=mode,
                                   axial_scale_px=[float(scale)] if fusion.axial else None,
                                   init_joint=None))
        variant, = folder.glob(base + "Encoder*.json")
        plans = json.loads(variant.read_text())
        assert plans["joint_depth"]["marker_fusion"] == asdict(fusion)
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    net = build_joint_network(small_configuration(), fusion.channels, 2, False, fusion).eval()
    checkpoint = tmp_path / "checkpoint_final.safetensors"
    save_checkpoint_file({"network_weights": net.state_dict(), "joint_depth": plans["joint_depth"]}, str(checkpoint))
    device = torch.device("cpu")
    restored, _ = restore_network(tmp_path, checkpoint, device)
    image = np.arange(15 * 19, dtype=np.float32).reshape(15, 19)
    args = (image, [[3, 2], [13, 9]], True, 2., device)
    expected = predict_frame(net, *args)
    actual = predict_frame(restored, *args)
    for name in actual:
        np.testing.assert_equal(actual[name], expected[name])
    assert actual["masks"].shape == image.shape and not actual["depth_raw"][actual["masks"] == 0].any()
    if fusion.axial:
        plans["joint_depth"]["marker_fusion"]["axial_scale_px"] = 32 if scale == 16 else 16
        (tmp_path / "plans.json").write_text(json.dumps(plans))
        with pytest.raises(ValueError, match="disagree on marker fusion"):
            restore_network(tmp_path, checkpoint, device)


def test_cli_generates_paired_scales_and_preserves_existing_runs(tmp_path):
    root = tmp_path / "Dataset505_Test"
    folder = root / "preprocessed" / root.name
    folder.mkdir(parents=True)
    base = "DepthCascadePlans"
    source = dict(dataset_name=root.name, plans_name=base, configurations={"2d": {"batch_size": 1}},
                  joint_depth={"seed": 7, "init_seg_checkpoint": "same-initial-weights"})
    (folder / (base + ".json")).write_text(json.dumps(source))
    script = Path(__file__).resolve().parents[2] / "phase2_joint.py"
    command = [sys.executable, str(script), "configure-fusion", "--dataset-root", str(root),
               "--fusion", "encoder-axial"]
    subprocess.run(command, check=True)
    legacy_path = folder / (base + "EncoderAxial.json")
    original = legacy_path.read_bytes()
    for _ in range(2):
        subprocess.run(command + ["--axial-scale-px", "16", "32"], check=True)
    assert legacy_path.read_bytes() == original
    image = np.ones((8, 40), dtype=np.float32)
    inputs = {}
    for scale in (16, 32):
        name = f"{base}EncoderAxial{scale}px"
        plan = json.loads((folder / (name + ".json")).read_text())
        assert plan.pop("plans_name") == name
        fusion = MarkerFusion(**plan["joint_depth"].pop("marker_fusion"))
        assert fusion == MarkerFusion("encoder-axial", scale)
        assert plan == {k: v for k, v in source.items() if k != "plans_name"}
        inputs[scale] = prepare_image(image, [[2, 3], [12, 3]], True, fusion=fusion)
    np.testing.assert_array_equal(inputs[16][:4], inputs[32][:4])
    assert inputs[16][4, 3, 20] == .5 and inputs[32][4, 3, 20] == .25


def test_end_metrics_distinguish_clipped_struts_from_false_positives():
    mask = np.array([[1, 0, 1, 1, 1, 0, 1]], dtype=bool)
    depth = np.array([[0, 1, 1, 1, 1, 0, 0]], dtype=np.float32)
    scores = frame_metrics({"masks": mask, "depth_raw": mask.astype(float)}, depth, [[1, 0], [4, 0]])
    assert scores["axial_outside_fp_pixels"] == 2
    assert scores["first_end_recall"] == 0 and scores["second_end_recall"] == 1
    assert scores["foreground_mae_raw"] == .25


def test_new_trainer_warm_start_one_update_and_resume(tmp_path, monkeypatch):
    from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.nnUNetTrainerMarkerGuidedDepth import (
        nnUNetTrainerMarkerGuidedDepthSmoke,
    )
    for name in ("nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.setenv(name, str(tmp_path))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    arch = tiny_plans_architecture()
    legacy = build_joint_network(small_configuration(), 4, 2, True)
    initial = tmp_path / "legacy.safetensors"
    save_checkpoint_file({"network_weights": legacy.state_dict()}, str(initial))
    fusion = MarkerFusion("encoder-axial")
    plans = dict(dataset_name="Dataset991_Tiny", plans_name="TinyAxial",
                 configurations={"2d": dict(data_identifier="2d", batch_size=1, patch_size=[32, 48], architecture=arch)},
                 joint_depth=dict(manifest=str(manifest), seed=11, target_scale=2.0,
                                  init_joint_checkpoint=str(initial), marker_fusion=asdict(fusion)))
    dataset = {"labels": {"background": 0, "stent": 1}, "channel_names": {str(i): str(i) for i in range(4)}}
    def trainer(resume):
        return nnUNetTrainerMarkerGuidedDepthSmoke(dict(plans, continue_training=resume), "2d", 0, dataset,
                                                   torch.device("cpu"))
    model = trainer(False)
    model.on_train_start()
    try:
        torch.testing.assert_close(model.network.depth_refiner.head.weight, legacy.depth_refiner.head.weight)
        assert model.num_input_channels == 5
        x = torch.randn(1, 5, 32, 48)
        x[:, 3] = 1
        target = torch.ones_like(x[:, :1])
        model.num_iterations_per_epoch = 1
        result = model.train_step(dict(data=x, target=target, depth_target=target, valid_pixels=target.bool(), keys=["s"]))
        assert np.isfinite(result["loss"]) and model.optimizer_updates == 1
        checkpoint = tmp_path / "checkpoint_latest.safetensors"
        model.save_checkpoint(checkpoint)
        restored = trainer(True)
        restored.load_checkpoint(checkpoint)
        assert restored.optimizer_updates == 1 and restored.current_epoch == 1
        torch.testing.assert_close(restored.network.backbone.adapters["2"].attention[-1].weight,
                                   model.network.backbone.adapters["2"].attention[-1].weight)
    finally:
        model.tb.close()
