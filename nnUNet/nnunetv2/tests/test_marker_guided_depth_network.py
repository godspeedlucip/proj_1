"""CPU checks for joint training, padding-aware supervision and segmentation compatibility."""
from types import SimpleNamespace

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_network import (
    build_marker_guided_network,
)
from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.network import (
    build_joint_network, joint_loss,
)


def small_configuration():
    return SimpleNamespace(
        network_arch_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
        network_arch_init_kwargs_req_import=[],
        network_arch_init_kwargs=dict(
            n_stages=3, features_per_stage=[8, 16, 32],
            conv_op=nn.Conv2d, kernel_sizes=[[3, 3]] * 3,
            strides=[[1, 1], [2, 2], [2, 2]], n_conv_per_stage=[1] * 3,
            n_conv_per_stage_decoder=[1] * 2, conv_bias=True,
            norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
            nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        ),
    )


def test_wraps_native_backbone_preserving_logits_and_segmentation_entry_point():
    torch.manual_seed(420)
    config = small_configuration()
    source = build_marker_guided_network(config, 4, 2, True).eval()
    net = build_joint_network(config, 4, 2, True).eval()
    net.backbone.load_state_dict(source.state_dict(), strict=True)
    assert {name.split(".")[0] for name in net.state_dict()} == {"backbone", "depth_refiner"}
    x = torch.randn(1, 4, 32, 48)
    x[:, 1:] = 0
    with torch.no_grad():
        native = source(x)
        ds = net(x, return_depth=True)["seg_logits_ds"]
        assert len(ds) == len(native) == 2
        for actual, expected in zip(ds, native):
            torch.testing.assert_close(actual, expected)
        net.decoder.deep_supervision = False
        single = net(x, return_depth=True)
        assert len(single["seg_logits_ds"]) == 1
        torch.testing.assert_close(net(x), native[0])
        torch.testing.assert_close(single["seg_logits"], native[0])


def test_depth_gradient_reaches_backbone_only_through_probability_inside_mask():
    torch.manual_seed(421)
    net = build_joint_network(small_configuration(), 4, 2, False)
    inputs = torch.randn(1, 4, 32, 48)
    outputs = net(inputs, return_depth=True)
    h, p, d = (outputs[k] for k in ("predicted_mask", "mask_probability", "depth_normalized"))
    assert 0 < h.sum() < h.numel() and not h.requires_grad
    assert d[h == 0].count_nonzero() == 0
    assert torch.equal(d[h == 1], outputs["conditional_depth"][h == 1])
    p.retain_grad()
    _, terms = joint_loss(outputs, torch.ones_like(h), torch.full_like(h, 2), torch.ones_like(h))
    terms["depth"].backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad[h == 0].count_nonzero() == 0 and p.grad[h == 1].count_nonzero() > 0
    seg_weight = net.backbone.decoder.seg_layers[-1].weight
    assert seg_weight.grad.norm() > 0
    refiner_grad = next(net.depth_refiner.parameters()).grad.clone()
    assert refiner_grad.norm() > 0

    # Break just the differentiable P*H input. X and H are independent of backbone weights.
    with net.depth_refiner.register_forward_pre_hook(lambda module, args: args[0].detach()):
        net.zero_grad(set_to_none=True)
        disconnected = net(inputs, return_depth=True)
        _, terms = joint_loss(disconnected, torch.ones_like(h), torch.full_like(h, 2), torch.ones_like(h))
        terms["depth"].backward()
    assert seg_weight.grad is None
    torch.testing.assert_close(next(net.depth_refiner.parameters()).grad, refiner_grad)


def test_tied_logits_make_empty_hard_mask_and_missed_targets_still_train_segmentation():
    net = build_joint_network(small_configuration(), 4, 2, False)
    head = net.backbone.decoder.seg_layers[-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    out = net(torch.randn(1, 4, 32, 48), return_depth=True)
    p = out["mask_probability"]
    assert torch.all(p == 0.5)
    assert out["predicted_mask"].count_nonzero() == out["depth_normalized"].count_nonzero() == 0
    total, terms = joint_loss(out, torch.ones_like(p), torch.full_like(p, 2), torch.ones_like(p))
    torch.testing.assert_close(terms["fg"], torch.tensor(1.975))
    assert terms["bg"] == 0 and terms["empty_bg"] == 1 and terms["empty_fg"] == 0
    total.backward()
    assert head.bias.grad.norm() > 0
    assert all(param.grad is None or param.grad.count_nonzero() == 0 for param in net.depth_refiner.parameters())


def test_masked_image_does_not_leak_outside_pixels_with_fixed_probability():
    torch.manual_seed(422)
    net = build_joint_network(small_configuration(), 4, 2, False).eval()
    inputs = torch.randn(1, 4, 32, 48)
    with torch.no_grad():
        first = net(inputs, return_depth=True)
        outside = first["predicted_mask"] == 0
        assert outside.any() and (~outside).any()
        inputs[:, :1][outside] += 10
        with net.backbone.register_forward_hook(lambda module, args, result: first["seg_logits"]):
            changed = net(inputs, return_depth=True)
        torch.testing.assert_close(changed["conditional_depth"], first["conditional_depth"], rtol=0, atol=0)


def test_regression_means_over_valid_regions_of_each_image_including_misses():
    logits = torch.zeros(2, 2, 1, 4)
    mask = torch.tensor([[[[1, 0, 0, 0]]], [[[1, 1, 1, 0]]]])
    valid = torch.tensor([[[[1, 1, 1, 0]]], [[[1, 1, 1, 0]]]])
    prediction = torch.tensor([[[[0., 2., 4., 999.]]], [[[1., 1., 1., 999.]]]])
    target = torch.tensor([[[[2., 0., 0., 0.]]], [[[1., 1., 1., 0.]]]])
    total, terms = joint_loss({"seg_logits_ds": [logits], "depth_normalized": prediction}, mask, target, valid)
    torch.testing.assert_close(terms["fg"], torch.tensor(1.975 / 2))
    torch.testing.assert_close(terms["bg"], torch.tensor(1.5))
    torch.testing.assert_close(terms["depth"], torch.tensor(1.975 / 2 + 0.15))
    torch.testing.assert_close(total, terms["seg"] + terms["depth"])
    assert terms["empty_bg"] == 1 and terms["empty_fg"] == 0


def test_first_joint_update_changes_both_branches_with_no_weight_decay():
    torch.manual_seed(424)
    net = build_joint_network(small_configuration(), 4, 2, True)
    optimizer = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": 1e-5},
        {"params": net.depth_refiner.parameters(), "lr": 1e-4},
    ], weight_decay=0.0)
    seg_weight = net.backbone.decoder.seg_layers[-1].weight
    depth_weight = net.depth_refiner.head.weight
    original_seg, original_depth = seg_weight.detach().clone(), depth_weight.detach().clone()
    outputs = net(torch.randn(1, 4, 32, 48), return_depth=True)
    mask = torch.ones_like(outputs["predicted_mask"])
    assert outputs["predicted_mask"].sum() > 0
    loss, _ = joint_loss(outputs, mask, mask * 2, mask)
    loss.backward()
    optimizer.step()
    assert not torch.equal(original_seg, seg_weight)
    assert not torch.equal(original_depth, depth_weight)


def test_empty_target_regions_and_fully_padded_scales_stay_finite():
    net = build_joint_network(small_configuration(), 4, 2, False)
    with torch.no_grad():
        net.depth_refiner.head.bias.fill_(3.0)
        net.depth_refiner.head.weight.zero_()
        seg_head = net.backbone.decoder.seg_layers[-1]
        seg_head.weight.zero_()
        seg_head.bias.copy_(torch.tensor([0., 1.]))
    outputs = net(torch.zeros(1, 4, 32, 48), return_depth=True)
    assert (outputs["depth_normalized"] > 1).all()
    mask = torch.zeros_like(outputs["predicted_mask"])
    total, terms = joint_loss(outputs, mask, mask, torch.ones_like(mask))
    assert total.isfinite() and terms["fg"] == 0 and terms["empty_fg"] == 1 and terms["bg"] > 1
    total, terms = joint_loss(outputs, mask, mask, torch.zeros_like(mask))
    assert total == 0 and terms["seg"] == 0 and terms["empty_fg"] == terms["empty_bg"] == 1


def test_deep_supervision_excludes_partial_padding_blocks_and_skips_lowest_level():
    logits = [torch.zeros(1, 2, side, side, requires_grad=True) for side in (8, 4, 2)]
    target = torch.ones(1, 1, 8, 8)
    valid = torch.zeros_like(target)
    valid[:, :, :5, :5] = 1
    _, parts = joint_loss({"seg_logits_ds": logits, "depth_normalized": target}, [target], target, valid)
    parts["seg"].backward()
    assert torch.all(logits[0].grad[:, :, :5, :5] != 0)
    assert logits[0].grad[:, :, 5:, :].count_nonzero() == logits[0].grad[:, :, :, 5:].count_nonzero() == 0
    assert torch.all(logits[1].grad[:, :, :2, :2] != 0)
    assert logits[1].grad[:, :, 2:, :].count_nonzero() == logits[1].grad[:, :, :, 2:].count_nonzero() == 0
    assert logits[2].grad is None


def test_strict_reload_of_first_full_frame_smoke_encoder_names():
    source = build_joint_network(small_configuration(), 4, 2, False)
    smoke = {}
    for name, value in source.state_dict().items():
        if name.startswith("depth_refiner.input_block."):
            name = name.replace(".input_block.", ".encoder.0.", 1)
        elif name.startswith("depth_refiner.encoder."):
            _, _, stage, layer = name.split(".", 3)
            name = f"depth_refiner.encoder.{int(stage) + 1}.{layer}"
        smoke[name] = value
    restored = build_joint_network(small_configuration(), 4, 2, False)
    restored.load_state_dict(smoke, strict=True)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
