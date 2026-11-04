from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_network import MarkerGuidedPlainConvUNet
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.residual_prior_adapter import ResidualPriorAdapter


def test_adapter_initial_identity_beta_and_zero_gate():
    adapter = ResidualPriorAdapter(8, 4)
    assert torch.allclose(adapter.beta, torch.tensor(0.05), atol=1e-7)
    image, prior = torch.randn(2, 8, 16, 12), torch.randn(2, 4, 16, 12)
    gate = torch.ones(2, 1, 16, 12)
    assert torch.equal(adapter(image, prior, gate), image)
    with torch.no_grad():
        adapter.attention[-1].weight.normal_()
    zero_gate = torch.zeros_like(gate)
    assert torch.equal(adapter(image, prior, zero_gate), image)
    full = adapter(image, prior, gate)
    half = adapter(image, prior, gate * 0.5)
    assert not torch.equal(full, image)
    assert torch.allclose(half - image, (full - image) * 0.5, atol=1e-6)


def test_adapter_backward_starts_at_attention_and_is_finite():
    adapter = ResidualPriorAdapter(8, 4)
    image = torch.randn(1, 8, 16, 16, requires_grad=True)
    prior = torch.randn(1, 4, 16, 16, requires_grad=True)
    output = adapter(image, prior, torch.ones(1, 1, 16, 16))
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert adapter.attention[-1].weight.grad is not None
    assert torch.isfinite(adapter.attention[-1].weight.grad).all()


def _small_network(n_stages=3, patch=(32, 48)):
    return MarkerGuidedPlainConvUNet(
        input_channels=4, num_classes=3, deep_supervision=True,
        n_stages=n_stages, features_per_stage=[8 * (2 ** i) for i in range(n_stages)],
        conv_op=nn.Conv2d, kernel_sizes=[[3, 3]] * n_stages,
        strides=[[1, 1]] + [[2, 2]] * (n_stages - 1),
        n_conv_per_stage=[1] * n_stages, n_conv_per_stage_decoder=[1] * (n_stages - 1),
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True}, dropout_op=None,
        dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True})


def test_plan_driven_skip_sizes_and_deep_supervision_order():
    model = _small_network(n_stages=3)
    assert model.fusion_stage_indices == [0, 1]
    outputs = model(torch.randn(2, 4, 32, 48))
    assert isinstance(outputs, list) and len(outputs) == 2
    assert outputs[0].shape == (2, 3, 32, 48)
    assert outputs[1].shape == (2, 3, 16, 24)
    smaller = _small_network(n_stages=2)
    assert smaller.fusion_stage_indices == [0, 1]
    logits = smaller(torch.randn(1, 4, 32, 32))
    assert all(torch.isfinite(out).all() for out in logits)


def test_zero_priors_produce_finite_logits_and_all_prior_branches_are_in_graph():
    model = _small_network(n_stages=3)
    x = torch.randn(1, 4, 32, 32)
    x[:, 1:] = 0
    out = model(x)
    sum(t.square().mean() for t in out).backward()
    assert all(torch.isfinite(t).all() for t in out)
    assert model.prior_encoder.stem[0].weight.grad is not None
