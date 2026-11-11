import numpy as np
import pytest
import torch

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.projection import ProjectionConsistencyLoss, project_masked
from nnunetv2.phase3.projection_geometry import make_projection_geometry
from .fixtures import make_config
from .projection_reference import reference_projection


@pytest.mark.parametrize("shape", [(3, 4, 6), (5, 5, 7), (4, 6, 4)])
def test_projection_matches_independent_scalar_reference(shape):
    rng = np.random.default_rng(19)
    values = rng.random(shape)
    valid = (rng.random(shape) > .2).astype(np.float64)
    geometry = make_projection_geometry(shape, (0., 30., 90.))
    actual, mask = project_masked(torch.tensor(values)[None, None],
                                  torch.tensor(valid)[None, None], geometry)
    expected, expected_mask = reference_projection(values, valid, geometry)
    np.testing.assert_allclose(actual.detach().numpy()[0, :, 0], expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(mask.numpy()[0, :, 0], expected_mask)
    assert geometry.canvas_shape[0] >= np.ceil(np.hypot(shape[1], shape[2]))


def test_axis_aligned_rays_keep_depth_and_rotate_in_plane():
    geometry = make_projection_geometry((7, 4, 6), (0., 90.))
    valid = torch.ones((1, 1, 7, 4, 6), dtype=torch.float64)
    a = torch.zeros_like(valid)
    a[0, 0, 1, 1, 4] = 1
    b = torch.zeros_like(valid)
    b[0, 0, 5, 1, 4] = 1
    pa, _ = project_masked(a, valid, geometry)
    pb, _ = project_masked(b, valid, geometry)
    assert not torch.allclose(pa, pb)
    assert pa[0, 0, 0, 1].sum() > 0 and pa[0, 0, 0, 5].sum() == 0
    assert not torch.allclose(pa[0, 0], pa[0, 1])


def test_ignored_logits_never_penalized_and_gradcheck():
    config = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                        cnn__batch_size=1, training__max_optimizer_steps=4,
                                        training__updates_per_epoch=2))
    loss = ProjectionConsistencyLoss(config)
    target = torch.zeros((1, 1, 3, 4, 5), dtype=torch.int64)
    target[0, 0, 1, 1, 1] = 1
    target[0, 0, 0, 2, 2] = 2
    logits = torch.randn((1, 2, 3, 4, 5), requires_grad=True)
    changed = logits.detach().clone()
    changed[:, :, 0, 2, 2] += 1000
    assert torch.allclose(loss(logits, target), loss(changed, target), atol=1e-7)
    loss(logits, target).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert len(loss.last_valid_ray_counts.shape) == 2
    with pytest.raises(ValueError, match="valid"):
        loss(logits, torch.full_like(target, 2))
    geometry = make_projection_geometry((2, 3, 4), (0., 35.))
    value = torch.rand((1, 1, 2, 3, 4), dtype=torch.float64, requires_grad=True)
    valid = torch.ones_like(value)
    assert torch.autograd.gradcheck(lambda x: project_masked(x, valid, geometry)[0],
                                    (value,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_projection_loss_averages_each_case_and_view_equally():
    config = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                        cnn__batch_size=1, training__max_optimizer_steps=4,
                                        training__updates_per_epoch=2))
    logits = torch.zeros((2, 2, 4, 5, 6))
    logits[0, 1] = 2.
    logits[1, 1] = -2.
    target = torch.zeros((2, 1, 4, 5, 6), dtype=torch.long)
    target[1] = 2
    target[1, :, :2, :2, :2] = 0
    geometry = make_projection_geometry((4, 5, 6), tuple(config.projection.angles))
    valid = target != 2
    projected, rays = project_masked(torch.softmax(logits, 1)[:, 1:2], valid, geometry)
    assert rays[0].sum() != rays[1].sum()
    expected = (projected.square() * rays).sum(dim=(2, 3, 4)) / rays.sum(dim=(2, 3, 4))
    assert torch.allclose(ProjectionConsistencyLoss(config)(logits, target), expected.mean(), atol=1e-7)


def test_projection_support_uses_configured_epsilon():
    geometry = make_projection_geometry((2, 3, 4), (0.,))
    value = torch.ones((1, 1, 2, 3, 4), dtype=torch.float64)
    tiny_support = torch.full_like(value, 1e-8)
    _, rays = project_masked(value, tiny_support, geometry)
    assert not rays.any()
