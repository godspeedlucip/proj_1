import torch
from torch import nn

from phase_1.marker_pipeline.epipolar import EpipolarSupport, fundamentals_from_projections
from phase_1.marker_pipeline.models.joint_epipolar import JointEpipolarMarkerNet
from phase_1.tests.test_epipolar_support import cone_matrices


def test_joint_model_has_three_two_peak_heatmaps_and_normalized_weights():
    support = EpipolarSupport(fundamentals_from_projections(cone_matrices(size=(33, 35))), chunk_rows=2)
    model = JointEpipolarMarkerNet(nn.Identity(), support)
    images = torch.zeros(1, 3, 1, 33, 35)
    images[0, 0, 0, 12, 12] = 8
    images[0, 0, 0, 20, 21] = 8
    initial, refined = model(images)
    assert initial.shape == refined.shape == (1, 3, 1, 33, 35)
    assert initial[0, 0, 0, 12, 12] == initial[0, 0, 0, 20, 21]
    assert torch.isfinite(refined).all()
    weights = model.last_view_weights
    assert weights.shape == (1, 3)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-6)


def test_target_refined_heatmap_depends_on_other_views_with_gradient():
    support = EpipolarSupport(fundamentals_from_projections(cone_matrices(size=(33, 35))), chunk_rows=2)
    model = JointEpipolarMarkerNet(nn.Identity(), support)
    with torch.no_grad():
        model.refine[-1].weight.fill_(0.1)
    images = torch.zeros(1, 3, 1, 33, 35, requires_grad=True)
    initial_a, refined_a = model(images)
    changed = images.detach().clone()
    changed[0, 1, 0, 16, 17] = 8
    initial_b, refined_b = model(changed)
    assert torch.equal(initial_a[:, 0], initial_b[:, 0])
    assert not torch.allclose(refined_a[:, 0], refined_b[:, 0])
    refined_a[:, 0].sum().backward()
    assert images.grad is not None
    assert images.grad[:, 1].abs().sum() > 0
