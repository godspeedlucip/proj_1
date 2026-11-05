import numpy as np
import pytest
import torch

from phase_1.marker_pipeline.epipolar import (
    EpipolarSupport,
    fundamentals_from_projections,
    validate_relative_geometry,
)


def cone_matrices(start=0, size=(65, 67)):
    h, w = size
    focal = 90.0
    intrinsic = np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]])
    matrices = []
    for angle in (start, start + 30, start + 60):
        a = np.deg2rad(angle)
        center = np.array([100 * np.cos(a), 100 * np.sin(a), 0.0])
        right = np.array([-np.sin(a), np.cos(a), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        forward = np.array([-np.cos(a), -np.sin(a), 0.0])
        rotation = np.stack((right, up, forward))
        matrices.append(intrinsic @ np.column_stack((rotation, -rotation @ center)))
    return np.stack(matrices)


def project(P, X):
    q = P @ np.r_[X, 1.0]
    return q[:2] / q[2]


def test_cone_projection_points_obey_pairwise_epipolar_geometry():
    Ps = cone_matrices()
    Fs = fundamentals_from_projections(Ps)
    for X in (np.array([0.0, 0.0, 0.0]), np.array([5.0, -4.0, 3.0])):
        points = [project(P, X) for P in Ps]
        for target in range(3):
            for source in range(3):
                if target != source:
                    line = Fs[target, source] @ np.r_[points[target], 1.0]
                    assert abs(np.dot(line, np.r_[points[source], 1.0])) < 1e-5


def test_fixed_relative_geometry_accepts_rotation_and_rejects_detector_shift():
    reference = fundamentals_from_projections(cone_matrices(0))
    validate_relative_geometry(reference, cone_matrices(30), (65, 67))
    changed = cone_matrices(30)
    changed[1, 0, 3] += 100.0
    with pytest.raises(ValueError, match="geometry"):
        validate_relative_geometry(reference, changed, (65, 67))


def test_support_preserves_shape_gradient_and_zero_background():
    size = (65, 67)
    support = EpipolarSupport(fundamentals_from_projections(cone_matrices(size=size)), chunk_rows=3)
    images = torch.zeros(1, 3, 1, *size, requires_grad=True)
    maps = support(images)
    assert maps.shape == (1, 3, 2, *size)
    assert torch.allclose(maps, torch.zeros_like(maps), atol=1e-6)
    with torch.no_grad():
        images[0, 1, 0, 32, 33] = 1.0
    maps = support(images)
    assert torch.isfinite(maps).all()
    assert maps[0, 0].max() > 0
    maps.sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()


def test_partial_pool_cells_use_actual_detector_centers():
    centers=EpipolarSupport._pool_centers(65,4,device="cpu",dtype=torch.float32)
    assert len(centers)==17
    assert torch.isclose(centers[-1],torch.tensor(64.0))
    assert torch.isclose(centers[-2],torch.tensor(61.5))
    mapped=EpipolarSupport._full_to_pool(centers,centers)
    assert torch.allclose(mapped,torch.arange(17,dtype=torch.float32))


def test_upsampling_does_not_bleed_support_outside_source_detector():
    geometry=np.zeros((3,3,3,3),dtype=float)
    geometry[0,1]=np.array([[0.,0.,1.],[0.,0.,0.],[-2.,0.,0.]])
    support=EpipolarSupport(geometry,stride=4,samples=32)
    probabilities=torch.zeros(1,3,1,32,32)
    probabilities[:,1]=1
    result=support(probabilities)
    assert result[0,0,0,16,15] > 0
    assert torch.isclose(result[0,0,0,16,0],torch.tensor(1.0),atol=1e-5)
    assert result[0,0,0,16,16] == 0
    assert result[0,0,0,16,17] == 0


def test_partial_target_cell_is_upsampled_at_its_actual_center():
    geometry=np.zeros((3,3,3,3),dtype=float)
    geometry[0,1]=np.array([[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]])
    support=EpipolarSupport(geometry,stride=4,samples=32)
    probabilities=torch.zeros(1,3,1,65,65)
    probabilities[:,1,:,:,64]=1
    result=support(probabilities)
    assert result[0,0,0,32,61] < 0.1
    assert result[0,0,0,32,64] > 0.9
