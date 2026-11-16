import numpy as np
from stent_repro.geometry import Grid, circular_view, project_points, backproject_each

def test_isocenter_and_magnification():
    view = circular_view(0, 1100, 600, (101,101), .3)
    uv = project_points(np.array([[0.,0.,0.],[1.,0.,0.]]), view)
    np.testing.assert_allclose(uv[0], [50,50], atol=1e-10)
    np.testing.assert_allclose(abs(uv[1,0]-uv[0,0]), 1100/600/.3, rtol=1e-12)

def test_backprojections_preserve_view_channel():
    grid = Grid((3,3,3), (1,1,1), (-1,-1,-1))
    views = [circular_view(a, 1100, 600, (11,11), .3) for a in (0,60,120)]
    p = np.zeros((3,11,11)); p[1,:,:] = 1
    out = backproject_each(p, views, grid)
    assert out.shape == (3,3,3,3)
    assert np.all(out[0] == 0) and np.all(out[2] == 0)
    assert out[1,1,1,1] == 1
