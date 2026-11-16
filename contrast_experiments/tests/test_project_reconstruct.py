import numpy as np
import pytest
from stent_repro.geometry import Grid, circular_view, forward_project, fdk


def phantom():
    grid = Grid((17,21,21), (1.,1.,1.), (-10.,-10.,-8.))
    z,y,x=np.meshgrid(np.arange(-8,9),np.arange(-10,11),np.arange(-10,11),indexing='ij')
    return ((x*x+y*y)<=25).astype(np.float32),grid

def views(angles):
    return [circular_view(a,200,100,(25,49),.8) for a in angles]

def test_forward_project_cylinder_has_correct_line_integral():
    volume,grid=phantom()
    projection=forward_project(volume,grid,views([0]),step_mm=.25)
    assert projection.shape == (1,25,49)
    assert projection[0,12,24] == pytest.approx(11.,abs=.35)
    assert projection[0,12,0] == 0

def test_fdk_full_circle_localizes_cylinder_and_physical_scale():
    volume,grid=phantom(); angles=np.arange(0,360,10).tolist(); v=views(angles)
    p=forward_project(volume,grid,v,step_mm=.5)
    result=fdk(p,v,grid,angles,angular_mode='periodic')
    assert result.shape==volume.shape
    assert result[8,10,10] == pytest.approx(1.0,abs=.25)
    assert result[8,10,10] > result[8,10,19] + .5

def test_fdk_reorders_projection_with_views_and_checks_angles():
    volume,grid=phantom(); angles=[0,45,90,135]; v=views(angles)
    p=forward_project(volume,grid,v,step_mm=1)
    a=fdk(p,v,grid,angles,angular_mode='limited')
    order=[2,0,3,1]
    b=fdk(p[order],[v[i] for i in order],grid,[angles[i] for i in order],angular_mode='limited')
    np.testing.assert_allclose(a,b,atol=1e-6)
    with pytest.raises(ValueError,match='angle'):
        fdk(p,v,grid,[10,45,90,135],angular_mode='limited')

def test_point_projection_direction_follows_geometry_and_view():
    from stent_repro.geometry import project_points
    v=views([0,90])
    point=np.array([[2.,0.,0.]])
    a=project_points(point,v[0])[0]; b=project_points(point,v[1])[0]
    assert a[0] > (v[0].detector_shape[1]-1)/2
    assert b[0] == pytest.approx((v[1].detector_shape[1]-1)/2)

def test_fdk_rejects_wrong_detector_direction():
    from dataclasses import replace
    _,grid=phantom(); v=views([0,45,90,135]); bad=list(v)
    bad[0]=replace(v[0],u_axis=-v[0].u_axis)
    with pytest.raises(ValueError,match='geometry|orientation'):
        fdk(np.zeros((4,25,49)),bad,grid,[0,45,90,135])

def test_fdk_recovers_off_axis_object_position():
    grid=Grid((9,21,21),(1,1,1),(-10,-10,-4))
    z,y,x=np.meshgrid(np.arange(-4,5),np.arange(-10,11),np.arange(-10,11),indexing='ij')
    volume=(((x-3)**2+(y+2)**2)<=4).astype(np.float32)
    angles=list(range(0,360,10)); v=views(angles)
    projections=forward_project(volume,grid,v,step_mm=.5)
    output=fdk(projections,v,grid,angles,angular_mode='periodic')
    plane=output[4]
    yy,xx=np.unravel_index(np.argmax(plane),plane.shape)
    assert abs((xx-10)-3)<=2 and abs((yy-10)+2)<=2
