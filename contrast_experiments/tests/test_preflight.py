import pytest
from stent_repro.preflight import preflight,geometry_validation


def test_cpu_readiness_does_not_certify_paper(workflow_config):
    r=preflight(workflow_config(),stage='dte',device='cpu',measure=False)
    assert r['device_available'] is True
    assert r['full_step_measured'] is False and r['paper_numerical_reproduction_ready'] is False
    assert r['estimated_parameter_optimizer_bytes']>0
    assert r['limitations']


def test_measure_executes_real_full_shape_step(workflow_config):
    r=preflight(workflow_config('B'),stage='dtr',device='cpu',measure=True)
    assert r['full_step_measured'] is True and r['optimizer_state_tensors']>0
    assert r['gradient_finite'] is True and r['measured_input_shape']==[2,3,16,16,16]
    assert r['formal_task_ready'] is False


def test_analytic_geometry_checks_are_bounded(workflow_config):
    r=geometry_validation(workflow_config())
    assert r['point_error_pixels']<.1
    assert r['central_ray_relative_error']<.05
    assert r['step_halving_change']<.01
    assert r['scope']=='analytic_cpu_reference_not_system_calibration'
