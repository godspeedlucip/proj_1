import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def geometry_evidence():
    from stent_repro.geometry_checks import geometry_suite

    config = Path(__file__).resolve().parents[1] / "configs/workflows/g0_b.yaml"
    return geometry_suite(config)


def test_analytic_points_preserve_signed_detector_coordinates(geometry_evidence):
    points = geometry_evidence["point_projection"]
    zero_view = points["configured_views"][0]
    assert zero_view["angle_deg"] == 0
    assert points["world_points_xyz_mm"][0] == [1.0, 2.0, 3.0]
    assert zero_view["observed_uv_pixels"][0] == pytest.approx([8.4803921569, 10.4411764706])
    assert points["max_error_pixels"] < 0.1
    assert points["handedness_passed"] is True


def test_analytic_ramps_verify_order_and_fdk_layout(geometry_evidence):
    ordering = geometry_evidence["view_order"]
    assert ordering["angles_deg"] == [0.0, 53.0, 123.0]
    assert ordering["max_analytic_error"] < 1e-4
    assert ordering["permuted_backprojection_max_error"] < 1e-7
    assert ordering["fdk_rejects_mismatched_angles"] is True


def test_rod_probes_compare_side_and_cap_intersections(geometry_evidence):
    upright, tilted = geometry_evidence["rod_probes"]
    assert upright["reference_path_mm"] == pytest.approx(1.7)
    assert tilted["reference_path_mm"] == pytest.approx(1.3750687483)
    assert tilted["axis_xyz"] != [0.0, 0.0, 1.0]
    for probe in (upright, tilted):
        assert probe["center_xyz_mm"][0] != 0
        assert probe["relative_error"] <= 0.05
        assert probe["step_halving_relative_change"] <= 0.01
    assert geometry_evidence["engineering_gate_passed"] is True


def test_fixed_world_resolution_measures_thin_feature_loss(geometry_evidence):
    evidence = geometry_evidence["resolution_convergence"]
    assert evidence["world_bounds_xyz_mm"] == [[-3.0] * 3, [3.0] * 3]
    assert evidence["angles_deg"] == [0.0, 53.0, 123.0]
    ring, strut = evidence["fixtures"]
    coarse, *_, fine = ring["resolutions"]
    assert coarse["status"] == "aliased_away"
    assert coarse["volume_relative_error"] == 1.0
    assert coarse["projection_relative_l1_error"] == 1.0
    assert coarse["step_halving_relative_l1"] is None
    assert fine["volume_relative_error"] < 0.15
    assert fine["projection_relative_l1_error"] < coarse["projection_relative_l1_error"]
    assert strut["reference_volume_mm3"] == pytest.approx(0.1357168026)
    assert strut["resolutions"][-1]["projection_relative_l1_error"] < 0.35
    for fixture in (ring, strut):
        for row in fixture["resolutions"]:
            spacing = row["spacing_mm"]
            assert row["origin_xyz_mm"] == [(-3 + spacing / 2)] * 3
            assert row["shape_zyx"] == [round(6 / spacing)] * 3
    assert evidence["use_for_pass_gate"] is False
    json.dumps(geometry_evidence, allow_nan=False)
