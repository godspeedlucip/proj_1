import json

import numpy as np
import pytest

from stent_repro import paper_metrics


def grid(shape, spacing=(0.1, 0.2, 0.3)):
    return {'shape': list(shape), 'spacing_mm': list(spacing), 'origin_mm': [0, 0, 0],
            'direction': np.eye(3).tolist()}


def test_mape_filters_both_images_then_uses_only_measured_support():
    predicted = np.full((5, 5), 6.0)
    measured = np.full((5, 5), 4.0)
    support = np.zeros((5, 5), dtype=bool)
    predicted[2, 2] = 100.0
    measured[2, 2] = 99.0
    support[2, 2] = True
    measured[0, 0] = 7.0

    result = paper_metrics.dte_mape_measured_support(predicted, measured, support)

    assert result['mape_percent'] == 50.0
    assert result['support_pixel_count'] == result['valid_pixel_count'] == 1
    assert result['zero_denominator_pixel_count'] == 0


def test_mape_reflects_detector_edges_and_filters_views_independently():
    measured = np.array([[[6.0, 6.0], [0.0, 0.0]], [[20.0, 20.0], [20.0, 20.0]]])
    predicted = np.full_like(measured, 2.0)
    support = np.zeros_like(measured, dtype=bool)
    support[:, 0, 0] = True

    result = paper_metrics.dte_mape_measured_support(predicted, measured, support)

    assert result['valid_pixel_count'] == 2
    assert result['mape_percent'] == pytest.approx((2 / 3 + 0.9) * 50)


def test_mape_audits_zeros_created_by_filtering_and_uses_absolute_denominator():
    measured = np.zeros((2, 3, 3))
    measured[0, 1, 1] = 5.0
    measured[1].fill(-4.0)
    predicted = np.zeros_like(measured)
    support = np.zeros_like(measured, dtype=bool)
    support[:, 1, 1] = True

    result = paper_metrics.dte_mape_measured_support(predicted, measured, support)

    assert result['support_pixel_count'] == 2
    assert result['valid_pixel_count'] == result['zero_denominator_pixel_count'] == 1
    assert result['mape_percent'] == 100.0


@pytest.mark.parametrize('support', [np.zeros((3, 3), bool), np.ones((3, 3), bool)])
def test_mape_undefined_without_a_valid_denominator(support):
    zeros = np.zeros((3, 3))
    result = paper_metrics.dte_mape_measured_support(zeros, zeros, support)

    assert result['mape_percent'] is None
    assert result['valid_pixel_count'] == 0
    assert result['zero_denominator_pixel_count'] == int(support.sum())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('support', [None, np.ones((2, 2)), np.full((3, 3), 0.8)])
def test_mape_rejects_missing_or_invalid_support(support):
    ones = np.ones((3, 3))
    with pytest.raises(ValueError, match='support'):
        paper_metrics.dte_mape_measured_support(ones, ones, support)


def test_tolerance_respects_anisotropic_xyz_spacing_and_inclusive_radius():
    truth = np.zeros((5, 5, 5), dtype=bool)
    truth[2, 2, 2] = True
    predicted = np.zeros_like(truth)
    predicted[2, 2, 3] = predicted[2, 3, 2] = predicted[3, 2, 2] = True

    result = paper_metrics.project_tolerance_metrics(predicted, truth, grid(truth.shape), 0.1)

    assert result['project_dilated_truth_precision'] == pytest.approx(1 / 3)
    assert result['project_eroded_truth_recall'] is None
    assert result['dilated_truth_voxels'] == 3
    assert result['eroded_truth_voxels'] == 0


def test_tolerance_erodes_against_exterior_background_and_zero_radius_is_exact():
    truth = np.ones((3, 3, 3), dtype=bool)
    predicted = np.zeros_like(truth)
    predicted[1, 1, 1] = True
    g = grid(truth.shape, (0.1, 0.1, 0.1))

    single_voxel_radius = paper_metrics.project_tolerance_metrics(predicted, truth, g, 0.1)
    zero_radius = paper_metrics.project_tolerance_metrics(predicted, truth, g, 0)

    assert single_voxel_radius['project_eroded_truth_recall'] == 1.0
    assert single_voxel_radius['eroded_truth_voxels'] == 1
    assert zero_radius['project_eroded_truth_recall'] == pytest.approx(1 / 27)
    assert zero_radius['project_dilated_truth_precision'] == 1.0


def test_empty_truth_does_not_create_phantom_dilation():
    truth = np.zeros((2, 2, 2), dtype=bool)
    predicted = np.ones_like(truth)
    result = paper_metrics.project_tolerance_metrics(predicted, truth, grid(truth.shape), 2.0)

    assert result['project_dilated_truth_precision'] == 0.0
    assert result['project_eroded_truth_recall'] is None
    assert result['dilated_truth_voxels'] == 0
