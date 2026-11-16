import numpy as np
import pytest

from stent_repro.projection_errors import (
    ProjectionResidualPair,
    apply_correlated_projection_errors,
    calibrate_correlated_projection_errors,
)


def _apply(clean, **overrides):
    settings = {
        'seed': 241,
        'missing_rate': 0.30,
        'false_positive_rate': 0.25,
        'correlation_sigma_px': 2.5,
        'false_positive_amplitude': 0.7,
    }
    return apply_correlated_projection_errors(clean, **{**settings, **overrides})


def _clean(shape):
    image = np.zeros(shape, dtype=np.float32)
    image[..., :shape[-1] // 2] = 2
    return image


def test_replays_masks_and_changes_only_their_intended_support():
    clean = _clean((2, 64, 64))
    original = clean.copy()
    first = _apply(clean)
    replay = _apply(clean)
    other = _apply(clean, seed=242)
    np.testing.assert_array_equal(first.projection, replay.projection)
    np.testing.assert_array_equal(first.missing_mask, replay.missing_mask)
    np.testing.assert_array_equal(first.false_positive_mask, replay.false_positive_mask)
    np.testing.assert_array_equal(clean, original)
    assert first.metadata['seed'] == 241
    assert first.metadata['model'] == 'correlated_gaussian_field'
    assert not np.array_equal(first.missing_mask, other.missing_mask)
    assert not np.array_equal(first.missing_mask[0], first.missing_mask[1])
    assert clean[first.missing_mask].min() == 2
    assert clean[first.false_positive_mask].max() == 0
    np.testing.assert_array_equal(first.projection[first.missing_mask], 0)
    np.testing.assert_allclose(first.projection[first.false_positive_mask], 0.7)
    unchanged = ~(first.missing_mask | first.false_positive_mask)
    np.testing.assert_array_equal(first.projection[unchanged], clean[unchanged])


def test_error_clusters_have_more_adjacent_pixels_than_independent_limit():
    clean = _clean((384, 384))
    correlated = _apply(clean)
    independent = _apply(clean, correlation_sigma_px=0)
    for name in ('missing_mask', 'false_positive_mask'):
        linked = getattr(correlated, name)
        independent_mask = getattr(independent, name)
        linked_adjacent = (linked[:-1] & linked[1:]).sum() / linked[:-1].sum()
        independent_adjacent = ((independent_mask[:-1] & independent_mask[1:]).sum()
                                / independent_mask[:-1].sum())
        assert linked_adjacent > independent_adjacent + 0.35


def test_explicit_support_and_rate_endpoints():
    clean = np.array([[2, 2], [2, 2]], dtype=float)
    support = np.array([[True, False], [False, True]])
    result = _apply(clean, support=support, missing_rate=1, false_positive_rate=1)
    np.testing.assert_array_equal(result.missing_mask, support)
    np.testing.assert_array_equal(result.false_positive_mask, ~support)
    np.testing.assert_allclose(result.projection, [[0., 2.7], [2.7, 0.]])
    identity = _apply(clean, missing_rate=0, false_positive_rate=0)
    np.testing.assert_array_equal(identity.projection, clean)


def test_calibration_recovers_the_simulated_error_parameters():
    clean = _clean((2, 384, 384))
    train = _apply(clean)
    validation = _apply(clean, seed=9846)
    pairs = [ProjectionResidualPair('train', clean, train.projection),
             ProjectionResidualPair('validation', clean, validation.projection)]
    fitted = calibrate_correlated_projection_errors(iter(pairs))
    assert fitted.parameters['missing_rate'] == pytest.approx(0.30, abs=0.025)
    assert fitted.parameters['false_positive_rate'] == pytest.approx(0.25, abs=0.025)
    assert fitted.parameters['false_positive_amplitude'] == pytest.approx(0.7)
    assert fitted.parameters['correlation_sigma_px'] == pytest.approx(2.5, abs=0.45)
    assert fitted.metadata['pair_counts_by_split'] == {'train': 1, 'validation': 1}


def test_calibration_uses_support_threshold_and_positive_residual_amplitude():
    clean = np.array([[0.1, 0.1], [2., 2.]])
    observed = np.array([[0.4, 0.1], [0.1, 2.]])
    fit = calibrate_correlated_projection_errors(
        [ProjectionResidualPair('train', clean, observed)], support_threshold=0.2)
    assert fit.parameters['missing_rate'] == 0.5
    assert fit.parameters['false_positive_rate'] == 0.5
    assert fit.parameters['false_positive_amplitude'] == pytest.approx(0.3)


@pytest.mark.parametrize('split', ['test', 'holdout', 'TRAIN'])
def test_calibration_rejects_excluded_splits_before_reading_their_arrays(split):
    valid = ProjectionResidualPair('train', _clean((2, 2)), _clean((2, 2)))
    with pytest.raises(ValueError, match='train or validation'):
        calibrate_correlated_projection_errors([valid, ProjectionResidualPair(split, None, None)])


def test_calibration_identifies_unobserved_and_unestimable_parameters():
    image = np.zeros((3, 3))
    fit = calibrate_correlated_projection_errors([ProjectionResidualPair('train', image, image)])
    assert fit.parameters == {
        'missing_rate': 0.0, 'false_positive_rate': 0.0,
        'false_positive_amplitude': 0.0, 'correlation_sigma_px': 0.0,
    }
    assert fit.metadata['eligible_pixels']['missing'] == 0
    assert fit.metadata['correlation_estimable'] is False


@pytest.mark.parametrize('overrides', [
    {'missing_rate': -1}, {'false_positive_rate': np.nan}, {'correlation_sigma_px': -1},
    {'false_positive_amplitude': 0}, {'seed': -5}, {'support': np.zeros((1, 1), bool)},
])
def test_correlated_projection_errors_reject_invalid_settings(overrides):
    with pytest.raises(ValueError):
        _apply(_clean((2, 2)), **overrides)
