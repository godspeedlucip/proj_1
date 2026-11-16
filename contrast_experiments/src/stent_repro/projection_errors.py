"""Spatially correlated surrogates for tool-only detector projections."""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq
from scipy.special import ndtri, owens_t


@dataclass(frozen=True)
class CorrelatedProjectionErrors:
    projection: np.ndarray
    missing_mask: np.ndarray
    false_positive_mask: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class ProjectionResidualPair:
    split: str
    clean: ArrayLike
    observed: ArrayLike


@dataclass(frozen=True)
class CorrelatedProjectionCalibration:
    parameters: dict[str, float]
    metadata: dict


def _scalar(value, name, upper=np.inf):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value <= upper):
        raise ValueError(f'{name} must be a finite number between 0 and {upper}')
    return float(value)


def _projection(value, name):
    array = np.asarray(value)
    if array.ndim not in (2, 3) or not array.size or array.dtype.kind not in 'uif':
        raise ValueError(f'{name} must be a nonempty numeric [H, W] or [views, H, W] array')
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array) & (array >= 0)):
        raise ValueError(f'{name} must contain finite, nonnegative tool line integrals')
    return array


def _kernel(sigma):
    if sigma < 0.125:
        return np.ones(1)
    radius = int(4 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1) / sigma
    kernel = np.exp(-0.5 * offsets**2)
    return kernel / kernel.sum()


def _threshold_field(shape, rate, sigma, rng):
    if rate in (0.0, 1.0):
        return np.full(shape, bool(rate), dtype=bool)
    kernel = _kernel(sigma)
    radius = len(kernel) // 2
    if not radius:
        return rng.standard_normal(shape) < ndtri(rate)

    # Generate independent halo pixels so the cropped field is stationary
    # at detector edges. Each view has a separate field and halo.
    h, w = shape[-2:]
    threshold = ndtri(rate) * np.dot(kernel, kernel)
    mask = np.empty(shape, dtype=bool)
    for view in mask.reshape(-1, h, w):
        noise = rng.standard_normal((h + 2 * radius, w + 2 * radius))
        field = gaussian_filter(noise, sigma, radius=radius, mode='constant')
        view[:] = field[radius:radius + h, radius:radius + w] < threshold
    return mask


def apply_correlated_projection_errors(
    clean: ArrayLike,
    *,
    seed: int,
    missing_rate: float,
    false_positive_rate: float,
    correlation_sigma_px: float,
    false_positive_amplitude: float,
    support: ArrayLike | None = None,
) -> CorrelatedProjectionErrors:
    """Apply separate thresholded Gaussian fields within and outside support.

    The rates are marginal probabilities on eligible pixels in every view.
    Missing pixels are set to zero, and false positives add a fixed line
    integral. Zero sigma is the independent limit of this model.
    """
    clean = _projection(clean, 'clean')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    missing_rate = _scalar(missing_rate, 'missing_rate', 1)
    false_positive_rate = _scalar(false_positive_rate, 'false_positive_rate', 1)
    sigma = _scalar(correlation_sigma_px, 'correlation_sigma_px')
    amplitude = _scalar(false_positive_amplitude, 'false_positive_amplitude')
    if false_positive_rate and amplitude == 0:
        raise ValueError('false_positive_amplitude must be positive for a positive false_positive_rate')
    support_source = 'positive_clean_line_integral' if support is None else 'explicit_boolean_mask'
    support = clean > 0 if support is None else np.asarray(support)
    if support.shape != clean.shape or support.dtype != bool:
        raise ValueError('support must be a boolean mask with the same shape as clean')

    rng = np.random.Generator(np.random.PCG64(seed))
    missing = _threshold_field(clean.shape, missing_rate, sigma, rng) & support
    inserted = _threshold_field(clean.shape, false_positive_rate, sigma, rng) & ~support
    projection = clean.copy()
    projection[missing] = 0
    with np.errstate(over='ignore'):
        projection[inserted] += amplitude
    if not np.isfinite(projection[inserted]).all():
        raise ValueError('false positive signal exceeds float64 range')
    return CorrelatedProjectionErrors(projection, missing, inserted, {
        'model': 'correlated_gaussian_field',
        'seed': int(seed),
        'rng': 'PCG64',
        'support_source': support_source,
        'missing_rate': missing_rate,
        'false_positive_rate': false_positive_rate,
        'correlation_sigma_px': sigma,
        'false_positive_amplitude': amplitude,
        'missing_pixels': int(missing.sum()),
        'false_positive_pixels': int(inserted.sum()),
    })


def _neighbor_counts(eligible, error):
    """Count valid adjacent pairs, event endpoints, and joint events."""
    counts = np.zeros(3, dtype=np.int64)
    for left, right in ((np.s_[..., :-1, :], np.s_[..., 1:, :]),
                        (np.s_[..., :, :-1], np.s_[..., :, 1:])):
        valid = eligible[left] & eligible[right]
        a = error[left] & valid
        b = error[right] & valid
        counts += (valid.sum(), a.sum() + b.sum(), (a & b).sum())
    return counts


def _latent_correlations(neighbors):
    """Estimate latent Gaussian lag-one correlations from binary joint rates."""
    estimates = []
    for count, endpoints, joint_count in neighbors:
        if not count or endpoints in (0, 2 * count):
            continue
        rate = endpoints / (2 * count)
        joint = joint_count / count
        if joint <= rate**2:
            correlation = 0.0
        elif joint >= rate:
            correlation = 1.0
        else:
            z = ndtri(rate)

            def residual(rho):
                # Equal-threshold bivariate Gaussian CDF via Owen's T.
                return rate - 2 * owens_t(z, np.sqrt((1 - rho) / (1 + rho))) - joint

            correlation = brentq(residual, 0, 1)
        estimates.append((correlation, int(count)))
    return estimates


def _kernel_lag_one(sigma):
    kernel = _kernel(sigma)
    return np.dot(kernel[:-1], kernel[1:]) / np.dot(kernel, kernel)


def calibrate_correlated_projection_errors(
    pairs: Iterable[ProjectionResidualPair], *, support_threshold: float = 0.0,
) -> CorrelatedProjectionCalibration:
    """Fit pixel-pooled error rates, median false-positive residual and sigma.

    Both arrays use tool-only line integrals, with values above the same
    threshold defining their support. Estimate sigma from pooled same-class
    horizontal and vertical lag-one pair counts, within each view. Each pair
    must have a train or validation label supplied by the caller.
    """
    threshold = _scalar(support_threshold, 'support_threshold')
    split_counts = {'train': 0, 'validation': 0}
    eligible = np.zeros(2, dtype=np.int64)
    errors = np.zeros(2, dtype=np.int64)
    neighbors = np.zeros((2, 3), dtype=np.int64)
    positive_residuals = []
    min_detector_side = np.inf
    for pair in pairs:
        if not isinstance(pair, ProjectionResidualPair) or pair.split not in split_counts:
            raise ValueError('calibration requires residual pairs from train or validation')
        clean = _projection(pair.clean, 'clean')
        observed = _projection(pair.observed, 'observed')
        if clean.shape != observed.shape:
            raise ValueError('clean and observed must have the same shape')
        split_counts[pair.split] += 1
        min_detector_side = min(min_detector_side, *clean.shape[-2:])
        support = clean > threshold
        measured_support = observed > threshold
        missing = support & ~measured_support
        inserted = ~support & measured_support
        for idx, (population, error) in enumerate(((support, missing), (~support, inserted))):
            eligible[idx] += population.sum()
            errors[idx] += error.sum()
            neighbors[idx] += _neighbor_counts(population, error)
        if inserted.any():
            positive_residuals.append(observed[inserted] - clean[inserted])

    if not sum(split_counts.values()):
        raise ValueError('calibration requires at least one train or validation residual pair')
    estimates = _latent_correlations(neighbors)
    max_sigma = max(1.0, min_detector_side / 4)
    sigma = 0.0
    at_resolution_limit = False
    if estimates:
        correlations, counts = zip(*estimates)
        rho = np.average(correlations, weights=counts)
        if rho >= _kernel_lag_one(max_sigma):
            sigma = max_sigma
            at_resolution_limit = True
        elif rho > 0:
            sigma = brentq(lambda s: _kernel_lag_one(s) - rho, 0.0, max_sigma)

    rates = np.divide(errors, eligible, out=np.zeros(2), where=eligible > 0)
    amplitude = float(np.median(np.concatenate(positive_residuals))) if positive_residuals else 0.0
    return CorrelatedProjectionCalibration({
        'missing_rate': float(rates[0]),
        'false_positive_rate': float(rates[1]),
        'false_positive_amplitude': amplitude,
        'correlation_sigma_px': float(sigma),
    }, {
        'model': 'correlated_gaussian_field',
        'support_threshold': threshold,
        'pair_counts_by_split': split_counts,
        'eligible_pixels': dict(zip(('missing', 'false_positive'), map(int, eligible))),
        'error_pixels': dict(zip(('missing', 'false_positive'), map(int, errors))),
        'correlation_estimable': bool(estimates),
        'correlation_sigma_at_resolution_limit': at_resolution_limit,
        'correlation_sigma_max_px': max_sigma,
        'eligible_adjacent_pairs': dict(zip(('missing', 'false_positive'), map(int, neighbors[:, 0]))),
    })
