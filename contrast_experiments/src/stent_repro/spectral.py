"""Energy-bin Beer–Lambert projection for caller-supplied spectral inputs."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp


@dataclass(frozen=True)
class SpectralProjection:
    line_integral: np.ndarray
    transmission: np.ndarray
    metadata: dict


def _source(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} requires a nonempty source citation or identifier')
    return value.strip()


def _nonnegative(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'uif' or not array.size:
        raise ValueError(f'{name} must contain numeric values')
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array) & (array >= 0)):
        raise ValueError(f'{name} must be finite and nonnegative')
    return array


def _energy_vector(value, name, shape):
    vector = _nonnegative(value, name)
    if vector.shape != shape:
        raise ValueError(f'{name} shape must match energy_kev shape')
    return vector


def spectral_project(
    path_lengths_mm: Mapping[str, ArrayLike],
    *,
    energy_kev: ArrayLike,
    spectrum_weights: ArrayLike,
    detector_response: ArrayLike,
    spectrum_source: str,
    detector_response_source: str,
    materials: Mapping[str, Mapping],
) -> SpectralProjection:
    """Integrate primary transmission then take the negative natural logarithm.

    Rays for all materials share the same shape. Spectrum weights represent
    incident photons per bin, including any bin widths. Response is relative
    signal per incident photon. Each material's mass attenuation coefficients
    are already sampled at ``energy_kev`` in the same order. A source identifier
    accompanies each input for traceability. Its contents are caller supplied.
    """
    energy = _nonnegative(energy_kev, 'energy_kev')
    if energy.ndim != 1 or np.any(energy <= 0) or np.any(np.diff(energy) <= 0):
        raise ValueError('energy_kev must be a vector of positive, strictly increasing energies')
    weights = _energy_vector(spectrum_weights, 'spectrum_weights', energy.shape)
    response = _energy_vector(detector_response, 'detector_response', energy.shape)
    active = np.flatnonzero((weights > 0) & (response > 0))
    if not active.size:
        raise ValueError('the spectrum and detector response must overlap in a positive weight bin')

    metadata = {
        'model': 'polychromatic_beer_lambert',
        'path_length_units': 'mm',
        'projection_units': 'dimensionless_line_integral',
        'energy_kev': energy.tolist(),
        'spectrum_weights': weights.tolist(),
        'detector_response': response.tolist(),
        'spectrum_source': _source(spectrum_source, 'spectrum_source'),
        'detector_response_source': _source(detector_response_source, 'detector_response_source'),
        'materials': {},
    }
    if (not isinstance(path_lengths_mm, Mapping) or not isinstance(materials, Mapping)
            or not path_lengths_mm or path_lengths_mm.keys() != materials.keys()):
        raise ValueError('path_lengths_mm and materials must name the same nonempty set of materials')
    if any(not isinstance(name, str) or not name.strip() for name in materials):
        raise ValueError('materials require nonempty string names')

    fields = {'density_g_cm3', 'density_source', 'mass_attenuation_cm2_g', 'attenuation_source'}
    sources = metadata['materials']
    inputs = []
    shape = None
    for name in sorted(materials):
        path = _nonnegative(path_lengths_mm[name], f'path_lengths_mm[{name}]')
        if shape is not None and path.shape != shape:
            raise ValueError('all material path arrays must share the same shape')
        shape = path.shape
        spec = materials[name]
        if not isinstance(spec, Mapping) or spec.keys() != fields:
            raise ValueError(f'materials[{name}] requires exactly {sorted(fields)}')
        density = _nonnegative(spec['density_g_cm3'], f'{name}.density_g_cm3')
        if density.ndim != 0 or density <= 0:
            raise ValueError(f'{name}.density_g_cm3 must be a positive scalar')
        mass = _energy_vector(spec['mass_attenuation_cm2_g'],
                              f'{name}.mass_attenuation_cm2_g', energy.shape)
        with np.errstate(over='ignore'):
            attenuation = mass * (float(density) / 10)
        if not np.all(np.isfinite(attenuation)):
            raise ValueError(f'{name} linear attenuation exceeds float64 range')
        sources[name] = {
            'density_g_cm3': float(density),
            'density_source': _source(spec['density_source'], f'{name}.density_source'),
            'mass_attenuation_cm2_g': mass.tolist(),
            'attenuation_source': _source(spec['attenuation_source'], f'{name}.attenuation_source'),
            'linear_attenuation_per_mm': attenuation.tolist(),
        }
        inputs.append((path, attenuation))

    # Log weights avoid overflow or joint underflow in spectrum × response.
    log_weights = np.log(weights[active]) + np.log(response[active])
    log_weights -= log_weights.max()
    log_weights -= logsumexp(log_weights)
    log_transmission = np.full(shape, -np.inf)
    lost_transmission = np.zeros(shape)
    optical_depth = np.empty(shape)
    scratch = np.empty(shape)
    # Stream the energy axis to keep peak memory proportional to the ray count.
    with np.errstate(over='ignore', under='ignore'):
        for index, log_weight in zip(active, log_weights):
            optical_depth.fill(0)
            for path, attenuation in inputs:
                np.multiply(path, attenuation[index], out=scratch)
                optical_depth += scratch
            np.subtract(log_weight, optical_depth, out=scratch)
            np.logaddexp(log_transmission, scratch, out=log_transmission)
            np.negative(optical_depth, out=scratch)
            np.expm1(scratch, out=scratch)
            lost_transmission -= np.exp(log_weight) * scratch

        # Log-domain sums handle opacity. expm1/log1p preserve weak losses.
        line_integral = np.where(lost_transmission <= 0.5,
                                 -np.log1p(-np.minimum(lost_transmission, 0.5)),
                                 -log_transmission)
        if not np.all(np.isfinite(line_integral)):
            raise ValueError('projected optical depth exceeds float64 range')
        transmission = np.asarray(np.exp(-line_integral))
    return SpectralProjection(line_integral, transmission, metadata)
