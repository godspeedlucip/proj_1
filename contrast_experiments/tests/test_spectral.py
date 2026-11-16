import math

import numpy as np
import pytest

from stent_repro.spectral import spectral_project


def _material(density, coefficients):
    return {
        'density_g_cm3': density,
        'density_source': 'analytic test fixture density',
        'mass_attenuation_cm2_g': coefficients,
        'attenuation_source': 'analytic test fixture coefficients',
    }


def _project(paths, materials, **overrides):
    inputs = {
        'energy_kev': [20, 60],
        'spectrum_weights': [1, 1],
        'detector_response': [1, 1],
        'spectrum_source': 'analytic test fixture spectrum',
        'detector_response_source': 'analytic test fixture response',
        'materials': materials,
    }
    return spectral_project(paths, **{**inputs, **overrides})


def test_single_active_bin_converts_to_inverse_mm_and_keeps_empty_ray_exact():
    result = _project({'metal': [0, 5]}, {'metal': _material(2.5, [4, 1])},
                      detector_response=[1, 0])
    np.testing.assert_array_equal(result.line_integral, [0.0, 5.0])
    np.testing.assert_allclose(result.transmission, [1.0, math.exp(-5.0)])
    assert result.metadata['materials']['metal']['density_source'] == 'analytic test fixture density'
    assert result.metadata['detector_response_source'] == 'analytic test fixture response'
    assert result.metadata['energy_kev'] == [20.0, 60.0]


def test_material_attenuations_combine_before_weighted_energy_sum():
    result = _project({'metal': [2, 0], 'water': [3, 0]},
                      {'metal': _material(10, [2, 1]), 'water': _material(1, [5, 1])},
                      spectrum_weights=[2, 3], detector_response=[2, 1])
    expected = (4 * math.exp(-5.5) + 3 * math.exp(-2.3)) / 7
    np.testing.assert_allclose(result.transmission, [expected, 1])
    np.testing.assert_allclose(result.line_integral, [-math.log(expected), 0])


def test_extreme_opacity_and_large_weights_keep_line_integral_finite():
    result = _project({'metal': [1000.]}, {'metal': _material(1, [20, 10])},
                      spectrum_weights=[3e300, 1e300], detector_response=[1e300, 1e300])
    np.testing.assert_allclose(result.line_integral, [1000 + math.log(4)])
    np.testing.assert_array_equal(result.transmission, [0])


def test_thin_ray_preserves_tiny_line_integral():
    result = _project({'metal': [1e-18]}, {'metal': _material(10, [4, 2])})
    assert result.line_integral[0] == pytest.approx(3e-18, rel=1e-12, abs=0)


@pytest.mark.parametrize(('overrides', 'message'), [
    ({'spectrum_source': ' '}, 'spectrum_source'),
    ({'detector_response_source': None}, 'detector_response_source'),
    ({'energy_kev': [60, 20]}, 'increasing'),
    ({'spectrum_weights': [1, 0], 'detector_response': [0, 1]}, 'positive'),
    ({'detector_response': [1]}, 'shape'),
])
def test_rejects_invalid_spectrum_or_response(overrides, message):
    with pytest.raises(ValueError, match=message):
        _project({'metal': [1]}, {'metal': _material(1, [1, 1])}, **overrides)


@pytest.mark.parametrize(('paths', 'materials', 'message'), [
    ({'water': [1]}, {'metal': _material(1, [1, 1])}, 'materials'),
    ({'metal': [-1]}, {'metal': _material(1, [1, 1])}, 'path_lengths_mm'),
    ({'metal': [np.nan]}, {'metal': _material(1, [1, 1])}, 'path_lengths_mm'),
    ({'metal': [1]}, {'metal': _material(0, [1, 1])}, 'density_g_cm3'),
    ({'metal': [1]}, {'metal': _material(1, [1, -1])}, 'mass_attenuation_cm2_g'),
    ({'metal': [1]}, {'metal': {**_material(1, [1, 1]), 'attenuation_source': ''}}, 'attenuation_source'),
    ({'metal': [1], 'water': [[1]]}, {'metal': _material(1, [1, 1]), 'water': _material(1, [1, 1])}, 'shape'),
])
def test_rejects_invalid_paths_or_materials(paths, materials, message):
    with pytest.raises(ValueError, match=message):
        _project(paths, materials)
