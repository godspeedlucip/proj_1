"""Project implementations of metric protocols described in Paper A."""

from numbers import Real

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter

from stent_repro.evaluation import validate_grid


def _binary_mask(array, shape, name):
    mask = np.asarray(array)
    if mask.shape != shape or not np.isin(mask, [False, True]).all():
        raise ValueError(f'{name} must be a binary mask matching the image shape')
    return mask.astype(bool, copy=False)


def dte_mape_measured_support(pred, measured_prior_subtracted, measured_tool_support):
    """Compute percent MAPE on measured support after median-filtering both images.

    The caller supplies aligned projection values on the same scale and an
    explicit support mask defined from the measured acquisition. Accept a
    detector image ``[v, u]`` or a stack ``[view, v, u]``. Filter each detector
    image independently with a 3x3 window and reflected edges. Count supported
    pixels with zero filtered measured value and omit them from the mean.
    """
    predicted = np.asarray(pred, dtype=np.float64)
    measured = np.asarray(measured_prior_subtracted, dtype=np.float64)
    if (predicted.shape != measured.shape or measured.ndim not in (2, 3) or
            not measured.size or not np.isfinite(predicted).all() or not np.isfinite(measured).all()):
        raise ValueError('DTE MAPE requires finite, matching detector images or view stacks')
    support = _binary_mask(measured_tool_support, measured.shape, 'measured tool support')
    kernel = (3, 3) if measured.ndim == 2 else (1, 3, 3)
    measured_filtered = median_filter(measured, size=kernel, mode='reflect')[support]
    predicted_filtered = median_filter(predicted, size=kernel, mode='reflect')[support]
    nonzero = measured_filtered != 0
    count = int(np.count_nonzero(nonzero))
    percent = None
    if count:
        target = measured_filtered[nonzero]
        with np.errstate(over='ignore'):
            relative_error = np.abs((predicted_filtered[nonzero] - target) / target)
            percent = float(100 * np.mean(relative_error))
        if not np.isfinite(percent):
            raise ValueError('DTE MAPE exceeds finite floating-point range')
    return {
        'mape_percent': percent,
        'support_pixel_count': int(len(nonzero)),
        'valid_pixel_count': count,
        'zero_denominator_pixel_count': int(len(nonzero) - count),
        'median_filter_size_px': [3, 3],
        'median_filter_boundary': 'reflect',
    }


def project_tolerance_metrics(pred, truth, grid, radius_mm):
    """Calculate project-specific tolerance precision and recall on 3-D masks.

    Precision is prediction overlap with dilated truth divided by prediction
    volume. Recall is prediction overlap with eroded truth divided by eroded
    truth volume. The footprint is a closed Euclidean ball on voxel centres
    with physical spacing. Erosion treats the field exterior as background.
    An empty precision or recall denominator yields ``None``.
    """
    if (not isinstance(radius_mm, Real) or isinstance(radius_mm, bool) or
            not np.isfinite(radius_mm) or radius_mm < 0):
        raise ValueError('radius_mm must be finite and nonnegative')
    validated = validate_grid(grid)
    shape = tuple(validated['shape'])
    predicted = _binary_mask(pred, shape, 'prediction')
    target = _binary_mask(truth, shape, 'truth')
    spacing = validated['spacing_mm'][::-1]  # array axes are z, y, x
    dilated = eroded = target
    if radius_mm > 0 and target.any():
        dilated = distance_transform_edt(~target, sampling=spacing) <= radius_mm
        padded = np.pad(target, 1, constant_values=False)
        interior = distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
        eroded = interior > radius_mm
    pred_n = int(np.count_nonzero(predicted))
    eroded_n = int(np.count_nonzero(eroded))
    precision_hits = int(np.count_nonzero(predicted & dilated))
    recall_hits = int(np.count_nonzero(predicted & eroded))
    return {
        'project_dilated_truth_precision': precision_hits / pred_n if pred_n else None,
        'project_eroded_truth_recall': recall_hits / eroded_n if eroded_n else None,
        'radius_mm': float(radius_mm),
        'pred_voxels': pred_n,
        'truth_voxels': int(np.count_nonzero(target)),
        'dilated_truth_voxels': int(np.count_nonzero(dilated)),
        'eroded_truth_voxels': eroded_n,
        'precision_hits': precision_hits,
        'recall_hits': recall_hits,
    }
