"""Independent scalar NumPy sampling reference for the projection tests."""

import math
import numpy as np


def _sample(volume, d, h, w):
    depth, height, width = volume.shape
    if not (0 <= d < depth):
        return 0.0
    total = 0.0
    for ih in (math.floor(h), math.floor(h) + 1):
        for iw in (math.floor(w), math.floor(w) + 1):
            if 0 <= ih < height and 0 <= iw < width:
                total += (1 - abs(h - ih)) * (1 - abs(w - iw)) * volume[d, ih, iw]
    return total


def reference_projection(value, valid, geometry):
    depth, height, width = geometry.input_shape
    side = geometry.canvas_shape[0]
    sd, sh, sw = geometry.spacing
    detector_spacing = geometry.detector_spacing
    result = np.zeros((len(geometry.angles), depth, side), dtype=np.float64)
    support = np.zeros_like(result, dtype=bool)
    for view, angle in enumerate(geometry.angles):
        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)
        for d in range(depth):
            for detector in range(side):
                t = (detector - (side - 1) / 2) * detector_spacing
                numerator = denominator = 0.0
                for ray in range(side):
                    r = (ray - (side - 1) / 2) * detector_spacing
                    h = (c * t + s * r + geometry.center_mm[0]) / sh
                    w = (-s * t + c * r + geometry.center_mm[1]) / sw
                    numerator += _sample(value * valid, d, h, w)
                    denominator += _sample(valid, d, h, w)
                if denominator > 1e-12:
                    result[view, d, detector] = numerator / denominator
                    support[view, d, detector] = True
    return result, support
