"""Physical detector grid for depth-preserving parallel projections."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProjectionGeometry:
    input_shape: tuple[int, int, int]
    canvas_shape: tuple[int, int]
    spacing: tuple[float, float, float]
    angles: tuple[float, ...]
    center_mm: tuple[float, float]
    detector_spacing: float


def make_projection_geometry(shape: tuple[int, int, int], angles: tuple[float, ...],
                             spacing=(.2, .2, .2)) -> ProjectionGeometry:
    if len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape):
        raise ValueError("Projection shape must have three edges of at least two voxels")
    if len(spacing) != 3 or any(not math.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError("Projection spacing must be positive and finite")
    if not angles or any(not math.isfinite(angle) for angle in angles):
        raise ValueError("Projection angles must be finite and nonempty")
    spacing = tuple(float(s) for s in spacing)
    angles = tuple(float(angle) for angle in angles)
    _, height, width = shape
    detector_spacing = min(spacing[1:])
    side = math.ceil(math.hypot(height * spacing[1], width * spacing[2]) / detector_spacing)
    if height % 2 == 0 and width % 2 == 0 and side % 2:
        side += 1
    center = ((height - 1) * spacing[1] / 2, (width - 1) * spacing[2] / 2)
    return ProjectionGeometry(tuple(shape), (side, side), spacing, angles, center, detector_spacing)
