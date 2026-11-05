"""Canonical ordering for fixed-spacing projection triplets."""

import math
from typing import Sequence


def _angular_error(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def canonical_triplet_indices(
    angles_deg: Sequence[float], *, tolerance_deg: float = 1e-3
) -> tuple[int, int, int]:
    """Return source indices in circular ``n*30, (n+1)*30, (n+2)*30`` order."""
    if len(angles_deg) != 3 or not math.isfinite(tolerance_deg) or tolerance_deg < 0:
        raise ValueError("joint detection requires three finite angles and a valid tolerance")
    angles = [float(angle) % 360.0 for angle in angles_deg]
    if not all(math.isfinite(angle) for angle in angles):
        raise ValueError("joint detection requires finite view angles")

    sorted_indices = sorted(range(3), key=lambda index: angles[index])
    sorted_angles = [angles[index] for index in sorted_indices]
    gaps = [
        (sorted_angles[(index + 1) % 3] - sorted_angles[index]) % 360.0
        for index in range(3)
    ]
    largest_gap = max(range(3), key=lambda index: gaps[index])
    order = tuple(sorted_indices[(largest_gap + 1 + offset) % 3] for offset in range(3))
    start = angles[order[0]]
    if _angular_error(start, round(start / 30.0) * 30.0) > tolerance_deg or any(
        _angular_error(angles[order[offset]], start + offset * 30.0) > tolerance_deg
        for offset in range(3)
    ):
        raise ValueError("joint detection requires consecutive 30 degree views")
    return order
