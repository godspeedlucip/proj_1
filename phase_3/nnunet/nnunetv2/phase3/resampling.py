"""Lossless resampling contract for already aligned FDK volumes."""

from __future__ import annotations

import numpy as np
import torch


def identity_resample(data: np.ndarray | torch.Tensor | None, new_shape,
                      current_spacing, new_spacing, **kwargs):
    """Keep array identity only when the original detector grid is unchanged."""
    if data is None:
        return None
    if not isinstance(data, (np.ndarray, torch.Tensor)) or data.ndim != 4:
        raise ValueError("FDK resampling expects a 4D channel-first array or tensor")
    shape = tuple(int(value) for value in new_shape)
    if tuple(data.shape[-3:]) != shape:
        raise ValueError(f"FDK grid shape change rejected: {tuple(data.shape[-3:])} -> {shape}")
    before = np.asarray(current_spacing, dtype=np.float64)
    after = np.asarray(new_spacing, dtype=np.float64)
    if (before.shape != (3,) or after.shape != (3,)
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.any(np.abs(before - after) > 1e-4)):
        raise ValueError(f"FDK spacing change rejected: {before} -> {after}")
    return data
