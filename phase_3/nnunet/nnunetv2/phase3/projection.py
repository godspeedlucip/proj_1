"""Differentiable, masked, direction-correct FDK projection regularization."""

from __future__ import annotations

from collections import OrderedDict
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .projection_geometry import ProjectionGeometry, make_projection_geometry


_GRID_CACHE: OrderedDict[tuple, Tensor] = OrderedDict()
_MAX_CACHED_GRIDS = 2


def _sampling_grid(geometry: ProjectionGeometry, angle: float, device, dtype) -> Tensor:
    key = (geometry, angle, str(device), dtype)
    if key in _GRID_CACHE:
        _GRID_CACHE.move_to_end(key)
        return _GRID_CACHE[key]
    depth, height, width = geometry.input_shape
    side = geometry.canvas_shape[0]
    sd, sh, sw = geometry.spacing
    axis = (torch.arange(side, device=device, dtype=dtype) - (side - 1) / 2) * geometry.detector_spacing
    detector, ray = torch.meshgrid(axis, axis, indexing="ij")
    theta = math.radians(angle)
    h = (math.cos(theta) * detector + math.sin(theta) * ray + geometry.center_mm[0]) / sh
    w = (-math.sin(theta) * detector + math.cos(theta) * ray + geometry.center_mm[1]) / sw
    x = (2 * (w + .5) / width - 1).expand(depth, -1, -1)
    y = (2 * (h + .5) / height - 1).expand(depth, -1, -1)
    z = (2 * (torch.arange(depth, device=device, dtype=dtype) + .5) / depth - 1)
    z = z[:, None, None].expand(-1, side, side)
    grid = torch.stack((x, y, z), dim=-1).unsqueeze(0)
    _GRID_CACHE[key] = grid
    if len(_GRID_CACHE) > _MAX_CACHED_GRIDS:
        _GRID_CACHE.popitem(last=False)
    return grid


def project_masked(value: Tensor, valid: Tensor,
                   geometry: ProjectionGeometry, *, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Return normalized [B,V,1,D,L] projections and valid detector rays."""
    if value.ndim != 5 or value.shape[1] != 1 or tuple(value.shape[2:]) != geometry.input_shape:
        raise ValueError("Projection value must be [B,1,D,H,W] matching geometry")
    if valid.shape != value.shape or valid.device != value.device:
        raise ValueError("Projection validity must share shape and device")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError("Projection value must use float32 or float64")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Projection support epsilon must be positive and finite")
    mask = valid.to(value.dtype)
    if not torch.isfinite(value).all() or not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError("Projection value/valid must be finite with valid in [0,1]")
    projections, supports = [], []
    volume = torch.cat((value * mask, mask), dim=1)
    for angle in geometry.angles:
        grid = _sampling_grid(geometry, angle, value.device, value.dtype).expand(value.shape[0], -1, -1, -1, -1)
        sampled = F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        numerator, denominator = sampled.sum(dim=-1).split(1, dim=1)
        support = denominator > eps
        projections.append(torch.where(support, numerator / denominator.clamp_min(eps), torch.zeros_like(numerator)))
        supports.append(support)
    return torch.stack(projections, dim=1), torch.stack(supports, dim=1)


class ProjectionConsistencyLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.angles = tuple(config.projection.angles)
        self.spacing = tuple(config.normalization.spacing_mm)
        self.eps = float(config.projection.eps)
        self.last_valid_ray_counts = None

    def forward(self, final_logits: Tensor, target: Tensor) -> Tensor:
        if final_logits.ndim != 5 or final_logits.shape[1] != 2:
            raise ValueError("Projection requires [B,2,D,H,W] logits")
        if target.ndim == 4:
            target = target[:, None]
        if target.shape != final_logits[:, :1].shape:
            raise ValueError("Projection target must match highest-resolution logits")
        if not torch.isfinite(final_logits).all() or not torch.isfinite(target).all():
            raise ValueError("Projection logits and target must be finite")
        if not torch.all((target == 0) | (target == 1) | (target == 2)):
            raise ValueError("Projection target must use labels 0,1,2")
        valid = target != 2
        if not torch.all(valid.flatten(1).any(dim=1)):
            raise ValueError("Projection requires a valid voxel in every sample")
        geometry = make_projection_geometry(tuple(final_logits.shape[2:]), self.angles, self.spacing)
        with torch.autocast(device_type=final_logits.device.type, enabled=False):
            probabilities = torch.softmax(final_logits.float(), dim=1)[:, 1:2]
            predicted, rays = project_masked(probabilities, valid, geometry, eps=self.eps)
            with torch.no_grad():
                truth, truth_rays = project_masked((target == 1).float(), valid, geometry, eps=self.eps)
            if not torch.equal(rays, truth_rays):
                raise RuntimeError("Prediction and target detector supports differ")
            self.last_valid_ray_counts = rays.detach().sum(dim=(2, 3, 4))
            if not bool((self.last_valid_ray_counts > 0).all()):
                raise ValueError("Projection has no valid detector rays for a case/view")
            per_case_view = ((predicted - truth).square() * rays).sum(dim=(2, 3, 4)) / self.last_valid_ray_counts
            return per_case_view.mean()
