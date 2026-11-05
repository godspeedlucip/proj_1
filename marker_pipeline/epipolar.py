"""Cone-beam epipolar geometry and differentiable heatmap support."""

from itertools import combinations
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _cross_matrix(point: np.ndarray) -> np.ndarray:
    x, y, z = point
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def fundamentals_from_projections(Ps: Sequence[np.ndarray]) -> np.ndarray:
    """Return ``F[target, source]`` for three finite-center cone-beam views."""
    if len(Ps) != 3:
        raise ValueError("geometry requires exactly three projection matrices")
    matrices = [np.asarray(P, dtype=float) for P in Ps]
    for P in matrices:
        if P.shape != (3, 4) or not np.isfinite(P).all() or np.linalg.matrix_rank(P) != 3:
            raise ValueError("geometry requires finite rank-3 projection matrices")
        if np.linalg.matrix_rank(P[:, :3]) != 3:
            raise ValueError("geometry requires finite cone-beam source centers")
    centers = [-np.linalg.solve(P[:, :3], P[:, 3]) for P in matrices]
    result = np.zeros((3, 3, 3, 3), dtype=float)
    for target in range(3):
        for source in range(3):
            if target == source:
                continue
            epipole = matrices[source] @ np.r_[centers[target], 1.0]
            fundamental = _cross_matrix(epipole) @ matrices[source] @ np.linalg.pinv(matrices[target])
            norm = np.linalg.norm(fundamental)
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError("geometry has degenerate relative projection")
            result[target, source] = fundamental / norm
    return result


def validate_relative_geometry(
    reference_F: np.ndarray,
    case_Ps: Sequence[np.ndarray],
    image_shape: tuple[int, int],
    *,
    tolerance_px: float = 0.25,
) -> None:
    """Reject changes to the frozen detector-to-detector epipolar mapping."""
    reference = np.asarray(reference_F, dtype=float)
    if reference.shape != (3, 3, 3, 3) or not np.isfinite(reference).all():
        raise ValueError("invalid reference geometry")
    h, w = image_shape
    if h < 2 or w < 2 or not np.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("invalid image shape or geometry tolerance")
    actual = fundamentals_from_projections(case_Ps)
    targets = np.array([[x, y, 1.0] for y in (0, (h - 1) / 2, h - 1)
                        for x in (0, (w - 1) / 2, w - 1)])
    corners = np.array([[x, y, 1.0] for y in (0, h - 1) for x in (0, w - 1)])
    for target in range(3):
        for source in range(3):
            if target == source:
                continue
            for point in targets:
                expected = reference[target, source] @ point
                observed = actual[target, source] @ point
                e_norm, o_norm = np.linalg.norm(expected[:2]), np.linalg.norm(observed[:2])
                if e_norm < 1e-12 or o_norm < 1e-12:
                    raise ValueError("geometry has an undefined epipolar line")
                expected, observed = expected / e_norm, observed / o_norm
                if np.dot(expected[:2], observed[:2]) < 0:
                    observed = -observed
                if np.max(np.abs(corners @ (expected - observed))) > tolerance_px:
                    raise ValueError("case geometry differs from frozen relative geometry")


class EpipolarSupport(nn.Module):
    """Sample source heatmaps along fixed epipolar lines in bounded row chunks."""

    def __init__(
        self,
        reference_F: np.ndarray,
        stride: int = 4,
        samples: int = 32,
        chunk_rows: int = 32,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        values = np.asarray(reference_F, dtype=float)
        if values.shape != (3, 3, 3, 3) or not np.isfinite(values).all():
            raise ValueError("reference_F must have shape [3,3,3,3]")
        if stride < 1 or samples < 2 or chunk_rows < 1 or temperature <= 0:
            raise ValueError("invalid epipolar sampling configuration")
        self.register_buffer("reference_F", torch.as_tensor(values, dtype=torch.float64))
        self.stride = int(stride)
        self.samples = int(samples)
        self.chunk_rows = int(chunk_rows)
        self.temperature = float(temperature)

    @staticmethod
    def _pool_centers(full_size: int, stride: int, *, device, dtype) -> torch.Tensor:
        starts = torch.arange(0, full_size, stride, device=device, dtype=dtype)
        ends = (starts + stride - 1).clamp(max=full_size - 1)
        return (starts + ends) / 2

    @staticmethod
    def _full_to_pool(coordinates: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if centers.numel() == 1:
            return torch.zeros_like(coordinates)
        left = torch.searchsorted(centers.contiguous(), coordinates.contiguous(), right=True) - 1
        left = left.clamp(0, centers.numel() - 2)
        fraction = (coordinates - centers[left]) / (centers[left + 1] - centers[left])
        return left.to(coordinates.dtype) + fraction

    @staticmethod
    def _line_grid(lines: torch.Tensor, height: int, width: int, samples: int):
        """Clip N lines to the detector and return N×K normalized samples."""
        a, b, c = lines.unbind(-1)
        tiny = 1e-12
        a_safe = torch.where(a.abs() > tiny, a, torch.ones_like(a))
        b_safe = torch.where(b.abs() > tiny, b, torch.ones_like(b))
        xmax, ymax = float(width - 1), float(height - 1)
        xs = torch.stack((torch.zeros_like(a), torch.full_like(a, xmax), -c / a_safe,
                          -(b * ymax + c) / a_safe), -1)
        ys = torch.stack((-c / b_safe, -(a * xmax + c) / b_safe,
                          torch.zeros_like(a), torch.full_like(a, ymax)), -1)
        valid = torch.stack((b.abs() > tiny, b.abs() > tiny, a.abs() > tiny,
                             a.abs() > tiny), -1)
        valid = valid & (xs >= -1e-5) & (xs <= xmax + 1e-5) & (ys >= -1e-5) & (ys <= ymax + 1e-5)
        candidates = torch.stack((xs.clamp(0, xmax), ys.clamp(0, ymax)), -1)
        pairs = list(combinations(range(4), 2))
        distances = torch.stack([
            torch.where(valid[:, i] & valid[:, j],
                        (candidates[:, i] - candidates[:, j]).square().sum(-1),
                        torch.full_like(a, -1.0))
            for i, j in pairs
        ], -1)
        best = distances.argmax(-1)
        has_segment = distances.gather(1, best[:, None]).squeeze(1) > 1e-10
        first = torch.as_tensor([i for i, _ in pairs], device=lines.device)[best]
        second = torch.as_tensor([j for _, j in pairs], device=lines.device)[best]
        rows = torch.arange(lines.shape[0], device=lines.device)
        start, end = candidates[rows, first], candidates[rows, second]
        t = torch.linspace(0, 1, samples, device=lines.device, dtype=lines.dtype)
        points = start[:, None] * (1 - t[None, :, None]) + end[:, None] * t[None, :, None]
        points[..., 0] = 2 * points[..., 0] / max(width - 1, 1) - 1
        points[..., 1] = 2 * points[..., 1] / max(height - 1, 1) - 1
        return points, has_segment

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 5 or probabilities.shape[1:3] != (3, 1):
            raise ValueError("epipolar input must have shape [B,3,1,H,W]")
        batch, _, _, height, width = probabilities.shape
        if height < 2 or width < 2:
            raise ValueError("epipolar input is too small")
        coarse = F.max_pool2d(
            probabilities.reshape(batch * 3, 1, height, width),
            self.stride, self.stride, ceil_mode=True,
        ).reshape(batch, 3, 1, (height + self.stride - 1) // self.stride,
                  (width + self.stride - 1) // self.stride)
        coarse_h, coarse_w = coarse.shape[-2:]
        x_centers = self._pool_centers(width, self.stride,
                                       device=probabilities.device, dtype=probabilities.dtype)
        y_centers = self._pool_centers(height, self.stride,
                                       device=probabilities.device, dtype=probabilities.dtype)
        target_x = self._full_to_pool(torch.arange(width,device=probabilities.device,
                                                    dtype=probabilities.dtype),x_centers)
        target_y = self._full_to_pool(torch.arange(height,device=probabilities.device,
                                                    dtype=probabilities.dtype),y_centers)
        target_x = 2 * target_x / (coarse_w - 1) - 1 if coarse_w > 1 else torch.zeros_like(target_x)
        target_y = 2 * target_y / (coarse_h - 1) - 1 if coarse_h > 1 else torch.zeros_like(target_y)
        grid_y,grid_x=torch.meshgrid(target_y,target_x,indexing="ij")
        target_grid=torch.stack((grid_x,grid_y),dim=-1)[None].expand(batch,-1,-1,-1)
        output = []
        for target in range(3):
            source_maps = []
            for source in range(3):
                if source == target:
                    continue
                fundamental = self.reference_F[target, source].to(
                    dtype=probabilities.dtype, device=probabilities.device
                )
                chunks = []
                for y0 in range(0, coarse_h, self.chunk_rows):
                    y1 = min(y0 + self.chunk_rows, coarse_h)
                    yy, xx = torch.meshgrid(
                        y_centers[y0:y1], x_centers,
                        indexing="ij",
                    )
                    points = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
                    lines = points @ fundamental.T
                    grid, valid = self._line_grid(lines, height, width, self.samples)
                    source_x = self._full_to_pool((grid[..., 0] + 1) * (width - 1) / 2, x_centers)
                    source_y = self._full_to_pool((grid[..., 1] + 1) * (height - 1) / 2, y_centers)
                    grid = torch.stack((2 * source_x / max(coarse_w - 1, 1) - 1,
                                        2 * source_y / max(coarse_h - 1, 1) - 1), dim=-1)
                    if coarse_w == 1:
                        grid[..., 0] = 0
                    if coarse_h == 1:
                        grid[..., 1] = 0
                    grid = grid.reshape(y1 - y0, coarse_w * self.samples, 2)
                    sampled = F.grid_sample(
                        coarse[:, source], grid[None].expand(batch, -1, -1, -1),
                        mode="bilinear", padding_mode="border", align_corners=True,
                    ).reshape(batch, 1, y1 - y0, coarse_w, self.samples)
                    support = self.temperature * (
                        torch.logsumexp(sampled / self.temperature, dim=-1)
                        - np.log(self.samples)
                    )
                    support = support * valid.reshape(1, 1, y1 - y0, coarse_w)
                    chunks.append(support)
                low_res = torch.cat(chunks, dim=2)
                full_support = F.grid_sample(low_res,target_grid,mode="bilinear",
                                             padding_mode="border",align_corners=True)
                validity_chunks = []
                for y0 in range(0, height, self.chunk_rows):
                    y1 = min(y0 + self.chunk_rows, height)
                    yy, xx = torch.meshgrid(
                        torch.arange(y0, y1, device=probabilities.device, dtype=probabilities.dtype),
                        torch.arange(width, device=probabilities.device, dtype=probabilities.dtype),
                        indexing="ij",
                    )
                    target_points = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
                    _, valid = self._line_grid(target_points @ fundamental.T, height, width, 2)
                    validity_chunks.append(valid.reshape(1, 1, y1 - y0, width))
                full_valid = torch.cat(validity_chunks, dim=2)
                source_maps.append(full_support * full_valid)
            output.append(torch.cat(source_maps, dim=1))
        return torch.stack(output, dim=1)
