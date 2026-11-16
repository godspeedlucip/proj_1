"""Fixed mathematical fixtures for the CPU cone-beam geometry implementation."""
from dataclasses import dataclass, replace

import numpy as np

from stent_repro.configuration import resolve_config
from stent_repro.geometry import Grid, backproject_each, circular_view, fdk, forward_project, project_points


_ANGLES = (0., 53., 123.)
_SDD = 200.
_SID = 100.
_SHAPE = (25, 29)
_PITCH = 0.5
_SPACINGS = (1., 0.5, 0.25, 0.125, 0.0625)


def _analytic_uv(points, angle, sid, sdd, shape, pitch):
    x, y, z = points.T
    cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    scale = sdd / ((sid - x * sine + y * cosine) * pitch)
    h, w = shape
    return np.column_stack(((w - 1) / 2 + scale * (x * cosine + y * sine), (h - 1) / 2 + scale * z))


def _point_probe(g):
    sid, sdd = g["sid_mm"], g["sdd_mm"]
    shape, pitch = tuple(g["detector_shape"]), g["pixel_size_mm"]
    points = np.array([[1., 2., 3.], [-2., .75, -1.5], [.25, -1.25, 0.]]) * min(1., sid / 16)

    def measure(angle):
        view = circular_view(angle, sdd, sid, shape, pitch)
        expected = _analytic_uv(points, angle, sid, sdd, shape, pitch)
        observed = project_points(points, view)
        measured = np.isfinite(observed).all()
        forward = (view.detector_center - view.source) / sdd
        return {
            "angle_deg": angle,
            "expected_uv_pixels": expected.tolist(),
            "observed_uv_pixels": observed.tolist() if measured else None,
            "max_error_pixels": float(np.max(np.linalg.norm(observed - expected, axis=1))) if measured else None,
            "determinant_u_v_source_to_detector": float(np.linalg.det(np.stack([view.u_axis, view.v_axis, forward]))),
        }

    configured = [measure(angle) for angle in g["angles_deg"]]
    cardinal = [measure(angle) for angle in (0., 90., 180., 270.)]
    rows = configured + cardinal
    errors = [row["max_error_pixels"] for row in rows]
    return {
        "world_points_xyz_mm": points.tolist(), "configured_views": configured, "cardinal_probes": cardinal,
        "max_error_pixels": None if None in errors else max(errors),
        "threshold_pixels": 0.1, "expected_determinant": -1.,
        "handedness_passed": all(
            abs(row["determinant_u_v_source_to_detector"] + 1.) < 1e-10 for row in rows
        ),
    }


def _grid_xyz(grid):
    sx, sy, sz = grid.spacing_mm
    ox, oy, oz = grid.origin_mm
    nz, ny, nx = grid.shape
    z, y, x = np.meshgrid(
        oz + np.arange(nz) * sz, oy + np.arange(ny) * sy, ox + np.arange(nx) * sx,
        indexing="ij", sparse=True,
    )
    return x, y, z


def _order_probe(views):
    # Detector ramps distinguish u/v swaps, flips, and channel permutations.
    grid = Grid((3, 4, 5), (.37, .59, .23), (.17, -.42, .31))
    points = np.stack(np.broadcast_arrays(*_grid_xyz(grid)), axis=-1).reshape(-1, 3)
    v, u = np.indices(_SHAPE)
    markers = (2., 31., 73.)
    images = np.stack([marker + 3. * u - 2. * v for marker in markers])
    expected = []
    for angle, marker in zip(_ANGLES, markers):
        uv = _analytic_uv(points, angle, _SID, _SDD, _SHAPE, _PITCH)
        expected.append((marker + 3. * uv[:, 0] - 2. * uv[:, 1]).reshape(grid.shape))
    observed = backproject_each(images, views, grid)
    permutation = [2, 0, 1]
    ordered_views = [views[i] for i in permutation]
    ordered_angles = [_ANGLES[i] for i in permutation]
    reordered = backproject_each(images[permutation], ordered_views, grid)
    fdk_original = fdk(images, views, grid, _ANGLES)
    fdk_reordered = fdk(images[permutation], ordered_views, grid, ordered_angles)
    try:
        fdk(images, ordered_views, grid, _ANGLES)
    except ValueError:
        rejected = True
    else:
        rejected = False
    return {
        "angles_deg": list(_ANGLES),
        "grid_shape_zyx": list(grid.shape), "grid_origin_xyz_mm": list(grid.origin_mm),
        "grid_spacing_xyz_mm": list(grid.spacing_mm), "applied_permutation": permutation,
        "max_analytic_error": float(np.max(np.abs(observed - expected))),
        "permuted_backprojection_max_error": float(np.max(np.abs(reordered - observed[permutation]))),
        "permuted_fdk_max_error": float(np.max(np.abs(fdk_reordered - fdk_original))),
        "fdk_rejects_mismatched_angles": rejected,
    }


def _fixture_grid(spacing):
    return Grid((round(6. / spacing),) * 3, (spacing,) * 3, (-3. + spacing / 2.,) * 3)


def _rays(view):
    h, w = view.detector_shape
    v, u = np.indices((h, w))
    offsets_u = (u.ravel() - (w - 1) / 2.) * view.pixel_size_mm
    offsets_v = (v.ravel() - (h - 1) / 2.) * view.pixel_size_mm
    rays = (view.detector_center - view.source
            + offsets_u[:, None] * view.u_axis + offsets_v[:, None] * view.v_axis)
    lengths = np.linalg.norm(rays, axis=1)
    return rays / lengths[:, None], lengths


@dataclass(frozen=True)
class _Cylinder:
    name: str
    center: tuple
    direction: tuple
    radius: float
    length: float

    @property
    def axis(self):
        return np.array(self.direction) / np.linalg.norm(self.direction)

    @property
    def volume(self):
        return float(np.pi * self.radius ** 2 * self.length)

    def description(self):
        return {
            "center_xyz_mm": list(self.center), "axis_xyz": self.axis.tolist(),
            "radius_mm": self.radius, "length_mm": self.length,
        }

    def mask(self, x, y, z):
        dx, dy, dz = x - self.center[0], y - self.center[1], z - self.center[2]
        ax, ay, az = self.axis
        axial = dx * ax + dy * ay + dz * az
        radial_squared = dx ** 2 + dy ** 2 + dz ** 2 - axial ** 2
        return (radial_squared <= self.radius ** 2) & (np.abs(axial) <= self.length / 2.)

    def projection(self, view):
        """Intersect each finite ray with both the side wall and end caps."""
        directions, ray_limits = _rays(view)
        source_offset = view.source - self.center
        axis = self.axis
        ray_axial, source_axial = directions @ axis, source_offset @ axis
        ray_radial = directions - ray_axial[:, None] * axis
        source_radial = source_offset - source_axial * axis
        a = np.sum(ray_radial ** 2, axis=1)
        b = ray_radial @ source_radial
        c = source_radial @ source_radial - self.radius ** 2
        discriminant = b ** 2 - a * c
        side_cross = a > 1e-14
        cap_cross = np.abs(ray_axial) > 1e-14
        root = np.sqrt(np.maximum(0, discriminant))
        with np.errstate(divide="ignore", invalid="ignore"):
            side_lo = np.where(side_cross, (-b - root) / a, -np.inf)
            side_hi = np.where(side_cross, (-b + root) / a, np.inf)
            cap_a = (-self.length / 2. - source_axial) / ray_axial
            cap_b = (self.length / 2. - source_axial) / ray_axial
        cap_lo = np.where(cap_cross, np.minimum(cap_a, cap_b), -np.inf)
        cap_hi = np.where(cap_cross, np.maximum(cap_a, cap_b), np.inf)
        start = np.maximum(np.maximum(side_lo, cap_lo), 0.)
        stop = np.minimum(np.minimum(side_hi, cap_hi), ray_limits)
        hit_side = np.where(side_cross, discriminant >= 0., c <= 0.)
        hit_caps = cap_cross | (abs(source_axial) <= self.length / 2.)
        return np.where(hit_side & hit_caps, np.maximum(stop - start, 0.), 0.).reshape(view.detector_shape)


def _probe_rod(rod):
    grid = _fixture_grid(0.0625)
    mask = rod.mask(*_grid_xyz(grid))
    view = circular_view(0, _SDD, _SID, (1, 1), _PITCH)
    shift = np.array([rod.center[0], 0., rod.center[2]])
    view = replace(view, source=view.source + shift, detector_center=view.detector_center + shift)
    expected = float(rod.projection(view).item())
    step = 0.04
    first = float(forward_project(mask, grid, [view], step_mm=step).item())
    second = float(forward_project(mask, grid, [view], step_mm=step / 2.).item())
    return {
        "name": rod.name, **rod.description(),
        "ray_source_xyz_mm": view.source.tolist(), "ray_target_xyz_mm": view.detector_center.tolist(),
        "reference": "analytic finite cylinder side-and-cap intersection",
        "grid_spacing_mm": grid.spacing_mm[0], "steps_mm": [step, step / 2.],
        "reference_path_mm": expected, "observed_paths_mm": [first, second],
        "relative_error": abs(first - expected) / expected,
        "step_halving_relative_change": abs(first - second) / abs(second) if second else None,
        "path_error_threshold": 0.05, "step_change_threshold": 0.01,
    }


@dataclass(frozen=True)
class _SweptRing:
    center: tuple = (.3, -.2, .08)
    major_radius: float = 1.5
    radius: float = .12
    name: str = "swept_ring"

    @property
    def volume(self):
        return float(2. * np.pi ** 2 * self.major_radius * self.radius ** 2)

    def description(self):
        return {
            "center_xyz_mm": list(self.center), "axis_xyz": [0., 0., 1.],
            "major_radius_mm": self.major_radius, "strut_radius_mm": self.radius,
        }

    def mask(self, x, y, z):
        radial = np.hypot(x - self.center[0], y - self.center[1]) - self.major_radius
        return radial ** 2 + (z - self.center[2]) ** 2 <= self.radius ** 2

    def projection(self, view, step):
        # This ray interval encloses the entire fixture for every detector ray.
        directions, _ = _rays(view)
        samples = _SID - 6. + (np.arange(round(12. / step)) + .5) * step
        paths = []
        for begin in range(0, len(directions), 128):
            xyz = view.source + directions[begin:begin + 128, None, :] * samples[None, :, None]
            paths.append(self.mask(xyz[..., 0], xyz[..., 1], xyz[..., 2]).sum(axis=1) * step)
        return np.concatenate(paths).reshape(view.detector_shape)


def _relative_l1(observed, expected):
    mass = np.abs(expected).sum()
    return float(np.abs(observed - expected).sum() / mass) if mass else None


def _resolution_probe(fixture, views, reference, reference_details):
    rows = []
    for spacing in _SPACINGS:
        grid = _fixture_grid(spacing)
        mask = fixture.mask(*_grid_xyz(grid))
        voxels = int(np.count_nonzero(mask))
        volume = voxels * spacing ** 3
        half, quarter = spacing / 2., spacing / 4.
        projection = forward_project(mask, grid, views, step_mm=half)
        refined = forward_project(mask, grid, views, step_mm=quarter)
        rows.append({
            "spacing_mm": spacing, "origin_xyz_mm": list(grid.origin_mm), "shape_zyx": list(grid.shape),
            "status": "sampled" if voxels else "aliased_away",
            "occupied_voxels": voxels, "strut_diameter_in_voxels": 2. * fixture.radius / spacing,
            "volume_mm3": volume, "volume_relative_error": abs(volume - fixture.volume) / fixture.volume,
            "ray_steps_mm": [half, quarter],
            "projection_relative_l1_error": _relative_l1(refined, reference),
            "step_halving_relative_l1": _relative_l1(projection, refined),
        })
    return {
        "name": fixture.name, **fixture.description(),
        "reference_volume_mm3": fixture.volume,
        "reference_projection": reference_details,
        "resolutions": rows,
    }


def _resolution_suite(views):
    ring = _SweptRing()
    step = .0125
    ring_first = np.stack([ring.projection(view, step) for view in views])
    ring_refined = np.stack([ring.projection(view, step / 2.) for view in views])
    strut = _Cylinder("thin_tilted_strut", (.33, -.15, .08), (.36, .60, .71), .12, 3.)
    strut_reference = np.stack([strut.projection(view) for view in views])
    return {
        "world_bounds_xyz_mm": [[-3.] * 3, [3.] * 3], "angles_deg": list(_ANGLES),
        "detector_shape_hw": list(_SHAPE), "pixel_size_mm": _PITCH, "sdd_mm": _SDD, "sid_mm": _SID,
        "occupancy": "point sampling at voxel centers of fixed continuous solids",
        "use_for_pass_gate": False,
        "fixtures": [
            _resolution_probe(ring, views, ring_refined, {
                "method": "continuous solid indicator via midpoint integration",
                "steps_mm": [step, step / 2.],
                "step_halving_relative_l1": _relative_l1(ring_first, ring_refined),
            }),
            _resolution_probe(strut, views, strut_reference, {
                "method": "analytic finite cylinder side-and-cap intersection",
            }),
        ],
    }


def geometry_suite(config):
    """Return JSON-ready CPU point, ordering, ray-length and grid-resolution evidence.

    Points use the configured acquisition parameters. Other checks use fixed
    physical fixtures and detector settings across all profiles.
    """
    config = resolve_config(config)
    points = _point_probe(config["geometry"])
    views = [circular_view(angle, _SDD, _SID, _SHAPE, _PITCH) for angle in _ANGLES]
    order = _order_probe(views)
    rods = [
        _probe_rod(_Cylinder("off_axis_cylinder", (.85, -.45, .35), (0., 0., 1.), .85, 4.5)),
        _probe_rod(_Cylinder("tilted_finite_rod", (.85, -.45, .35), (.24, .8, .55), .85, 1.1)),
    ]
    gates = {
        "points": (
            points["max_error_pixels"] is not None
            and points["max_error_pixels"] < points["threshold_pixels"]
            and points["handedness_passed"]
        ),
        "view_order": (
            order["max_analytic_error"] < 1e-4
            and order["permuted_backprojection_max_error"] < 1e-7
            and order["permuted_fdk_max_error"] < 1e-7
            and order["fdk_rejects_mismatched_angles"]
        ),
        "analytic_rods": all(
            rod["relative_error"] <= rod["path_error_threshold"]
            and rod["step_halving_relative_change"] is not None
            and rod["step_halving_relative_change"] <= rod["step_change_threshold"]
            for rod in rods
        ),
    }
    return {
        "scope": "synthetic_cpu_geometry_fixtures",
        "point_projection": points, "view_order": order, "rod_probes": rods,
        "resolution_convergence": _resolution_suite(views),
        "engineering_gates": gates, "engineering_gate_passed": all(gates.values()),
    }
