"""Stride-aligned inference and synchronized network measurements."""
from itertools import chain, product
from statistics import median
from time import perf_counter

import torch
from torch.nn import functional as F


def _spatial_shape(x):
    if x.ndim not in (4, 5) or any(size == 0 for size in x.shape) or not x.is_floating_point():
        raise ValueError("expected a nonempty floating-point NCHW or NCDHW input")
    return x.shape[2:]


def _positive_stride(stride):
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")


def _validate_tile_grid(tile_shape, overlap, dims, stride):
    _positive_stride(stride)
    if not isinstance(tile_shape, (list, tuple)) or len(tile_shape) != dims:
        raise ValueError("tile_shape must have one size per spatial axis")
    if type(overlap) is int:
        overlap = (overlap,) * dims
    if not isinstance(overlap, (list, tuple)) or len(overlap) != dims:
        raise ValueError("overlap must be an integer or have one size per spatial axis")
    for size, shared in zip(tile_shape, overlap):
        if type(size) is not int or type(shared) is not int or not 0 <= shared < size:
            raise ValueError("each tile must have integer overlap in [0, tile size)")
        if size % stride or (size - shared) % stride:
            raise ValueError("tile sizes and steps must be multiples of network stride")
    return tuple(tile_shape), tuple(overlap)


def _pad_right(x, shape):
    padding = tuple(
        value
        for size, target in reversed(tuple(zip(x.shape[2:], shape)))
        for value in (0, target - size)
    )
    return F.pad(x, padding, mode="replicate") if any(padding) else x


def _predict(model, x, shape):
    prediction = model(x)
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.ndim != x.ndim
        or prediction.shape[0] != x.shape[0]
        or prediction.shape[1] < 1
        or prediction.shape[2:] != x.shape[2:]
        or not prediction.is_floating_point()
    ):
        raise ValueError("network must return a spatially shape-preserving floating-point tensor")
    prediction = prediction[(..., *(slice(0, size) for size in shape))]
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("network output contains non-finite values")
    return prediction


def _whole_predict(model, x, stride):
    shape = x.shape[2:]
    padded_shape = tuple(size + (-size % stride) for size in shape)
    return _predict(model, _pad_right(x, padded_shape), shape)


def _tile_starts(size, tile, step):
    final = ((max(size - tile, 0) + step - 1) // step) * step
    return range(0, final + 1, step)


def _blend_weights(tile_shape, overlap, device, dtype):
    weights = torch.ones(tile_shape, device=device, dtype=dtype)
    for axis, (tile, shared) in enumerate(zip(tile_shape, overlap)):
        if shared:
            positions = torch.arange(tile, device=device)
            distance = torch.minimum(positions + 1, tile - positions)
            ramp = distance.clamp(max=shared + 1).to(dtype) / (shared + 1)
            axis_shape = [1] * len(tile_shape)
            axis_shape[axis] = tile
            weights.mul_(ramp.view(axis_shape))
    return weights


@torch.inference_mode()
def tiled_predict(model, x, tile_shape, overlap, stride):
    """Return a blended prediction and spatial integer coverage counts.

    Tile origins and extents align with the network stride. Borders are
    replicated on the right for the network and cropped before accumulation.
    Accumulators stay on the input device, so a CPU input can stream tiles
    through a CUDA model. Context at tile boundaries can differ from a whole
    forward even with overlap.
    """
    shape = _spatial_shape(x)
    if any(module.training for module in model.modules()):
        raise ValueError("tiled prediction requires a model in eval mode")
    tile_shape, overlap = _validate_tile_grid(tile_shape, overlap, len(shape), stride)
    starts = [
        _tile_starts(size, tile, tile - shared)
        for size, tile, shared in zip(shape, tile_shape, overlap)
    ]
    reference = next(chain(model.parameters(), model.buffers()), x)
    coverage = torch.zeros(shape, device=x.device, dtype=torch.int32)
    accum = None
    for origin in product(*starts):
        region = tuple(slice(o, min(o + tile, size)) for o, tile, size in zip(origin, tile_shape, shape))
        patch = x[(..., *region)].to(reference.device)
        patch_shape = patch.shape[2:]
        prediction = _predict(model, _pad_right(patch, tile_shape), patch_shape).to(x.device)
        if accum is None:
            output_dtype = prediction.dtype
            dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
            accum = torch.zeros((x.shape[0], prediction.shape[1], *shape), device=x.device, dtype=dtype)
            weight_sum = torch.zeros(shape, device=x.device, dtype=dtype)
            weights = _blend_weights(tile_shape, overlap, x.device, dtype)
        local_weight = weights[tuple(slice(0, size) for size in patch_shape)]
        accum[(..., *region)].add_(prediction * local_weight)
        weight_sum[region].add_(local_weight)
        coverage[region].add_(1)

    accum.div_(weight_sum)
    return accum.to(output_dtype), coverage


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_path(forward, x_cpu, device, warmup, repeats):
    phases = ("host_to_device", "forward", "device_to_host", "total")
    samples = {phase: [] for phase in phases}
    for index in range(warmup + repeats):
        _synchronize(device)
        if index == warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        x = x_cpu.to(device)
        _synchronize(device)
        input_ready = perf_counter()
        prediction, coverage = forward(x)
        _synchronize(device)
        computed = perf_counter()
        host = prediction.cpu()
        _synchronize(device)
        host_ready = perf_counter()
        del x, prediction
        if index >= warmup:
            for phase, duration in zip(phases, (
                input_ready - started, computed - input_ready,
                host_ready - computed, host_ready - started,
            )):
                samples[phase].append(duration)
        if index + 1 < warmup + repeats:
            del coverage, host
    report = {
        "status": "ok",
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    for phase, values in samples.items():
        report[f"{phase}_seconds"] = values
        report[f"median_{phase}_seconds"] = median(values)
    if coverage is not None:
        report["coverage_min"] = int(coverage.min())
        report["coverage_max"] = int(coverage.max())
    return report, host


@torch.inference_mode()
def benchmark_forward(
    model, x_cpu, *, device="cpu", stride=1, tile_shape=None, overlap=None,
    warmup=2, repeats=5, compare_whole=True,
):
    """Measure synchronized, device-resident network paths and stage transfers.

    Each path transfers the input once and the prediction once per iteration.
    Forward timing includes padding, validation and tiled blending. CUDA OOM
    during one path is recorded so the other path can still be measured.
    """
    shape = _spatial_shape(x_cpu)
    _positive_stride(stride)
    if x_cpu.device.type != "cpu":
        raise ValueError("benchmark input must start on CPU")
    if any(module.training for module in model.modules()):
        raise ValueError("benchmark requires a model in eval mode")
    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("warmup and repeats must be positive integers")
    if type(compare_whole) is not bool:
        raise ValueError("compare_whole must be boolean")
    if tile_shape is None and overlap is not None:
        raise ValueError("overlap requires tile_shape")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("synchronized benchmarking supports CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    paths = {}
    if compare_whole or tile_shape is None:
        paths["whole"] = lambda x: (_whole_predict(model, x, stride), None)
    if tile_shape is not None:
        tile_shape, overlap = _validate_tile_grid(
            tile_shape, 0 if overlap is None else overlap, len(shape), stride,
        )
        paths["tiled"] = lambda x: tiled_predict(model, x, tile_shape, overlap, stride)
    model.to(device)
    report = {
        "device": str(device), "input_shape": list(x_cpu.shape), "stride": stride,
        "tile_shape": tile_shape, "overlap": overlap,
        "warmup": warmup, "repeats": repeats, "paths": {}, "disagreement": None,
        "timing_scope": (
            "warmed network forward with separate whole-input host-to-device and "
            "prediction device-to-host timings, synchronized per phase on CUDA. "
            "Forward includes device-resident padding, validation, and tiled blending. "
            "CPU timings include no inter-device transfer. Excludes model setup, I/O, "
            "geometry, and disagreement metrics."
        ),
    }
    outputs = {}
    for name, forward in paths.items():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            report["paths"][name], outputs[name] = _measure_path(
                forward, x_cpu, device, warmup, repeats,
            )
        except torch.cuda.OutOfMemoryError as exc:
            report["paths"][name] = {
                "status": "cuda_out_of_memory", "error": str(exc),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            }
        if report["paths"][name]["status"] == "cuda_out_of_memory":
            torch.cuda.empty_cache()
    if {"whole", "tiled"} <= outputs.keys():
        absolute = (outputs["whole"].double() - outputs["tiled"].double()).abs_()
        report["disagreement"] = {
            "max_absolute_error": absolute.max().item(),
            "mean_absolute_error": absolute.mean().item(),
            "rmse": absolute.square_().mean().sqrt().item(),
            "equivalence_guaranteed": False,
        }
    return report
