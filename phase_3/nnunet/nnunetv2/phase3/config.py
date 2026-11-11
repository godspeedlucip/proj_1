"""Validated, immutable and portable Phase 3 experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "phase3_nnunet_experiment_v1"
UPSTREAM_COMMIT = "468cf803df9b267150ae2b6c0c59b8ac84f16227"

DEFAULT_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "upstream": {"version": "2.8.1", "commit": UPSTREAM_COMMIT},
    "run_kind": "formal",
    "seed": 2026,
    "cnn": {"channels": [32, 64, 128, 256, 320], "patch_size": [128, 128, 128],
            "batch_size": 2, "dimension": "3d_fullres", "num_input_channels": 2,
            "num_output_channels": 2, "norm": "InstanceNorm3d"},
    "swin": {"enabled": True, "init": "ssl", "monai_version": "1.3.0",
             "feature_size": 48, "frozen": True, "execute_through": "layers3"},
    "fusion": {"enabled": True, "bottleneck_enabled": True,
               "decoder_conditioning": True, "alpha_init": 0.0},
    "tubular": {"enabled": True, "dilations": [1, 2, 3, 5]},
    "segmentation": {"loss": "foreground_dice_bce", "smooth": 1.0},
    "deep_supervision": {"enabled": True, "weights": [4/7, 2/7, 1/7, 0.0]},
    "projection": {"enabled": True, "weight": 0.1, "angles": [0.0, 30.0, 60.0],
                   "coordinate_frame": "canonical_patch", "target_source": "label",
                   "operator_version": "label_parallel_v1", "eps": 1e-6},
    "normalization": {"spacing_mm": [0.2, 0.2, 0.2], "spacing_tolerance_mm": 1e-4,
                      "percentiles": [0.5, 99.5], "samples_per_group": 65536,
                      "stats_seed": 2026},
    "training": {"max_optimizer_steps": 10000, "updates_per_epoch": 250,
                 "optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.99,
                 "weight_decay": 3e-5, "nesterov": True, "poly_power": 0.9,
                 "foreground_probability": 0.33, "num_workers": 0,
                 "compile": False},
    "checkpoint": {"selection": "validation_group_dice"},
    "inference": {"tile_step_size": 0.5, "use_gaussian": True,
                  "use_mirroring": False, "threshold": 0.5,
                  "postprocessing": False},
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigSection(MappingProxyType({key: _freeze(item) for key, item in value.items()}))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, ConfigSection):
        return {key: _thaw(item) for key, item in value.values.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ConfigSection:
    values: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class Phase3Config:
    _section: ConfigSection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._section, name)

    @property
    def deep_supervision_weights(self) -> tuple[float, ...]:
        return self.deep_supervision.weights if self.deep_supervision.enabled else (1.0,)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._section)


def _same_keys(raw: Any, template: Any, where: str = "config") -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    missing = set(template) - set(raw)
    unknown = set(raw) - set(template)
    if missing or unknown:
        raise ValueError(f"{where}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key, value in template.items():
        if isinstance(value, dict):
            _same_keys(raw[key], value, f"{where}.{key}")


def _fixed(raw: dict, template: dict, section: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        if type(raw[section][key]) is not type(template[section][key]) or raw[section][key] != template[section][key]:
            raise ValueError(f"Unsupported {section}.{key}")


def _integer(value: Any, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}")


def _finite(value: Any, name: str, minimum: float | None = None, maximum: float | None = None) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"Invalid {name}")


def resolve_config(raw: dict[str, Any]) -> Phase3Config:
    """Reject unknown fields and unsupported combinations before any I/O."""
    template = DEFAULT_CONFIG
    _same_keys(raw, template)
    data = deepcopy(raw)
    if data["schema_version"] != SCHEMA_VERSION or data["upstream"] != template["upstream"]:
        raise ValueError("Unsupported schema or upstream source")
    if data["run_kind"] not in ("formal", "smoke"):
        raise ValueError("run_kind must be formal or smoke")
    _integer(data["seed"], "seed", 0)
    _fixed(data, template, "cnn", ("channels", "dimension", "num_input_channels", "num_output_channels", "norm"))
    patch = data["cnn"]["patch_size"]
    if not isinstance(patch, list) or len(patch) != 3:
        raise ValueError("Invalid cnn.patch_size")
    for edge in patch:
        _integer(edge, "cnn.patch_size", 32)
        if edge % 32:
            raise ValueError("cnn.patch_size must be divisible by 32")
    _integer(data["cnn"]["batch_size"], "cnn.batch_size", 1)
    _fixed(data, template, "swin", ("monai_version", "feature_size", "frozen"))
    for section, keys in (("swin", ("enabled",)), ("fusion", ("enabled", "bottleneck_enabled", "decoder_conditioning")),
                          ("tubular", ("enabled",)), ("deep_supervision", ("enabled",)),
                          ("projection", ("enabled",))):
        for key in keys:
            if type(data[section][key]) is not bool:
                raise ValueError(f"{section}.{key} must be boolean")
    if data["swin"]["init"] not in ("ssl", "random"):
        raise ValueError("swin.init must be ssl or random")
    if data["swin"]["execute_through"] not in ("layers2", "layers3"):
        raise ValueError("Unsupported Swin execution depth")
    if data["swin"]["enabled"] != data["fusion"]["enabled"]:
        raise ValueError("Swin and fusion switches conflict")
    if data["fusion"]["bottleneck_enabled"] and not data["swin"]["enabled"]:
        raise ValueError("Bottleneck needs Swin")
    if data["swin"]["enabled"]:
        depth = "layers3" if data["fusion"]["bottleneck_enabled"] else "layers2"
        if data["swin"]["execute_through"] != depth:
            raise ValueError("Swin depth conflicts with bottleneck")
    _finite(data["fusion"]["alpha_init"], "fusion.alpha_init")
    _fixed(data, template, "tubular", ("dilations",))
    _fixed(data, template, "segmentation", ("loss", "smooth"))
    _fixed(data, template, "deep_supervision", ("weights",))
    _fixed(data, template, "projection", ("coordinate_frame", "target_source", "operator_version", "eps"))
    _finite(data["projection"]["weight"], "projection.weight", 0.0)
    if data["projection"]["enabled"] != (data["projection"]["weight"] > 0):
        raise ValueError("projection.enabled conflicts with projection.weight")
    angles = data["projection"]["angles"]
    if not isinstance(angles, list) or not angles:
        raise ValueError("projection.angles must be nonempty")
    for angle in angles:
        _finite(angle, "projection.angle")
    _fixed(data, template, "normalization", tuple(template["normalization"]))
    _fixed(data, template, "training", ("optimizer", "learning_rate", "momentum", "weight_decay", "nesterov",
                                     "poly_power", "foreground_probability", "num_workers", "compile"))
    _integer(data["training"]["max_optimizer_steps"], "max_optimizer_steps", 1)
    _integer(data["training"]["updates_per_epoch"], "updates_per_epoch", 1)
    if data["training"]["max_optimizer_steps"] % data["training"]["updates_per_epoch"]:
        raise ValueError("Training updates must form complete epochs")
    _fixed(data, template, "checkpoint", ("selection",))
    _fixed(data, template, "inference", tuple(template["inference"]))
    if data["run_kind"] == "formal":
        _fixed(data, template, "cnn", ("patch_size", "batch_size"))
        _fixed(data, template, "training", ("max_optimizer_steps", "updates_per_epoch"))
    elif (data["training"]["max_optimizer_steps"] >= template["training"]["max_optimizer_steps"]
          or data["training"]["updates_per_epoch"] >= template["training"]["updates_per_epoch"]):
        raise ValueError("Smoke run requires a reduced training budget")
    return Phase3Config(_freeze(data))


def config_digest(config: Phase3Config) -> str:
    serialized = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
