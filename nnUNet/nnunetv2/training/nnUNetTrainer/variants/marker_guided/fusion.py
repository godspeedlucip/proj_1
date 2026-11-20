"""Spatial marker fusion settings stored with the joint depth plans."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MarkerFusion:
    mode: str = "skip"
    axial_scale_px: float = 32.0

    def __post_init__(self):
        if self.mode not in ("skip", "encoder", "encoder-axial"):
            raise ValueError(f"Unknown marker fusion mode: {self.mode}")
        if not math.isfinite(self.axial_scale_px) or self.axial_scale_px <= 0:
            raise ValueError("axial_scale_px must be positive and finite")

    @property
    def axial(self):
        return self.mode == "encoder-axial"

    @property
    def channels(self):
        return 5 if self.axial else 4
