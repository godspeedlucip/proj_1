from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class View:
    view_id: int
    angle_deg: float
    projection_path: str
    image: np.ndarray
    marker_xy: np.ndarray
    P: np.ndarray
    mask: np.ndarray | None = None
    mask_path: str | None = None


@dataclass
class Case:
    case_id: str
    group_id: str
    views: list[View]
    difficulty: str = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)

