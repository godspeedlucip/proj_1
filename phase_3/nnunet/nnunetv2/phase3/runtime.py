"""Process-local checks for the pinned Phase 3 source and environment."""

from __future__ import annotations

from pathlib import Path
import sys


def verify_runtime(expected_root: Path) -> dict:
    """Fail early when another installed nnunetv2 shadows this source tree."""
    import nnunetv2

    root = Path(expected_root).resolve()
    package_file = Path(nnunetv2.__file__).resolve()
    if package_file.parent.parent != root:
        raise RuntimeError(f"nnunetv2 import root mismatch: {package_file} is outside {root}")
    import nnunetv2.paths as paths

    paths_file = Path(paths.__file__).resolve()
    if paths_file.parent.parent != root:
        raise RuntimeError(f"nnunetv2.paths import root mismatch: {paths_file} is outside {root}")
    import numpy
    import torch
    import monai

    return {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "package_file": str(package_file),
        "paths_file": str(paths_file),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "monai": monai.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
