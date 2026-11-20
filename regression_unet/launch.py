"""Direct Phase 2 CLI entry points; no shell environment setup is required."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT_ROOT / "model_outputs/models/phase2_nnunet/deps"),
                str(PROJECT_ROOT),
                str(PROJECT_ROOT / "model_networks/phase2"),
                str(PROJECT_ROOT / "model_networks/phase2/nnUNet")]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict", "tensorboard", "test"):
        raise SystemExit("usage: launch.py {train|predict|tensorboard|test} [arguments]")
    command = sys.argv[1]
    if command == "train":
        from regression_unet.train import main as train_main
        train_main(sys.argv[2:])
    elif command == "predict":
        from regression_unet.predict import main as predict_main
        predict_main(sys.argv[2:])
    elif command == "test":
        import pytest
        raise SystemExit(pytest.main([
            str(PROJECT_ROOT / "model_networks/phase2/regression_unet/test_regression_unet.py"),
            *sys.argv[2:]]))
    else:
        from tensorboard.main import run_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        run_main()


if __name__ == "__main__":
    main()
