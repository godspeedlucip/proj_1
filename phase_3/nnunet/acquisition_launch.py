"""Absolute-path entry point for the controlled acquisition experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.phase3.acquisition_run import main

if __name__ == '__main__':
    raise SystemExit(main())
