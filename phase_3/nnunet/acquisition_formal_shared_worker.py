"""Run a smoke-qualified acquisition arm on a shared GPU with enough VRAM.

This operational wrapper leaves the frozen experiment code unchanged.  It only
replaces the worker's idle-GPU wait with an immediate memory-headroom check.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MIN_FREE_MIB = 35000


def require_memory_headroom(_arm: Path, gpu: int) -> None:
    free_mib = int(subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True).strip())
    if free_mib < MIN_FREE_MIB:
        raise RuntimeError(
            f"GPU {gpu} has {free_mib} MiB free; at least {MIN_FREE_MIB} MiB "
            "is required for concurrent formal training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("a60_n2", "a60_n5"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--stop-after", choices=("complete",), required=True)
    parser.add_argument("--scheduler-adoption", choices=("acquisition_launch.py worker",),
                        required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    import json
    info = json.loads((run_dir / "run.json").read_text())
    sys.path.insert(0, info["code_root"])
    from nnunetv2.phase3 import acquisition_run

    # All five smoke runs and raw datasets are already audited.  Abort rather
    # than entering a data build on a GPU shared with another job.
    arm_dir = run_dir / args.arm
    required = (arm_dir / "raw/audit.json", arm_dir / "smoke/passed.json",
                arm_dir / "smoke/checkpoint_audit.json",
                arm_dir / "smoke/results_audit.json")
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"{args.arm} is missing audited raw data or smoke artifacts")
    if not json.loads((arm_dir / "smoke/passed.json").read_text())["passed"]:
        raise RuntimeError(f"{args.arm} smoke training did not pass")
    require_memory_headroom(arm_dir, args.gpu)
    acquisition_run._wait_for_free_gpu = require_memory_headroom
    acquisition_run.worker(run_dir, args.arm, args.gpu, args.stop_after)


if __name__ == "__main__":
    main()
