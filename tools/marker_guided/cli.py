"""Command line interface for validating Phase 1 packages and writing E5-U data."""

from __future__ import annotations

import argparse
import json

from .build_prior_channels import PriorConfig, build_e5u_dataset
from .validate_dataset import validate_prediction_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marker-guided")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "build-e5u"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--package", required=True)
        command.add_argument("--phase2-fold-id", required=True)
    build = sub.choices["build-e5u"]
    build.add_argument("--output-dir", required=True)
    build.add_argument("--dataset-id", type=int, required=True)
    build.add_argument("--sigma-0", type=float, default=2.0)
    build.add_argument("--k-sigma", type=float, default=1.0)
    build.add_argument("--w-0", type=float, default=3.0)
    build.add_argument("--k-w", type=float, default=1.0)
    build.add_argument("--endpoint-taper-px", type=float, default=4.0)
    build.add_argument("--d-min", type=float, default=1.0)
    args = parser.parse_args(argv)
    validated = validate_prediction_package(args.manifest, args.package, args.phase2_fold_id)
    if args.command == "validate":
        print(json.dumps({"valid": True, "phase2_fold_id": args.phase2_fold_id,
                          "samples": len(validated["joined"])}, indent=2))
        return 0
    config = PriorConfig(args.sigma_0, args.k_sigma, args.w_0, args.k_w,
                         args.endpoint_taper_px, args.d_min)
    output = build_e5u_dataset(args.manifest, args.package, args.phase2_fold_id,
                               args.output_dir, config, args.dataset_id)
    print(json.dumps({"output_dir": str(output), "dataset_id": args.dataset_id,
                      "phase2_fold_id": args.phase2_fold_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
