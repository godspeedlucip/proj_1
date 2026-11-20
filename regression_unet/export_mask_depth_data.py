"""Add explicit binary support labels to an owned continuous projection package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile


def export_mask_depth_dataset(source: str | Path, destination: str | Path) -> Path:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and destination must be separate packages")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "phase2_projection_regression_v2":
        raise ValueError("expected the pure stent model v2 projection package")
    if manifest.get("target_kind") != "normalized_continuous_stent_model_projection":
        raise ValueError("expected normalized pure stent projection targets")
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, temporary, copy_function=os.link)
        mask_paths = {}
        for entry in manifest["samples"]:
            target_relative = Path(entry["target"])
            if target_relative not in mask_paths:
                target = tifffile.imread(temporary / target_relative)
                if target.ndim != 2 or not np.isfinite(target).all() or np.any(target < 0) or np.any(target > 1):
                    raise ValueError(f"invalid continuous target: {target_relative}")
                mask_relative = Path("masks") / target_relative.relative_to("targets")
                mask_file = temporary / mask_relative
                mask_file.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(mask_file, (target > 0).astype(np.uint8),
                                 compression="zlib", compressionargs={"level": 1})
                mask_paths[target_relative] = mask_relative
            entry["mask"] = mask_paths[target_relative].as_posix()
        manifest["schema_version"] = "phase2_mask_depth_v1"
        manifest["mask_kind"] = "binary_support_of_normalized_continuous_stent_projection"
        manifest["mask_definition"] = "mask = (continuous_projection > 0)"
        temporary_manifest = temporary / "manifest.json.new"
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
        temporary_manifest.replace(temporary / "manifest.json")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export_mask_depth_dataset(args.source, args.output_dir))


if __name__ == "__main__":
    main()
