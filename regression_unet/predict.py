"""Export a float32 implant projection from a trained regression U-Net."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tifffile
import torch

from tools.marker_guided.build_prior_channels import build_marker_channels

from .model import build_unet


def load_checkpoint_model(path: str | Path, device: str = "cpu") -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_unet(checkpoint["plans"],
                       model_family=checkpoint.get("config", {}).get("model_family", "plain"))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def predict_projection(model: torch.nn.Module, image: np.ndarray,
                       marker_xy: list[list[float]] | tuple,
                       device: str = "cpu") -> np.ndarray:
    """Predict an aligned, nonnegative float map at the original image size."""
    return predict_maps(model, image, marker_xy, device)["projection"]


def predict_maps(model: torch.nn.Module, image: np.ndarray,
                 marker_xy: list[list[float]] | tuple,
                 device: str = "cpu") -> dict[str, np.ndarray]:
    """Return the projection and, for cascades, the predicted mask and depth."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("image must be a finite 2D array")
    point, band, validity = build_marker_channels(
        {"pair_status": "reliable", "marker_xy": marker_xy,
         "error_radius_px": [0.0, 0.0]}, image.shape)
    normalized = (image - image.mean()) / max(float(image.std()), 1e-8)
    inputs = np.stack((normalized, point, band, validity), axis=0)
    factor = 2 ** (len(model.encoder.stages) - 1)
    height, width = image.shape
    padded_height = math.ceil(height / factor) * factor
    padded_width = math.ceil(width / factor) * factor
    padded = np.zeros((4, padded_height, padded_width), dtype=np.float32)
    padded[:, :height, :width] = inputs
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()
    with torch.inference_mode():
        with torch.autocast(device_type=torch_device.type,
                            enabled=torch_device.type == "cuda"):
            tensor = torch.from_numpy(padded[None]).to(torch_device)
            if hasattr(model, "mask_network"):
                details = model(tensor, return_aux=True)
                values = {"projection": details["projection"],
                          "mask_probability": details["mask_probability"],
                          "conditional_depth": details["depth"]}
            else:
                values = {"projection": model(tensor)}
    return {key: value[0, 0, :height, :width].float().clamp(0, 1).cpu().numpy()
            for key, value in values.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--markers-json", type=Path, required=True,
                        help='JSON object with "marker_xy": [[x1,y1],[x2,y2]]')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path,
                        help="optional 0/1 TIFF from the cascade mask head at probability 0.5")
    parser.add_argument("--conditional-depth-output", type=Path,
                        help="optional continuous TIFF before multiplication by mask probability")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    markers = json.loads(args.markers_json.read_text(encoding="utf-8"))["marker_xy"]
    model = load_checkpoint_model(args.checkpoint, args.device)
    maps = predict_maps(model, tifffile.imread(args.image), markers, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, maps["projection"], compression="zlib", compressionargs={"level": 1})
    if args.mask_output is not None:
        if "mask_probability" not in maps:
            raise ValueError("mask-output requires a mask_then_depth checkpoint")
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(args.mask_output, (maps["mask_probability"] >= 0.5).astype(np.uint8),
                         compression="zlib", compressionargs={"level": 1})
    if args.conditional_depth_output is not None:
        if "conditional_depth" not in maps:
            raise ValueError("conditional-depth-output requires a mask_then_depth checkpoint")
        args.conditional_depth_output.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(args.conditional_depth_output, maps["conditional_depth"],
                         compression="zlib", compressionargs={"level": 1})


if __name__ == "__main__":
    main()
