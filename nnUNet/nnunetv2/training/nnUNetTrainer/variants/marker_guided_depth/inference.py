"""Full-detector joint inference, shared by native fold validation and TIFF prediction."""
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from .data import pad_frame, prepare_image


@torch.inference_mode()
def predict_frame(network, image, points, marker_enabled, scale, device, cache_dir=None):
    channels = prepare_image(image, points, marker_enabled, cache_dir)
    x = pad_frame(channels).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        result = network(x, return_depth=True)
    h, w = image.shape
    def plane(key):
        return result[key][0, 0, :h, :w].float().cpu().numpy()
    mask = plane("predicted_mask").astype(np.uint8)
    norm = plane("depth_normalized")
    raw = norm * np.float32(scale)
    prob = plane("mask_probability")
    if (not np.isfinite(raw).all() or not np.isfinite(prob).all() or
            raw.min() < 0 or np.any(raw[mask == 0]) or not np.array_equal(prob > 0.5, mask)):
        raise ValueError("Invalid hard-mask joint prediction")
    return {"masks": mask, "depth_raw": raw, "depth_normalized": norm, "mask_probability": prob}


def save_frame(folder, name, values, save_probabilities):
    for key, plane in values.items():
        if key == "mask_probability" and not save_probabilities:
            continue
        dest = folder / key / (name + ".tif")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(dest, plane, compression="zlib", compressionargs={"level": 1})


def frame_metrics(values, target):
    truth = target > 0
    pred = values["masks"] != 0
    intersection = np.count_nonzero(pred & truth)
    n, k = int(truth.sum()), int(pred.sum())
    return {"dice": 2 * intersection / (n + k) if n + k else 1.,
            "foreground_mae_raw": float(np.abs(values["depth_raw"][truth] - target[truth]).mean()) if n else None,
            "predicted_foreground": k, "missed_foreground": n - intersection}


def export_fold(trainer, save_probabilities=False):
    joint = trainer.plans_manager.plans["joint_depth"]
    path = Path(joint["manifest"])
    manifest = json.loads(path.read_text())
    rows = [r for r in manifest["samples"] if r["split"] == "val"]
    if trainer.smoke:
        rows = rows[:2]
    output = Path(trainer.output_folder) / "validation_joint"
    output.mkdir(parents=True, exist_ok=True)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    summaries = []
    for row in rows:
        image = tifffile.imread(path.parent / row["image"])
        depth = tifffile.imread(path.parent / row["depth"])
        args = (trainer.network, image, row["marker_xy"])
        options = (manifest["target_scale"], trainer.device, path.parent / "prior_cache")
        pred = predict_frame(*args, trainer.marker_enabled, *options)
        save_frame(output, row["sample_id"], pred, save_probabilities)
        item = {k: row[k] for k in ("sample_id", "image", "case_id", "group_id", "view_id", "angle_deg", "marker_xy")}
        item.update(frame_metrics(pred, depth))
        if trainer.marker_enabled:
            without = predict_frame(*args, False, *options)
            item["same_checkpoint_marker_off"] = frame_metrics(without, depth)
        summaries.append(item)
    metadata = {"trainer": type(trainer).__name__, "marker_enabled": trainer.marker_enabled,
                "marker_source": manifest["marker_source"], "threshold": 0.5, "scale": manifest["target_scale"],
                "depth_definition": manifest["depth_definition"], "manifest_sha256": joint["manifest_sha256"],
                "case_count": len(rows), "views": summaries}
    result = output / "projection_metadata.json"
    result.write_text(json.dumps(metadata, indent=2))
    trainer.print_to_log_file(f"Exported {len(summaries)} paired mask/depth predictions to {output}")
    return result


def restore_network(model_folder, checkpoint_path, device):
    from nnunetv2.utilities.checkpoints import load_checkpoint_file
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from .network import build_joint_network
    plans = json.loads((model_folder / "plans.json").read_text())
    network = build_joint_network(PlansManager(plans).get_configuration("2d"), 4, 2, False)
    checkpoint = load_checkpoint_file(str(checkpoint_path), map_location="cpu")
    network.load_state_dict(checkpoint["network_weights"], strict=True)
    return network.to(device).eval(), plans["joint_depth"]
