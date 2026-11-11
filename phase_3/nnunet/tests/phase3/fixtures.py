from copy import deepcopy
import json
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_CONFIG = {
    "schema_version": "phase3_nnunet_experiment_v1",
    "upstream": {"version": "2.8.1", "commit": "468cf803df9b267150ae2b6c0c59b8ac84f16227"},
    "run_kind": "formal",
    "seed": 2026,
    "cnn": {"channels": [32, 64, 128, 256, 320], "patch_size": [128, 128, 128],
            "batch_size": 2, "dimension": "3d_fullres", "num_input_channels": 2,
            "num_output_channels": 2, "norm": "InstanceNorm3d"},
    "swin": {"enabled": True, "init": "ssl", "monai_version": "1.3.0",
             "feature_size": 48, "frozen": True, "execute_through": "layers3"},
    "fusion": {"enabled": True, "bottleneck_enabled": True,
               "decoder_conditioning": True, "alpha_init": 0.0},
    "tubular": {"enabled": True, "dilations": [1, 2, 3, 5]},
    "segmentation": {"loss": "foreground_dice_bce", "smooth": 1.0},
    "deep_supervision": {"enabled": True, "weights": [4/7, 2/7, 1/7, 0.0]},
    "projection": {"enabled": True, "weight": 0.1, "angles": [0.0, 30.0, 60.0],
                   "coordinate_frame": "canonical_patch", "target_source": "label",
                   "operator_version": "label_parallel_v1", "eps": 1e-6},
    "normalization": {"spacing_mm": [0.2, 0.2, 0.2], "spacing_tolerance_mm": 1e-4,
                      "percentiles": [0.5, 99.5], "samples_per_group": 65536,
                      "stats_seed": 2026},
    "training": {"max_optimizer_steps": 10000, "updates_per_epoch": 250,
                 "optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.99,
                 "weight_decay": 3e-5, "nesterov": True, "poly_power": 0.9,
                 "foreground_probability": 0.33, "num_workers": 0,
                 "compile": False},
    "checkpoint": {"selection": "validation_group_dice"},
    "inference": {"tile_step_size": 0.5, "use_gaussian": True,
                  "use_mirroring": False, "threshold": 0.5,
                  "postprocessing": False},
}


def make_config(**overrides):
    raw = deepcopy(DEFAULT_CONFIG)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split("__")
        section = raw
        for key in keys[:-1]:
            section = section[key]
        section[keys[-1]] = value
    return raw


def synthetic_dataset(root: Path, shape=(16, 18, 20), count=4):
    """Four independent source groups with explicit 3-view provenance."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for i in range(count):
        image = np.zeros(shape, dtype=np.float32)
        label = np.zeros(shape, dtype=np.uint8)
        label[2:5, 3:10, 4:6] = 1
        image += label * (i + 1)
        affine = np.diag([0.2, 0.2, 0.2, 1.0])
        for stem, array in ((f"case_{i}_0000", image), (f"case_{i}", label)):
            nii = nib.Nifti1Image(array, affine)
            nii.header.set_xyzt_units("mm")
            nib.save(nii, root / f"{stem}.nii.gz")
        cases.append({
            "case_id": f"case_{i}", "group_id": f"g{i}",
            "source_model_ids": [f"source_{i}"], "patient_id": None,
            "image": f"case_{i}_0000.nii.gz", "label": f"case_{i}.nii.gz",
            "input_kind": "fdk_3view", "angles_deg": [0.0, 30.0, 60.0],
            "projection_source": "ideal_mask", "upstream_run_id": None,
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "fdk_stent_manifest_v1", "target_semantics": "stent_struts",
        "cases": cases, "upstream_registry": {},
        "group_provenance": {f"g{i}": "independent synthetic source" for i in range(count)},
    }), encoding="utf-8")
    splits = root / "splits.json"
    splits.write_text(json.dumps({
        "schema_version": "fdk_stent_splits_v1",
        "folds": {"0": {"train": ["g0", "g1"], "validation": ["g2"], "test": ["g3"]}},
    }), encoding="utf-8")
    return manifest, splits
