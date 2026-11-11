"""Original-grid FDK prediction through nnU-Net's native sliding window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from nnunetv2.imageio.fdk_reader_writer import FDKNiftiIO
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.preprocessing.preprocessors.fdk_preprocessor import FDKPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from .config import Phase3Config, config_digest
from .data_conversion import make_views
from .manifest import json_digest
from .nifti_io import SpatialMeta, VolumeCase
from .planning import validate_phase3_plans
from .prior_stats import PriorStats


@dataclass
class Prediction:
    probability: np.ndarray
    mask: np.ndarray
    meta: SpatialMeta
    telemetry: dict


def predict_volume(network, volume: VolumeCase, *, plans: dict, config: Phase3Config,
                   stats: PriorStats, device) -> Prediction:
    validate_phase3_plans(plans)
    if plans["phase3"]["config_digest"] != config_digest(config):
        raise ValueError("Prediction configuration differs from model plans")
    if plans["phase3"]["stats_digest"] != json_digest(stats.to_dict()):
        raise ValueError("Prediction intensity statistics differ from model plans")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Prediction device must be available CPU or CUDA")
    start = time.perf_counter()
    manager = PlansManager(plans)
    configuration = manager.get_configuration("3d_fullres")
    dataset_json = {"channel_names": {"0": "noNorm", "1": "noNorm"},
                    "labels": {"background": 0, "stent_struts": 1, "ignore": 2},
                    "file_ending": ".nii.gz"}
    views = make_views(volume, stats)
    spacing = tuple(float(v) for v in np.linalg.norm(volume.meta.affine_mm[:3, :3], axis=0)[::-1])
    preprocessed, _, properties = FDKPreprocessor(verbose=False).run_case_npy(
        views, None, {"spacing": spacing, "fdk_meta": volume.meta},
        manager, configuration, dataset_json)
    previous_training = network.training
    previous_ds = network.decoder.deep_supervision
    try:
        network.eval()
        network.set_deep_supervision(False)
        predictor = nnUNetPredictor(tile_step_size=config.inference.tile_step_size,
                                    use_gaussian=config.inference.use_gaussian,
                                    use_mirroring=False, device=device, verbose=False,
                                    allow_tqdm=False)
        predictor.manual_initialization(network, manager, configuration, None, dataset_json,
                                        "nnUNetTrainerPhase3", ())
        logits = predictor.predict_sliding_window_return_logits(
            torch.from_numpy(np.ascontiguousarray(preprocessed)))
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite prediction logits")
        logits = logits.cpu()
        _, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, manager, configuration, predictor.label_manager, properties,
            return_probabilities=True, num_threads_torch=min(2, torch.get_num_threads()))
    finally:
        network.set_deep_supervision(previous_ds)
        network.train(previous_training)
    foreground = np.ascontiguousarray(probabilities[1], dtype=np.float32)
    if foreground.shape != volume.image_raw.shape:
        raise ValueError("Native prediction failed to restore original shape")
    mask = (foreground > config.inference.threshold).astype(np.uint8)
    telemetry = {"seconds": time.perf_counter() - start, "device": str(device),
                 "tile_step_size": config.inference.tile_step_size,
                 "use_gaussian": config.inference.use_gaussian, "use_mirroring": False,
                 "original_shape_dhw": list(volume.image_raw.shape),
                 "threshold": config.inference.threshold}
    return Prediction(foreground, mask, volume.meta, telemetry)


def write_prediction(prediction: Prediction, *, case_id: str, output: Path) -> tuple[Path, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    probability_path = output / f"{case_id}_prob.nii.gz"
    mask_path = output / f"{case_id}_mask.nii.gz"
    writer = FDKNiftiIO()
    properties = {"fdk_meta": prediction.meta}
    writer.write_probability(prediction.probability, probability_path, properties)
    writer.write_seg(prediction.mask, str(mask_path), properties)
    return probability_path, mask_path
