from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_network import build_marker_guided_network
from nnunetv2.training.nnUNetTrainer.variants.marker_guided.marker_guided_transforms import get_marker_guided_training_transforms
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerMarkerGuidedFeatureFusion(nnUNetTrainer):
    def on_train_start(self):
        super().on_train_start()
        self._tb_writer = SummaryWriter(str(Path(self.output_folder) / "tensorboard"), flush_secs=30)
        frames = os.environ.get("PHASE2_REAL_FRAMES")
        markers = os.environ.get("PHASE2_REAL_MARKERS")
        if bool(frames) != bool(markers):
            raise ValueError("PHASE2_REAL_FRAMES and PHASE2_REAL_MARKERS must be set together")
        self._real_monitor = None
        if frames:
            if self.is_ddp:
                raise RuntimeError("per-epoch real-frame monitoring currently requires a single GPU")
            from tools.marker_guided.build_prior_channels import PriorConfig
            from .real_epoch_monitor import RealEpochMonitor
            prior = PriorConfig(**self.dataset_json["marker_guided"]["prior_config"])
            sample = os.environ.get("PHASE2_REAL_SAMPLE")
            self._real_monitor = RealEpochMonitor(
                frames, markers, Path(self.output_folder) / "real_predictions", prior_config=prior,
                selected_names=(sample,) if sample else None,
                require_empty_epochs=self.current_epoch == 0)
            prepared = self._real_monitor.prepare()
            self.print_to_log_file(f"Prepared {len(prepared)} real frames for per-epoch segmentation")

    def on_epoch_end(self):
        epoch = self.current_epoch + 1
        if self.global_rank == 0:
            writer = getattr(self, "_tb_writer", None)
            if writer is not None:
                for tag, key in (("train/loss", "train_losses"), ("valid/loss", "val_losses"),
                                 ("valid/pseudo_dice", "mean_fg_dice"), ("train/lr", "lrs")):
                    writer.add_scalar(tag, self.logger.get_value(key, step=-1), epoch)
                if getattr(self, "_real_monitor", None) is not None:
                    self._run_real_inference(epoch)
                writer.flush()
        super().on_epoch_end()

    def _run_real_inference(self, epoch: int):
        monitor = self._real_monitor
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device,
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        was_training = self.network.training
        self.set_deep_supervision_enabled(False)
        try:
            def segment(files):
                data, _, properties = preprocessor.run_case(
                    [str(path) for path in files], None, self.plans_manager,
                    self.configuration_manager, self.dataset_json)
                logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data)).cpu()
                mask = convert_predicted_logits_to_segmentation_with_correct_shape(
                    logits, self.plans_manager, self.configuration_manager, self.label_manager,
                    properties, num_threads_torch=1)
                return np.squeeze(mask, axis=0)

            summary = monitor.run_epoch(epoch, segment, self._tb_writer)
            self.print_to_log_file(f"Real epoch {epoch}: {summary['frame_count']} frames, "
                                   f"mean foreground fraction {summary['mean_foreground_fraction']:.6f}, "
                                   f"{summary['seconds']:.1f} s")
        finally:
            self.set_deep_supervision_enabled(True)
            self.network.train(was_training)

    def on_train_end(self):
        try:
            super().on_train_end()
        finally:
            writer = getattr(self, "_tb_writer", None)
            if writer is not None:
                writer.close()

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        if num_input_channels != 4:
            raise RuntimeError(f"Marker-guided E5-U expects exactly 4 input channels; received {num_input_channels}")
        if len(configuration_manager.patch_size) != 2:
            raise RuntimeError("Marker-guided E5-U supports only 2D plans")
        if configuration_manager.network_arch_class_name.split(".")[-1] != "PlainConvUNet":
            raise RuntimeError("Marker-guided E5-U supports only PlainConvUNet architecture plans")
        return build_marker_guided_network(configuration_manager, num_input_channels,
                                           num_output_channels, enable_deep_supervision)

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                               do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                               foreground_labels=None, regions=None, ignore_label=None):
        return get_marker_guided_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label)


class nnUNetTrainerMarkerGuidedSmoke(nnUNetTrainerMarkerGuidedFeatureFusion):
    """Short-budget Trainer for software smoke tests; architecture and pipeline stay identical."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
