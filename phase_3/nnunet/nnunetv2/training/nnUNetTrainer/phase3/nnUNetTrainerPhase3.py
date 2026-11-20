"""Native nnU-Net lifecycle with Phase 3 data, network and masked losses."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import distributed as dist

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.checkpointing import (atomic_save_checkpoint, build_phase3_checkpoint,
                                          evaluate_validation_group_dice, restore_phase3_state)
from nnunetv2.phase3.initialization import restore_rng_state
from nnunetv2.phase3.losses import Phase3CompositeLoss
from nnunetv2.phase3.network import build_phase3_network
from nnunetv2.phase3.planning import validate_phase3_plans
from nnunetv2.phase3.prior_weights import initialize_prior
from nnunetv2.phase3.training_support import build_phase3_loaders, get_strict_split, prior_artifact_from_context
from nnunetv2.phase3.tensorboard_reporting import Phase3TensorBoard, ValidationPreviewCollector
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPhase3(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        validate_phase3_plans(plans)
        if configuration != "3d_fullres" or fold != 0:
            raise ValueError("Phase 3 requires 3d_fullres and internal fold 0")
        if dist.is_available() and dist.is_initialized():
            raise ValueError("Phase 3 supports one process and one device")
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Phase 3 supports CPU smoke or single CUDA training")
        if set(dataset_json.get("channel_names", {})) != {"0", "1"}:
            raise ValueError("Phase 3 requires exactly two input views")
        target_semantics = plans["phase3"].get("target_semantics", "stent_struts")
        if dataset_json.get("labels") != {"background": 0, target_semantics: 1, "ignore": 2}:
            raise ValueError("Phase 3 labels differ from the prepared target semantics")
        self.config = resolve_config(plans["phase3"]["config"])
        self.continue_training = bool(plans.get("continue_training", False))
        self.global_step = 0
        self.prior_load_report = None
        self.best_group_dice = None
        self.best_step = None
        self._pending_phase3_state = None
        self._restore_mode = False
        self._phase3_tensorboard = None
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = self.config.training.learning_rate
        self.weight_decay = self.config.training.weight_decay
        self.oversample_foreground_percent = self.config.training.foreground_probability
        self.probabilistic_oversampling = True
        self.num_iterations_per_epoch = self.config.training.updates_per_epoch
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = self.config.training.max_optimizer_steps // self.num_iterations_per_epoch
        self.enable_deep_supervision = self.config.deep_supervision.enabled
        self.save_every = 1

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_phase3_network(plans_manager, configuration_manager, num_input_channels,
                                    num_output_channels, enable_deep_supervision)

    def _build_loss(self):
        return Phase3CompositeLoss(self.config)

    def _do_i_compile(self):
        return False

    def initialize(self):
        super().initialize()
        from nnunetv2.phase3.storage import FDKDatasetBlosc2
        self.dataset_class = FDKDatasetBlosc2

    def configure_optimizers(self):
        parameters = [part for part in self.network.parameters() if part.requires_grad]
        optimizer = torch.optim.SGD(parameters, self.initial_lr,
                                    momentum=self.config.training.momentum,
                                    weight_decay=self.weight_decay,
                                    nesterov=self.config.training.nesterov)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs,
                                    exponent=self.config.training.poly_power)
        return optimizer, scheduler

    def do_split(self):
        return get_strict_split(self)

    def get_dataloaders(self):
        self.dataloader_train, self.dataloader_val = build_phase3_loaders(self)
        return self.dataloader_train, self.dataloader_val

    def on_validation_epoch_end(self, val_outputs: list[dict]) -> None:
        if not val_outputs:
            raise ValueError("Phase 3 validation epoch has no patch results")
        loss = float(np.mean([row["loss"] for row in val_outputs]))
        if not np.isfinite(loss):
            raise ValueError("Phase 3 validation patch loss is nonfinite")
        counts = [np.sum(np.stack([np.asarray(row[key], dtype=np.float64)
                                   for row in val_outputs]), axis=0)
                  for key in ("tp_hard", "fp_hard", "fn_hard")]
        tp, fp, fn = counts
        denominator = 2 * tp + fp + fn
        per_class = np.full(denominator.shape, np.nan, dtype=np.float64)
        np.divide(2 * tp, denominator, out=per_class, where=denominator > 0)
        defined = denominator > 0
        mean_dice = float(np.mean(per_class[defined])) if np.any(defined) else float("nan")
        self.logger.log("mean_fg_dice", mean_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", per_class.tolist(), self.current_epoch)
        self.logger.log("val_losses", loss, self.current_epoch)
        if not np.any(defined):
            self.print_to_log_file("Phase 3 patch validation Dice is undefined: "
                                   "target and prediction are empty; use full-volume metrics",
                                   also_print_to_console=False)

    def on_train_start(self):
        torch.backends.cudnn.benchmark = False
        if not self.was_initialized:
            self.initialize()
        if (self.config.swin.enabled and self.config.swin.init == "ssl"
                and not self.continue_training and not self._restore_mode):
            self.prior_load_report = initialize_prior(self.network, prior_artifact_from_context())
        super().on_train_start()
        if self._pending_phase3_state is not None:
            state = self._pending_phase3_state
            self.dataloader_train.load_state_dict(state["train_loader_state"])
            self.dataloader_val.load_state_dict(state["val_loader_state"])
            restore_rng_state(state["rng"])
            self._pending_phase3_state = None
        if self._phase3_tensorboard is None:
            self._phase3_tensorboard = Phase3TensorBoard(
                Path(self.output_folder), purge_step=self.global_step + 1)
        torch.backends.cudnn.benchmark = False

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time.time(), self.current_epoch)
        collector = ValidationPreviewCollector(limit=3)
        summary = evaluate_validation_group_dice(self, on_case=collector.add)
        score = summary["mean"]["dice"]
        if score is None or not np.isfinite(score):
            raise ValueError("Full-volume validation group Dice is nonfinite")
        improved = self.best_group_dice is None or score > self.best_group_dice
        if improved:
            self.best_group_dice = score
            self.best_step = self.global_step
        self.print_to_log_file(f"Phase 3 validation group Dice: {score:.6f}; best: {self.best_group_dice:.6f}")
        if self._phase3_tensorboard is None:
            self._phase3_tensorboard = Phase3TensorBoard(
                Path(self.output_folder), purge_step=self.global_step + 1)
        training_scalars = {"train/learning_rate": float(self.optimizer.param_groups[0]["lr"])}
        for key, tag in (("train_losses", "train/loss"),
                         ("val_losses", "validation/patch_loss"),
                         ("mean_fg_dice", "validation/patch_dice")):
            try:
                value = self.logger.get_value(key, step=-1)
            except (KeyError, IndexError, TypeError):
                continue
            if value is not None and np.isscalar(value):
                training_scalars[tag] = float(value)
        report = self._phase3_tensorboard.log_epoch(
            self.current_epoch + 1, summary, collector.selected(),
            training_scalars=training_scalars, step=self.global_step)
        self.print_to_log_file(f"Phase 3 TensorBoard validation report: {report}")
        self.current_epoch += 1
        if improved:
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best.pth"))
        self.save_checkpoint(str(Path(self.output_folder) /
                                 f"checkpoint_epoch_{self.current_epoch:04d}.pth"))
        self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))

    def on_train_end(self):
        self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_final.pth"))
        self.print_to_log_file("Phase 3 training complete")
        if self._phase3_tensorboard is not None:
            self._phase3_tensorboard.close()

    def perform_actual_validation(self, save_probabilities: bool = False):
        summary = evaluate_validation_group_dice(self)
        self.logger.log_summary("final_val/group_dice", summary["mean"]["dice"])
        return summary

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0 and not self.disable_checkpointing:
            atomic_save_checkpoint(build_phase3_checkpoint(self), Path(filename))

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        if isinstance(filename_or_checkpoint, (str, Path)):
            path = Path(filename_or_checkpoint)
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location=self.device, weights_only=False)
        elif isinstance(filename_or_checkpoint, dict):
            payload = filename_or_checkpoint
        else:
            raise TypeError("Expected Phase 3 checkpoint path or payload")
        if not self.was_initialized:
            self.initialize()
        if payload.get("trainer_name") != self.__class__.__name__:
            raise ValueError("Checkpoint trainer class differs")
        restore_phase3_state(self, payload)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.grad_scaler is not None:
            if payload["grad_scaler_state"] is None:
                raise ValueError("CUDA checkpoint lacks gradient scaler state")
            self.grad_scaler.load_state_dict(payload["grad_scaler_state"])
        self.logger.load_checkpoint(payload["logging"])
        self._best_ema = payload["_best_ema"]
        self.inference_allowed_mirroring_axes = payload["inference_allowed_mirroring_axes"]
        self.my_init_kwargs = payload["init_args"]
        self._restore_mode = True

    def train_step(self, batch: dict) -> dict:
        start = time.perf_counter()
        data = batch["data"]
        if data.ndim != 5 or data.shape[1] != 2:
            raise ValueError("Training batch requires two aligned input views")
        data = data.to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([part.to(self.device, non_blocking=True) for part in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        parameters = [part for part in self.network.parameters() if part.requires_grad]
        amp_overflow_retries = 0
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True) if self.device.type == "cuda" else nullcontext():
                output = self.network(data)
                breakdown = self.loss.breakdown(output, target)
                total = breakdown.total
            if not torch.isfinite(total):
                raise ValueError("Nonfinite Phase 3 training loss")
            if self.grad_scaler is not None:
                self.grad_scaler.scale(total).backward()
                self.grad_scaler.unscale_(self.optimizer)
            else:
                total.backward()
            bad_parameters = [name for name, part in self.network.named_parameters()
                              if part.requires_grad and part.grad is not None
                              and not torch.isfinite(part.grad).all()]
            if bad_parameters:
                if self.grad_scaler is None or amp_overflow_retries >= 16:
                    raise ValueError("Nonfinite Phase 3 training gradient: "
                                     f"{bad_parameters[:8]}; AMP retries={amp_overflow_retries}")
                scale_before = self.grad_scaler.get_scale()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                scale_after = self.grad_scaler.get_scale()
                if not 0 < scale_after < scale_before:
                    raise ValueError("Phase 3 AMP overflow did not reduce the gradient scale")
                amp_overflow_retries += 1
                self.print_to_log_file("phase3_amp_overflow " + json.dumps({
                    "global_step": self.global_step + 1,
                    "keys": list(batch.get("keys", ())),
                    "retry": amp_overflow_retries,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "nonfinite_parameters": bad_parameters[:8],
                    "nonfinite_parameter_count": len(bad_parameters),
                }), also_print_to_console=False)
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 12)
            if not torch.isfinite(grad_norm):
                raise ValueError("Nonfinite Phase 3 gradient norm")
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            break
        if any(not torch.isfinite(part).all() for part in parameters):
            raise ValueError("Nonfinite Phase 3 parameter after optimizer step")
        self.global_step += 1
        telemetry = self.network.fusion_telemetry()
        peak_memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        diagnostics = {"global_step": self.global_step, "keys": list(batch.get("keys", ())),
                       "fusion": telemetry, "valid_rays": (breakdown.valid_rays.detach().cpu().tolist()
                                                                if breakdown.valid_rays is not None else None),
                       "aux_valid_samples": list(breakdown.valid_sample_counts),
                       "amp": self.grad_scaler is not None,
                       "amp_overflow_retries": amp_overflow_retries,
                       "amp_scale": self.grad_scaler.get_scale() if self.grad_scaler is not None else None,
                       "peak_memory_bytes": peak_memory,
                       "seconds": time.perf_counter() - start}
        self.print_to_log_file("phase3_step " + json.dumps(diagnostics),
                               also_print_to_console=False)
        if self._phase3_tensorboard is not None:
            self._phase3_tensorboard.log_step(self.global_step, {
                "train/step/total_loss": float(total.detach()),
                "train/step/segmentation_loss": float(breakdown.segmentation.detach()),
                "train/step/dice_loss": float(breakdown.dice.detach()),
                "train/step/bce_loss": float(breakdown.bce.detach()),
                "train/step/projection_loss": float(breakdown.projection.detach()),
                "train/step/gradient_norm": float(grad_norm.detach()),
                "train/step/amp_overflow_retries": amp_overflow_retries,
                "train/step/amp_scale": self.grad_scaler.get_scale()
                if self.grad_scaler is not None else None,
            })
        return {"loss": float(total.detach()), "segmentation": float(breakdown.segmentation.detach()),
                "dice": float(breakdown.dice.detach()), "bce": float(breakdown.bce.detach()),
                "projection": float(breakdown.projection.detach()),
                "grad_norm": float(grad_norm.detach()),
                "amp_overflow_retries": amp_overflow_retries,
                "global_step": self.global_step,
                "keys": list(batch.get("keys", ())), "seconds": time.perf_counter() - start}

    @torch.no_grad()
    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device)
        target = batch["target"]
        target = ([part.to(self.device) for part in target]
                  if isinstance(target, list) else target.to(self.device))
        with torch.autocast("cuda", enabled=True) if self.device.type == "cuda" else nullcontext():
            output = self.network(data)
            breakdown = self.loss.breakdown(output, target)
        final = output[0] if isinstance(output, list) else output
        truth = target[0] if isinstance(target, list) else target
        if truth.ndim == 4:
            truth = truth[:, None]
        valid = truth != 2
        predicted = final.argmax(dim=1, keepdim=True) == 1
        foreground = truth == 1
        tp = (predicted & foreground & valid).sum().cpu().numpy()[None]
        fp = (predicted & ~foreground & valid).sum().cpu().numpy()[None]
        fn = (~predicted & foreground & valid).sum().cpu().numpy()[None]
        return {"loss": float(breakdown.total.detach()), "tp_hard": tp,
                "fp_hard": fp, "fn_hard": fn}
