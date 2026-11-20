"""Native nnU-Net epoch loop for full-detector joint segmentation and depth."""
import hashlib
import json
import math
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from time import time

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import save_json
from torch.utils.tensorboard import SummaryWriter

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.checkpoints import (
    load_checkpoint_file, load_network_weights, safetensors_filename, save_checkpoint_file,
    without_training_state,
)
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def depth_case_counts(prediction, estimate, mask, depth, valid):
    """Sufficient statistics for one complete detector image, including missed GT."""
    if prediction.shape[0] != 1:
        raise ValueError("Depth selection requires one complete image per batch")
    valid = valid.bool()
    fg, bg = mask.bool() & valid, ~mask.bool() & valid
    pred = prediction.bool() & valid
    estimate, depth = estimate.float(), depth.float()
    error = (estimate - depth).abs()
    missed, outside = fg & ~pred, valid & ~pred
    fields = {
        "fg": fg.sum(), "bg": bg.sum(), "tp": (fg & pred).sum(),
        "fp": (bg & pred).sum(), "fn": missed.sum(),
        "fg_abs": (error * fg).sum(), "fg_sq": (error.square() * fg).sum(),
        "missed_abs": (error * missed).sum(), "bg_abs": (estimate.abs() * bg).sum(),
        "outside_nonzero": ((estimate != 0) & outside).sum(),
        "outside_abs": (estimate.abs() * outside).sum(), "empty": ~pred.any(),
    }
    return dict(zip(fields, torch.stack([v.double() for v in fields.values()]).tolist()))


def summarize_depth(cases, target_scale):
    def total(key):
        return sum(case[key] for case in cases)
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")
    fg_cases = [row for row in cases if row["fg"]]
    foreground = total("fg")
    mae = ratio(sum(row["fg_abs"] / row["fg"] for row in fg_cases), len(fg_cases))
    weighted = ratio(total("fg_abs"), foreground)
    rmse = math.sqrt(ratio(total("fg_sq"), foreground))
    tp, fp, fn = total("tp"), total("fp"), total("fn")
    return {
        "fg_mae_macro": mae, "fg_mae_pixel": weighted,
        "fg_mae_raw_macro": target_scale * mae, "fg_mae_raw_pixel": target_scale * weighted,
        "fg_rmse_pixel": rmse, "fg_rmse_raw_pixel": target_scale * rmse,
        "background_mae_pixel": ratio(total("bg_abs"), total("bg")),
        "background_sum": total("bg_abs"), "background_raw_sum": target_scale * total("bg_abs"),
        "missed_fg_mae_contribution": ratio(total("missed_abs"), foreground),
        "mask_dice": ratio(2 * tp, 2 * tp + fp + fn),
        "mask_precision": ratio(tp, tp + fp), "mask_recall": ratio(tp, tp + fn),
        "empty_prediction_fraction": total("empty") / len(cases),
        "outside_prediction_nonzero": total("outside_nonzero"), "outside_prediction_abs": total("outside_abs"),
        "images": len(cases), "images_with_foreground": len(fg_cases),
    }


class nnUNetTrainerMarkerGuidedDepth(nnUNetTrainer):
    smoke = False
    marker_enabled = True
    accumulation_steps = 2

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.is_ddp or self.configuration_manager.batch_size != 1:
            raise ValueError("The full-detector depth plans use a single process and batch size 1")
        self.joint = plans["joint_depth"]
        self.seed = int(self.joint["seed"])
        self.target_scale = float(self.joint["target_scale"])
        if not math.isfinite(self.target_scale) or self.target_scale <= 0:
            raise ValueError("Training target_scale must be positive and finite")
        self.num_epochs = 2 if self.smoke else 100
        self.num_iterations_per_epoch = 4 if self.smoke else 250
        self.initial_lr, self.depth_lr, self.weight_decay = 1e-5, 1e-4, 1e-4
        self.save_every = 1
        self.best_fg_mae = float("inf")
        self.optimizer_updates = self.microbatch = self.val_batch = 0
        self.restored = False
        self.tb = None
        self.contract = None

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        from .network import build_joint_network
        return build_joint_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision)

    def initialize(self):
        # Native initialization assumes stock .npz/.b2nd caches. Our loader uses paired TIFFs.
        if self.was_initialized:
            raise RuntimeError("Trainer has already been initialized")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.batch_size = 1
        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json)
        self.network = self.build_network_architecture(
            self.plans_manager, self.configuration_manager, self.num_input_channels,
            self.label_manager.num_segmentation_heads, self.enable_deep_supervision).to(self.device)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.grad_scaler is not None:
            # Full-resolution Dice gradients pass through sparse early convolution activations.
            # Start conservatively to avoid wasting the first accumulation window finding a scale.
            self.grad_scaler = torch.cuda.amp.GradScaler(init_scale=128.0)
        self.was_initialized = True

    def configure_optimizers(self):
        groups = [
            {"params": self.network.backbone.parameters(), "name": "backbone", "lr": self.initial_lr},
            {"params": self.network.depth_refiner.parameters(), "name": "depth_refiner", "lr": self.depth_lr},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def _contract(self):
        if self.contract is None:
            manifest = Path(self.joint["manifest"])
            manifest_hash = file_sha256(manifest)
            if self.joint.get("manifest_sha256", manifest_hash) != manifest_hash:
                raise ValueError(f"Training manifest changed since preparation: {manifest}")
            plans_bytes = json.dumps(self.plans_manager.plans, sort_keys=True, separators=(",", ":")).encode()
            self.contract = {
                "plans_sha256": hashlib.sha256(plans_bytes).hexdigest(),
                "manifest_sha256": manifest_hash,
                "seed": self.seed, "target_scale": self.target_scale, "marker_enabled": self.marker_enabled,
                "smoke": self.smoke, "gate_mode": "hard_mask", "mask_threshold": 0.5,
                "training_stage": "joint", "input_channels": ["image", "point", "band", "validity"],
                "depth_input_channels": ["image*mask", "mask", "probability*mask"],
                "preprocessing": "whole-frame image z-score; right/bottom stride padding; aligned flips",
            }
        return self.contract

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        if not self.restored:
            path = str(self.joint["init_seg_checkpoint"])
            self.print_to_log_file(f"Loading initial E5-U backbone from {path}")
            if "init_seg_sha256" in self.joint and file_sha256(path) != self.joint["init_seg_sha256"]:
                raise ValueError(f"Initial E5-U checkpoint changed since preparation: {path}")
            self.network.backbone.load_state_dict(load_network_weights(path, "cpu"), strict=True)
        self.set_deep_supervision_enabled(True)
        self._contract()
        model_dir = Path(self.output_folder_base)
        save_json(self.plans_manager.plans, str(model_dir / "plans.json"), sort_keys=False)
        save_json(self.dataset_json, str(model_dir / "dataset.json"), sort_keys=False)
        fingerprint = Path(self.preprocessed_dataset_folder_base) / "dataset_fingerprint.json"
        if fingerprint.is_file():
            shutil.copyfile(fingerprint, model_dir / fingerprint.name)
        self.print_plans()
        self.tb = SummaryWriter(str(Path(self.output_folder) / "tensorboard"),
                                purge_step=self.current_epoch + 1 if self.restored else None)
        self.print_to_log_file(f"Joint AdamW, accumulation {self.accumulation_steps}, "
                               f"marker={self.marker_enabled}, seed={self.seed}, "
                               f"start epoch={self.current_epoch + 1}")

    def get_dataloaders(self):
        from .data import make_loaders
        train, val, n_val = make_loaders(self.joint["manifest"], self.seed, self.smoke,
                                         self.current_epoch, self.marker_enabled)
        if n_val < 1:
            raise ValueError("Joint validation split is empty")
        self.num_val_iterations_per_epoch = n_val
        return train, val

    def on_train_epoch_start(self):
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        self.network.train()
        self.microbatch = self.val_batch = 0
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)
        self.print_to_log_file(f"Epoch {self.current_epoch + 1}/{self.num_epochs}, "
                               f"{self.num_iterations_per_epoch} training and "
                               f"{self.num_val_iterations_per_epoch} validation images")

    def _forward_loss(self, batch):
        from .network import joint_loss
        inputs, mask, depth, valid = [batch[key].to(self.device, non_blocking=True)
                                      for key in ("data", "target", "depth_target", "valid_pixels")]
        with torch.autocast("cuda") if self.device.type == "cuda" else nullcontext():
            outputs = self.network(inputs, return_depth=True)
        loss, parts = joint_loss(outputs, mask, depth, valid)
        return loss, parts, outputs, mask, depth, valid

    @staticmethod
    def _loss_values(loss, parts):
        values = {"loss": loss.detach(), **{k: v.detach() for k, v in parts.items()}}
        return dict(zip(values, torch.stack([v.float() for v in values.values()]).tolist()))

    def train_step(self, batch):
        window_start = (self.microbatch // self.accumulation_steps) * self.accumulation_steps
        window_length = min(self.accumulation_steps, self.num_iterations_per_epoch - window_start)
        if self.microbatch == window_start:
            self.optimizer.zero_grad(set_to_none=True)
        loss, parts, *_ = self._forward_loss(batch)
        values = self._loss_values(loss, parts)
        if not math.isfinite(values["loss"]):
            raise FloatingPointError(f"Non-finite training loss for {batch['keys']}")
        scaled_loss = loss / window_length
        if self.grad_scaler is None:
            scaled_loss.backward()
        else:
            self.grad_scaler.scale(scaled_loss).backward()
        self.microbatch += 1
        if self.microbatch == window_start + window_length:
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
            first_grad_norms = {}
            if not self.optimizer_updates:
                for group in self.optimizer.param_groups:
                    grads = [p.grad.detach().norm() for p in group["params"] if p.grad is not None]
                    first_grad_norms[group["name"]] = torch.stack(grads).norm().item() if grads else 0.0
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if self.grad_scaler is None:
                self.optimizer.step()
                self.optimizer_updates += 1
            else:
                old_scale = self.grad_scaler.get_scale()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if self.grad_scaler.get_scale() >= old_scale:
                    self.optimizer_updates += 1
                else:
                    self.print_to_log_file("AMP skipped optimizer update due to overflow")
            if first_grad_norms and self.optimizer_updates == 1:
                self.print_to_log_file(f"First optimizer update, unscaled gradient norms {first_grad_norms}")
        if self.microbatch <= 2 or self.microbatch % 25 == 0:
            self.print_to_log_file(f"train {self.microbatch}/{self.num_iterations_per_epoch} "
                                   f"loss={values['loss']:.6f} updates={self.optimizer_updates}")
        return values

    def validation_step(self, batch):
        loss, parts, outputs, mask, depth, valid = self._forward_loss(batch)
        values = self._loss_values(loss, parts)
        if not math.isfinite(values["loss"]):
            raise FloatingPointError(f"Non-finite validation loss for {batch['keys']}")
        values["case"] = depth_case_counts(outputs["predicted_mask"], outputs["depth_normalized"],
                                           mask, depth, valid)
        self.val_batch += 1
        if self.val_batch == 1 or self.val_batch % 100 == 0:
            self.print_to_log_file(f"validation {self.val_batch}/{self.num_val_iterations_per_epoch}")
        return values

    def _log_epoch_scalars(self, split, metrics):
        # Extend the native logger's epoch histories, which are also saved with each checkpoint.
        for key, value in metrics.items():
            name = f"{split}/{key}"
            self.logger.local_logger.my_fantastic_logging.setdefault(name, [])
            self.logger.log(name, value, self.current_epoch)
            if self.tb is not None:
                self.tb.add_scalar(name, value, self.current_epoch + 1)

    def on_train_epoch_end(self, train_outputs):
        means = {k: float(np.mean([v[k] for v in train_outputs])) for k in train_outputs[0]}
        self.logger.log("train_losses", means["loss"], self.current_epoch)
        self._log_epoch_scalars("train", {**means, "optimizer_updates": self.optimizer_updates,
                                          "lr_backbone": self.initial_lr, "lr_depth": self.depth_lr})
        self.lr_scheduler.step()

    def on_validation_epoch_end(self, val_outputs):
        metrics = summarize_depth([v["case"] for v in val_outputs], self.target_scale)
        metrics.update({k: float(np.mean([v[k] for v in val_outputs]))
                        for k in val_outputs[0] if k != "case"})
        if metrics["outside_prediction_nonzero"] or not math.isfinite(metrics["fg_mae_macro"]):
            raise ValueError(f"Invalid hard-mask validation, statistics {metrics}")
        self.logger.log("val_losses", metrics["loss"], self.current_epoch)
        self.logger.log("mean_fg_dice", metrics["mask_dice"], self.current_epoch)
        self.logger.log("dice_per_class_or_region", [metrics["mask_dice"]], self.current_epoch)
        self._log_epoch_scalars("valid", metrics)

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        mae = self.logger.get_value("valid/fg_mae_macro", -1)
        self.print_to_log_file(f"Epoch {self.current_epoch + 1}: "
                               f"train_loss={self.logger.get_value('train_losses', -1):.6f} "
                               f"val_loss={self.logger.get_value('val_losses', -1):.6f} "
                               f"fg_mae_macro={mae:.6f} dice={self.logger.get_value('mean_fg_dice', -1):.6f}")
        if self.tb is not None:
            self.tb.flush()
        self.logger.plot_progress_png(self.output_folder)
        if mae < self.best_fg_mae:
            self.best_fg_mae = mae
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best.safetensors"))
        exit_now = self._exit_signal_received()
        if self.current_epoch + 1 < self.num_epochs or exit_now:
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.safetensors"))
        self.current_epoch += 1
        if exit_now:
            self.print_to_log_file("Saved latest checkpoint at the requested epoch boundary")
            if self.tb is not None:
                self.tb.close()
            raise SystemExit(0)

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:
            return
        filename = safetensors_filename(str(filename))
        checkpoint = {
            "network_weights": self.network.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1, "init_args": self.my_init_kwargs,
            "trainer_name": type(self).__name__, "inference_allowed_mirroring_axes": None,
            "joint_depth": {**self._contract(), "best_fg_mae": self.best_fg_mae,
                           "optimizer_updates": self.optimizer_updates},
        }
        if Path(filename).name == "checkpoint_final.safetensors":
            checkpoint = without_training_state(checkpoint)
        else:
            checkpoint["rng"] = {"python": random.getstate(), "numpy": np.random.get_state(),
                                 "torch_cpu": torch.get_rng_state(),
                                 "torch_cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        save_checkpoint_file(checkpoint, filename)
        self.print_to_log_file(f"Saved {Path(filename).name} after epoch {self.current_epoch + 1}")

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = load_checkpoint_file(str(checkpoint), map_location="cpu")
        extra = checkpoint["joint_depth"]
        for key, value in self._contract().items():
            if extra.get(key) != value:
                raise ValueError(f"Incompatible joint checkpoint field {key}")
        if not self.was_initialized:
            self.initialize()
        # Optimizer.load_state_dict places each CPU state on its parameter device.
        super().load_checkpoint(checkpoint)
        self.best_fg_mae = extra["best_fg_mae"]
        self.optimizer_updates = extra["optimizer_updates"]
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "rng" in checkpoint:
            rng = checkpoint["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"].cpu())
            if self.device.type == "cuda" and rng["torch_cuda"] is not None:
                torch.cuda.set_rng_state(rng["torch_cuda"].cpu(), self.device)
        self.restored = True

    def on_train_end(self):
        try:
            super().on_train_end()
        finally:
            if self.tb is not None:
                self.tb.close()

    def perform_actual_validation(self, save_probabilities=False):
        from .inference import export_fold
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        try:
            with torch.inference_mode():
                return export_fold(self, save_probabilities)
        finally:
            self.set_deep_supervision_enabled(self.enable_deep_supervision)


class nnUNetTrainerMarkerGuidedDepthSmoke(nnUNetTrainerMarkerGuidedDepth):
    smoke = True


class nnUNetTrainerMarkerGuidedDepthNoMarker(nnUNetTrainerMarkerGuidedDepth):
    marker_enabled = False


class nnUNetTrainerMarkerGuidedDepthNoMarkerSmoke(nnUNetTrainerMarkerGuidedDepthNoMarker):
    smoke = True
