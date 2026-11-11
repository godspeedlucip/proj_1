"""Formal STP 128^3 / batch 2 AMP capacity and native sliding-window check."""

import json
import os
from pathlib import Path
import time

import pytest
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from nnunetv2.phase3.cli import load_preset
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.inference import predict_volume
from nnunetv2.phase3.losses import Phase3CompositeLoss
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.network import build_phase3_network
from nnunetv2.phase3.nifti_io import VolumeCase, read_nifti
from nnunetv2.phase3.planning import make_phase3_plans
from nnunetv2.phase3.prior_weights import initialize_prior, prepare_prior_artifact
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from .fixtures import synthetic_dataset


ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "tmp" / "phase3_nnunet" / "gpu_capacity.json"


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("PHASE3_GPU_TESTS") != "1", reason="Formal GPU capacity test is opt-in")
def test_formal_stp_amp_step_and_native_sliding_prediction(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    weights = Path(os.getenv("PHASE3_SSL_WEIGHTS", str(ROOT / "tmp/fdk_v3/official_ssl/model_swinvit.pt")))
    if not weights.is_file():
        pytest.skip("Official SSL weights unavailable")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    free_before, total_bytes = torch.cuda.mem_get_info()
    report = {"status": "running", "device": torch.cuda.get_device_name(0),
              "total_bytes": total_bytes, "free_before_bytes": free_before,
              "torch": torch.__version__, "configuration": "full",
              "patch_size": [128, 128, 128], "batch_size": 2, "amp": "float16"}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    try:
        mp, sp = synthetic_dataset(tmp_path / "source")
        manifest = load_manifest(mp, require_labels=True)
        prepared = prepare_raw_fold(manifest, load_splits(sp), source_fold=0,
                                    dataset_id=915, output=tmp_path / "fold")
        config = load_preset("full")
        manager = PlansManager(make_phase3_plans(prepared, config))
        network = build_phase3_network(manager, manager.get_configuration("3d_fullres"), 2, 2).cuda()
        artifact = prepare_prior_artifact(weights, tmp_path / "prior_artifact.json")
        audit = initialize_prior(network, artifact)
        assert audit.loaded_parameter_fraction == 1.0
        report["ssl_sha256"] = audit.sha256
        report["parameter_count"] = sum(part.numel() for part in network.parameters())
        report["trainable_parameter_count"] = sum(part.numel() for part in network.parameters() if part.requires_grad)
        optimizer = torch.optim.SGD((part for part in network.parameters() if part.requires_grad),
                                    lr=.01, momentum=.99, weight_decay=3e-5, nesterov=True)
        scaler = torch.GradScaler("cuda")
        loss_module = Phase3CompositeLoss(config)
        data = torch.cat((torch.randn((2, 1, 128, 128, 128), device="cuda"),
                          torch.rand((2, 1, 128, 128, 128), device="cuda")), dim=1)
        target = torch.zeros((2, 1, 128, 128, 128), dtype=torch.long, device="cuda")
        target[:, :, 20:110, 61:63, 63:65] = 1
        targets = [F.interpolate(target.float(), size=(n, n, n), mode="nearest").long()
                   for n in (128, 64, 32, 16)]
        network.train()
        assert not network.prior.training
        first_conv = next(layer for layer in network.encoder.modules() if isinstance(layer, torch.nn.Conv3d))
        before = first_conv.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = network(data)
            breakdown = loss_module.breakdown(outputs, targets)
        assert torch.isfinite(breakdown.total)
        scaler.scale(breakdown.total).backward()
        scaler.unscale_(optimizer)
        trainable = [part for part in network.parameters() if part.requires_grad]
        assert all(part.grad is None or torch.isfinite(part.grad).all() for part in trainable)
        torch.nn.utils.clip_grad_norm_(trainable, 12)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        report["step_seconds"] = time.perf_counter() - started
        report["step_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["step_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["total_loss"] = float(breakdown.total.detach())
        report["valid_ray_counts"] = breakdown.valid_rays.detach().cpu().tolist()
        assert not torch.equal(before, first_conv.weight)
        assert all(part.grad is None for part in network.prior.parameters())
        del data, outputs, breakdown, target, targets
        torch.cuda.empty_cache()
        raw_path = tmp_path / "gpu_external.nii.gz"
        nii = nib.Nifti1Image(np.zeros((65, 97, 130), dtype=np.float32),
                              np.diag([.2, .2, .2, 1.]))
        nii.header.set_xyzt_units("mm")
        nib.save(nii, raw_path)
        raw, meta = read_nifti(raw_path)
        volume = VolumeCase("gpu_external", "external", raw, None, meta)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = predict_volume(network, volume, plans=manager.plans, config=config,
                                    stats=prepared.stats, device="cuda")
        torch.cuda.synchronize()
        report["sliding_seconds"] = time.perf_counter() - started
        report["sliding_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["sliding_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["sliding_original_shape_dhw"] = list(raw.shape)
        assert prediction.probability.shape == volume.image_raw.shape
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
