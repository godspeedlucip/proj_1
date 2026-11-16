from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from stent_repro.networks import UNet
from stent_repro.preflight import benchmark_stage, preflight, resource_contract
from stent_repro.training.engine import train_prepared


@pytest.mark.parametrize("shape,tile,overlap", [
    ((2, 3, 5, 15), (8, 8), (2, 2)),
    ((1, 2, 1, 11, 15), (4, 8, 8), (0, 2, 2)),
])
def test_tiled_identity_preserves_boundaries(shape, tile, overlap):
    from stent_repro.runtime import tiled_predict

    x = torch.randn(shape, generator=torch.Generator().manual_seed(31))
    result, coverage = tiled_predict(nn.Identity().eval(), x, tile, overlap, 2)

    torch.testing.assert_close(result, x)
    assert coverage.shape == x.shape[2:]
    assert coverage.min() == 1
    assert coverage.max() > 1
    assert coverage.reshape(-1)[-1] >= 1


@pytest.mark.parametrize("tile,overlap,stride", [
    ((5, 8), (2, 2), 2),
    ((8, 8), (1, 2), 2),
    ((8, 8), (8, 0), 2),
    ((8, 8), (-2, 0), 2),
    ((8, 8), 2, 0),
    ((8,), 0, 2),
    ((8, 8), (2, True), 2),
])
def test_tiled_prediction_rejects_invalid_grid(tile, overlap, stride):
    from stent_repro.runtime import tiled_predict

    with pytest.raises(ValueError):
        tiled_predict(nn.Identity().eval(), torch.zeros(1, 1, 7, 7), tile, overlap, stride)


def test_tiled_prediction_requires_eval_mode():
    from stent_repro.runtime import tiled_predict

    with pytest.raises(ValueError, match="eval"):
        tiled_predict(nn.Identity(), torch.zeros(1, 1, 8, 8), (4, 4), 0, 2)


def test_benchmark_records_non_identical_tiled_context():
    from stent_repro.runtime import benchmark_forward

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(1)
    report = benchmark_forward(
        model, torch.ones(1, 1, 9, 7), stride=2, tile_shape=(6, 6),
        overlap=2, warmup=1, repeats=2,
    )

    for path in ("whole", "tiled"):
        timing = report["paths"][path]
        assert timing["status"] == "ok"
        assert len(timing["forward_seconds"]) == 2
        assert timing["median_forward_seconds"] > 0
        assert timing["median_host_to_device_seconds"] >= 0
        assert timing["median_device_to_host_seconds"] >= 0
    assert report["disagreement"]["max_absolute_error"] > 0
    assert report["paths"]["tiled"]["coverage_min"] == 1
    assert report["paths"]["tiled"]["coverage_max"] > 1


def test_checkpoint_recomputation_preserves_batchnorm_and_dropout_gradients():
    torch.manual_seed(17)
    baseline = UNet(2, 1, (2, 4), dropout=0.2)
    checkpointed = deepcopy(baseline)
    x = torch.randn(2, 1, 8, 8)

    torch.manual_seed(19)
    expected = baseline(x)
    expected.square().mean().backward()
    torch.manual_seed(19)
    actual = checkpointed(x, activation_checkpointing=True)
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected)
    for initial, recomputed in zip(baseline.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(recomputed.grad, initial.grad)
    for initial, recomputed in zip(baseline.buffers(), checkpointed.buffers()):
        torch.testing.assert_close(recomputed, initial)
    counters = [
        tensor for name, tensor in checkpointed.named_buffers()
        if name.endswith("num_batches_tracked")
    ]
    assert counters and all(counter.item() == 1 for counter in counters)


def _train_kwargs(device):
    return dict(
        method="B", stage="dte", batch_size=2, learning_rate=0.001,
        seed=11, device=device, normalization_mode="none",
    )


def _tiny_training_data():
    x = torch.linspace(0, 1, 4 * 8 * 8).reshape(4, 1, 8, 8)
    return TensorDataset(x, x.square())


@pytest.mark.parametrize("precision,device", [
    ("amp_bfloat16", "cpu"),
    pytest.param("amp_float16", "cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required for float16 AMP",
    )),
])
def test_checkpoint_resume_replays_model_and_scaler(tmp_path, precision, device):
    torch.manual_seed(17)
    model = UNet(2, 1, (2, 4), dropout=0.2)
    ds = _tiny_training_data()
    kwargs = dict(_train_kwargs(device), precision=precision, activation_checkpointing=True)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    train_prepared(ds, ds, deepcopy(model), full, epochs=2, **kwargs)
    train_prepared(ds, ds, deepcopy(model), resumed, epochs=1, **kwargs)
    train_prepared(
        ds, ds, deepcopy(model), resumed, epochs=2, resume=resumed / "last.pt", **kwargs,
    )
    expected = torch.load(full / "last.pt", weights_only=True, map_location="cpu")
    actual = torch.load(resumed / "last.pt", weights_only=True, map_location="cpu")

    assert actual["contract"]["precision"] == precision
    assert actual["contract"]["activation_checkpointing"] is True
    assert actual["scaler"] == expected["scaler"]
    if device == "cuda":
        assert actual["scaler"]["_growth_tracker"] == 4
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, atol=0, rtol=0)
    for kwargs_change in ({"precision": "float32"}, {"activation_checkpointing": False}):
        with pytest.raises(ValueError, match="contract"):
            train_prepared(
                ds, ds, deepcopy(model), resumed, epochs=3, resume=resumed / "last.pt",
                **(kwargs | kwargs_change),
            )


def test_legacy_training_checkpoint_resumes_with_float32_defaults(tmp_path, workflow_config):
    ds = _tiny_training_data()
    config = workflow_config("B")
    kwargs = dict(_train_kwargs("cpu"), run_info={"resolved_config": config})
    model = nn.Conv2d(1, 1, 1)
    train_prepared(ds, ds, model, tmp_path, epochs=1, **kwargs)
    legacy_path = tmp_path / "legacy.pt"
    legacy = torch.load(tmp_path / "last.pt", weights_only=True)
    assert legacy["contract"].pop("precision") == "float32"
    assert legacy["contract"].pop("activation_checkpointing") is False
    del legacy["scaler"]
    legacy_config = legacy["contract"]["run_info"]["resolved_config"]
    for stage in legacy_config["training"].values():
        del stage["precision"], stage["activation_checkpointing"]
    del legacy_config["inference"]["tiling"]
    torch.save(legacy, legacy_path)

    config["training"]["dte"]["epochs"] = 2
    train_prepared(ds, ds, model, tmp_path, epochs=2, resume=legacy_path, **kwargs)
    assert torch.load(tmp_path / "last.pt", weights_only=True)["epoch"] == 2


@pytest.mark.parametrize("option", [
    {"precision": "half"},
    {"precision": "amp_float16"},
    {"activation_checkpointing": 1},
])
def test_training_rejects_unsupported_runtime_options(tmp_path, option):
    ds = _tiny_training_data()
    with pytest.raises(ValueError):
        train_prepared(ds, ds, nn.Conv2d(1, 1, 1), tmp_path, epochs=1,
                       **_train_kwargs("cpu"), **option)


def test_resource_contract_binds_training_precision_and_checkpointing(workflow_config):
    config = workflow_config("B")
    config["training"]["dtr"].update(precision="amp_bfloat16", activation_checkpointing=True)
    contract = resource_contract(config, "dtr", "cpu")

    assert contract["precision"] == "amp_bfloat16"
    assert contract["activation_checkpointing"] is True


def test_preflight_applies_checkpointed_bfloat16_dice_step(workflow_config):
    config = workflow_config("B")
    config["training"]["dtr"].update(precision="amp_bfloat16", activation_checkpointing=True)
    result = preflight(config, stage="dtr", measure=True)

    assert result["resource_contract"]["loss"] == "soft_dice_B"
    assert result["optimizer_step_applied"] is True
    assert result["full_step_measured"] is True
    assert 0 < result["loss_value"] < 1


def test_stage_benchmark_uses_declared_network_batch(workflow_config):
    report = benchmark_stage(workflow_config("B"), stage="dtr", warmup=1, repeats=1)

    assert report["input_shape"] == [2, 3, 16, 16, 16]
    assert report["stride"] == 4
    assert report["paths"]["whole"]["status"] == "ok"
    assert report["paths"]["whole"]["peak_cuda_bytes"] is None
    assert report["disagreement"] is None


def test_preflight_identifies_invalid_training_shape(workflow_config):
    config = workflow_config("B")
    config["training"]["dtr"]["patch_shape"] = [9, 8, 8]
    result = preflight(config, stage="dtr", measure=True)

    assert result["full_step_measured"] is False
    assert result["measurement_phase"] == "validating_runtime"
    assert "stride" in result["measurement_error"]


def test_explicit_float32_training_ignores_outer_autocast(tmp_path):
    class TypedConv(nn.Conv2d):
        def forward(self, x):
            result = super().forward(x)
            assert result.dtype == torch.float32
            return result

    ds = _tiny_training_data()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        train_prepared(ds, ds, TypedConv(1, 1, 1), tmp_path, epochs=1, **_train_kwargs("cpu"))
