import os

import pytest
import torch

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.initialization import capture_rng_state, named_rng, restore_rng_state
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.network import build_phase3_network
from nnunetv2.phase3.planning import make_phase3_plans
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from .fixtures import make_config, synthetic_dataset


def _baseline_config():
    return resolve_config(make_config(
        run_kind="smoke", cnn__patch_size=[32, 32, 32], cnn__batch_size=1,
        training__max_optimizer_steps=4, training__updates_per_epoch=2,
        swin__enabled=False, swin__execute_through="layers2",
        fusion__enabled=False, fusion__bottleneck_enabled=False,
        tubular__enabled=False, projection__enabled=False, projection__weight=0.0))


@pytest.fixture
def plans_pair(tmp_path):
    mp, sp = synthetic_dataset(tmp_path / "input")
    prepared = prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                                source_fold=0, dataset_id=904, output=tmp_path / "fold")
    plans = make_phase3_plans(prepared, _baseline_config())
    manager = PlansManager(plans)
    return manager, manager.get_configuration("3d_fullres")


def test_cpu_cnn_uses_one_view_and_native_deep_supervision(plans_pair):
    torch.set_num_threads(2)
    manager, configuration = plans_pair
    network = build_phase3_network(manager, configuration, 2, 2, enable_deep_supervision=True)
    assert next(module for module in network.encoder.modules() if isinstance(module, torch.nn.Conv3d)).in_channels == 1
    assert network.decoder.deep_supervision
    data = torch.randn((1, 2, 32, 32, 32), requires_grad=True)
    outputs = network(data)
    assert isinstance(outputs, list)
    assert [tuple(t.shape) for t in outputs] == [
        (1, 2, 32, 32, 32), (1, 2, 16, 16, 16),
        (1, 2, 8, 8, 8), (1, 2, 4, 4, 4)]
    torch.stack([t.square().mean() for t in outputs]).sum().backward()
    assert data.grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(data.grad[:, 1]) == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
    network.eval()
    with torch.no_grad():
        changed = data.detach().clone()
        changed[:, 1] += 1000
        assert all(torch.equal(a, b) for a, b in zip(network(data.detach()), network(changed)))
        network.set_deep_supervision(False)
        assert isinstance(network(data.detach()), torch.Tensor)


def test_adapter_matches_native_plainconvunet_with_identical_state(plans_pair):
    torch.set_num_threads(2)
    manager, configuration = plans_pair
    adapter = build_phase3_network(manager, configuration, 2, 2, enable_deep_supervision=True).eval()
    native = get_network_from_plans(configuration.network_arch_class_name,
                                    configuration.network_arch_init_kwargs,
                                    configuration.network_arch_init_kwargs_req_import,
                                    input_channels=1, output_channels=2,
                                    allow_init=False, deep_supervision=True).eval()
    assert set(adapter.state_dict()) == set(native.state_dict())
    native.load_state_dict(adapter.state_dict(), strict=True)
    x = torch.randn((1, 2, 32, 32, 32))
    with torch.no_grad():
        expected = native(x[:, :1])
        actual = adapter(x)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(actual, expected))
    assert [tuple(t.shape[1:]) for t in adapter.encode_cnn(x)] == [
        (32, 32, 32, 32), (64, 16, 16, 16), (128, 8, 8, 8),
        (256, 4, 4, 4), (320, 2, 2, 2)]


def test_named_rng_replays_without_advancing_ambient_stream():
    before = capture_rng_state()
    with named_rng(2026, "cnn"):
        first = torch.rand(5)
    after = capture_rng_state()
    assert torch.equal(before["torch"], after["torch"])
    with named_rng(2026, "cnn"):
        assert torch.equal(first, torch.rand(5))
    restore_rng_state(before)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("PHASE3_GPU_TESTS") != "1", reason="GPU capacity test is opt-in")
def test_formal_128_cube_backward_on_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    mp, sp = synthetic_dataset(tmp_path / "input")
    prepared = prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                                source_fold=0, dataset_id=905, output=tmp_path / "fold")
    cfg = resolve_config(make_config(swin__enabled=False, swin__execute_through="layers2",
                                     fusion__enabled=False, fusion__bottleneck_enabled=False,
                                     tubular__enabled=False, projection__enabled=False,
                                     projection__weight=0.0))
    manager = PlansManager(make_phase3_plans(prepared, cfg))
    network = build_phase3_network(manager, manager.get_configuration("3d_fullres"), 2, 2).cuda()
    x = torch.randn((1, 2, 128, 128, 128), device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = network(x)
        loss = sum(v.float().square().mean() for v in y)
    loss.backward()
    assert torch.isfinite(x.grad).all()
