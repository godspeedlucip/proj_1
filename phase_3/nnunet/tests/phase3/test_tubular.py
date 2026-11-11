import torch

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.network import build_phase3_network
from nnunetv2.phase3.planning import make_phase3_plans
from nnunetv2.phase3.tubular import TubularEnhancement
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from .fixtures import make_config, synthetic_dataset


def _network(tmp_path, tubular):
    mp, sp = synthetic_dataset(tmp_path / "input")
    prepared = prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                                source_fold=0, dataset_id=909, output=tmp_path / "fold")
    config = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                        cnn__batch_size=1, training__max_optimizer_steps=4,
                                        training__updates_per_epoch=2, swin__enabled=False,
                                        swin__execute_through="layers2", fusion__enabled=False,
                                        fusion__bottleneck_enabled=False, tubular__enabled=tubular,
                                        projection__enabled=False, projection__weight=0.0))
    manager = PlansManager(make_phase3_plans(prepared, config))
    return build_phase3_network(manager, manager.get_configuration("3d_fullres"), 2, 2)


def test_tubular_refines_only_final_head_and_keeps_auxiliary_heads(tmp_path):
    torch.set_num_threads(2)
    with_t = _network(tmp_path / "with", True).eval()
    without_t = _network(tmp_path / "without", False).eval()
    assert with_t.tubular is not None and without_t.tubular is None
    assert not any(k.startswith("tubular.") for k in without_t.state_dict())
    assert with_t.tubular.fusion[3].weight.std() < 3e-4
    x = torch.randn((1, 2, 32, 32, 32))
    with torch.no_grad():
        a, b = with_t(x), without_t(x)
    assert all(torch.equal(p, q) for p, q in zip(a[1:], b[1:]))
    assert a[0].shape == b[0].shape
    with torch.no_grad():
        with_t.tubular.fusion[3].weight.zero_()
        with_t.tubular.fusion[3].bias.zero_()
        assert torch.equal(with_t(x)[0], b[0])


def test_tubular_tensor_output_and_gradients(tmp_path):
    torch.set_num_threads(2)
    net = _network(tmp_path, True)
    net.set_deep_supervision(False)
    y = net(torch.randn((1, 2, 32, 32, 32)))
    assert isinstance(y, torch.Tensor) and y.shape == (1, 2, 32, 32, 32)
    y.square().mean().backward()
    assert net.tubular.fusion[3].weight.grad is not None
    assert torch.isfinite(net.tubular.fusion[3].weight.grad).all()


def test_tubular_identity_and_parameter_defaults():
    module = TubularEnhancement()
    assert module.dil1.dilation == (1, 1, 1)
    assert module.dil5.dilation == (5, 5, 5)
    assert module.dil1.out_channels == 8
    with torch.no_grad():
        module.fusion[3].weight.zero_()
        module.fusion[3].bias.zero_()
        x = torch.rand((1, 2, 8, 8, 8))
        assert torch.equal(module(x), x)
