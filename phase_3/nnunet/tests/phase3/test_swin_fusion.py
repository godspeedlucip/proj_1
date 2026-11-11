import torch
from monai.networks.nets.swin_unetr import SwinTransformer

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.network import build_phase3_network
from nnunetv2.phase3.planning import make_phase3_plans
from nnunetv2.phase3.swin_prior import SwinPriorEncoder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from .fixtures import make_config, synthetic_dataset


def _make_network(tmp_path, **overrides):
    mp, sp = synthetic_dataset(tmp_path / "source")
    prepared = prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                                source_fold=0, dataset_id=908, output=tmp_path / "fold")
    cfg = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                     cnn__batch_size=1, training__max_optimizer_steps=4,
                                     training__updates_per_epoch=2, tubular__enabled=False,
                                     projection__enabled=False, projection__weight=0.0, **overrides))
    manager = PlansManager(make_phase3_plans(prepared, cfg))
    return build_phase3_network(manager, manager.get_configuration("3d_fullres"), 2, 2)


def test_swin_prefix_matches_monai_and_stops_at_requested_stage():
    torch.set_num_threads(2)
    prior = SwinPriorEncoder(execute_through="layers2").eval()
    x = torch.rand((1, 1, 32, 32, 32))
    with torch.no_grad():
        actual = prior(x)
        stem = prior.swinViT
        z = stem.pos_drop(stem.patch_embed(x))
        expected = {}
        for level in (1, 2):
            z = getattr(stem, f"layers{level}")[0](z.contiguous())
            expected[level] = stem.proj_out(z, normalize=True)
    assert set(actual) == {1, 2}
    assert [(k, tuple(v.shape[1:])) for k, v in actual.items()] == [
        (1, (96, 8, 8, 8)), (2, (192, 4, 4, 4))]
    assert all(torch.allclose(actual[k], expected[k], atol=1e-6, rtol=1e-6) for k in actual)


def test_zero_alpha_matches_cnn_and_frozen_prior_has_no_gradient(tmp_path):
    torch.set_num_threads(2)
    fused = _make_network(tmp_path / "fused")
    baseline = _make_network(tmp_path / "baseline", swin__enabled=False,
                             swin__execute_through="layers2", fusion__enabled=False,
                             fusion__bottleneck_enabled=False)
    fused.eval()
    baseline.eval()
    x = torch.rand((1, 2, 32, 32, 32))
    with torch.no_grad():
        a, b = fused(x), baseline(x)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert all(not p.requires_grad for p in fused.prior.parameters())
    fused.train()
    assert not fused.prior.training
    with torch.no_grad():
        fused.bottleneck.alpha.fill_(0.5)
        for module in fused.skip_fusions.values():
            module.alpha.fill_(0.5)
            module.gate[-1].bias.fill_(1.0)
    loss = sum(out.square().mean() for out in fused(x))
    loss.backward()
    assert fused.bottleneck.align.weight.grad is not None
    assert torch.isfinite(fused.bottleneck.align.weight.grad).all()
    assert fused.skip_fusions["2"].gate[-1].weight.grad is not None
    assert all(p.grad is None for p in fused.prior.parameters())
    assert fused.fusion_telemetry()["2"]["beta"] > 0


def test_decoder_conditioning_changes_path(tmp_path):
    torch.set_num_threads(2)
    network = _make_network(tmp_path)
    module = network.skip_fusions["3"]
    feature = torch.randn((1, 256, 4, 4, 4))
    prior = torch.randn((1, 192, 4, 4, 4))
    up = torch.randn_like(feature)
    with torch.no_grad():
        module.alpha.fill_(1)
        module.gate[-1].weight.normal_(0, .01)
        conditioned = module(feature, prior, up)
        module.decoder_conditioning = False
        unconditioned = module(feature, prior, up)
    assert not torch.equal(conditioned, unconditioned)
