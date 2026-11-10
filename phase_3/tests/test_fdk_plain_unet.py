import subprocess
import sys

import pytest
import torch


def test_plain_odd_shape_and_identity_callbacks():
    from phase_3.fdk_unet.plain_unet import PlainUNet3D

    torch.manual_seed(2026)
    model = PlainUNet3D().eval()
    x = torch.randn(1, 1, 33, 35, 37)
    with torch.no_grad():
        features = model.encode(x)
        assert [f.shape[1] for f in features] == [16, 32, 64, 128, 256]
        actual = model.decode(features, bottleneck_transform=lambda f: f,
                              skip_transform=lambda level, f, u: f)
        assert actual.shape == x.shape
        torch.testing.assert_close(actual, model(x), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model.decode(features), model(x), atol=1e-6, rtol=1e-5)


def test_decode_calls_each_block_once_and_does_not_mutate_features():
    from phase_3.fdk_unet.plain_unet import PlainUNet3D

    model = PlainUNet3D().eval()
    calls, order = [], []
    for name, module in list(model.up_projection.named_children()) + [
        ('d' + name, block) for name, block in model.decoder.named_children()
    ]:
        module.register_forward_hook(lambda m, i, o, n=name: calls.append(n))
    features = model.encode(torch.randn(1, 1, 32, 32, 32))
    before = [f.clone() for f in features]
    model.decode(features, bottleneck_transform=lambda f: (order.append(4) or f),
                 skip_transform=lambda level, f, u: (order.append(level) or f))
    assert order == [4, 3, 2, 1, 0]
    assert len(calls) == len(set(calls)) == 8
    for x, y in zip(features, before):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    with pytest.raises(ValueError):
        model.decode(features, bottleneck_transform=lambda f: f[:, :1])


def test_plain_import_without_monai():
    code = '''
import builtins
old = builtins.__import__
def guarded(name, *a, **k):
    if name.split('.')[0] in ('monai', 'nnunetv2', 'dynamic_network_architectures'):
        raise ImportError(name)
    return old(name, *a, **k)
builtins.__import__ = guarded
from phase_3.fdk_unet import PlainUNet3D, UNet3D
assert PlainUNet3D().head.out_channels == 1
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_plain_backward_and_input_validation():
    from phase_3.fdk_unet.plain_unet import PlainUNet3D

    model = PlainUNet3D()
    model(torch.randn(1, 1, 32, 32, 32)).square().mean().backward()
    for block in model.encoder:
        assert torch.isfinite(block[0].weight.grad).all()
        assert block[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(model.head.weight.grad).all()
    for x in (torch.zeros(1, 1, 31, 32, 32), torch.full((1, 1, 32, 32, 32), float('nan'))):
        with pytest.raises(ValueError):
            model(x)


@pytest.mark.slow
def test_plain_actual_128_shapes():
    from phase_3.fdk_unet.plain_unet import PlainUNet3D

    model = PlainUNet3D().eval()
    with torch.no_grad():
        features = model.encode(torch.zeros(1, 1, 128, 128, 128))
        assert [tuple(f.shape[2:]) for f in features] == [(s, s, s) for s in (128, 64, 32, 16, 8)]
        assert model.decode(features).shape == (1, 1, 128, 128, 128)
