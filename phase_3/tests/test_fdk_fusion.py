import torch
import pytest


@pytest.mark.parametrize('guided', [True, False])
def test_gate_depends_on_u_only_when_enabled(guided):
    from phase_3.fdk_unet.fusion import DecoderGuidedFusion
    m = DecoderGuidedFusion(96, 64, decoder_conditioning=guided)
    with torch.no_grad():
        m.alpha.fill_(0.4)
        m.align.weight.fill_(0.01)
        m.gate[0].weight.fill_(0.01)
        m.gate[2].weight.fill_(0.01)
    f, g = torch.ones(1,64,3,3,3), torch.ones(1,96,3,3,3)
    u = torch.ones_like(f, requires_grad=True)
    first = m(f,g,u)
    second = m(f,g,u*2)
    first.sum().backward()
    if guided:
        assert not torch.equal(first, second)
        assert u.grad.abs().sum() > 0
    else:
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        assert u.grad is None or not u.grad.any()


def test_gradient_activation_order():
    from phase_3.fdk_unet.fusion import BottleneckFusion
    m = BottleneckFusion(8, 16)
    f, g = torch.ones(1,16,3,3,3), torch.ones(1,8,3,3,3)
    with torch.no_grad():
        m.align.weight.fill_(0.1)
        m.gate[0].weight.fill_(0.01)
    m(f,g).sum().backward()
    assert m.alpha.grad.abs() > 0
    assert m.align.weight.grad.abs().sum() == 0
    m.zero_grad()
    with torch.no_grad(): m.alpha.fill_(0.1)
    m(f,g).sum().backward()
    assert m.align.weight.grad.abs().sum() > 0
    assert m.gate[2].weight.grad.abs().sum() > 0
    assert m.gate[0].weight.grad.abs().sum() == 0
    m.zero_grad()
    with torch.no_grad(): m.gate[2].weight.fill_(0.01)
    m(f,g).sum().backward()
    assert m.gate[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError): m(f, g[:,:,:2])
