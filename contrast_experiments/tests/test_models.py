import torch
from stent_repro.eulig2021.models import A_DTE, A_DTR
from stent_repro.voth2022.models import B_DTE, B_DTR

def test_a_models_output_and_gradients():
    for model, channels in [(A_DTE(),1),(A_DTR(3),3)]:
        model.eval()
        x = torch.randn(1,channels,32,32,requires_grad=True)
        y = model(x)
        assert y.shape == (1,1,32,32)
        assert y.min() >= 0
        y.mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

def test_b_3d_preserves_channels_and_gradients():
    model = B_DTR().eval()
    x = torch.randn(1,3,16,16,16,requires_grad=True)
    y = model(x)
    assert y.shape == (1,1,16,16,16)
    assert torch.all((y >= 0) & (y <= 1))
    y.mean().backward()
    assert torch.isfinite(x.grad).all()

def test_b_dte_full_channel_layout_on_meta():
    with torch.device('meta'):
        model = B_DTE().eval()
        y = model(torch.empty(1,1,384,384,device='meta'))
    assert y.shape == (1,1,384,384)
    assert model.channels == (64,128,256,512,1024,2048,4096)
