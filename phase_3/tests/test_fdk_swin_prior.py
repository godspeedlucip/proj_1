import builtins
import subprocess
import sys
import pytest
import torch


@pytest.mark.monai
@pytest.mark.parametrize('prefix', ['layers2','layers3'])
@pytest.mark.parametrize('edge',[32,pytest.param(128,marks=pytest.mark.gpu)])
def test_prefix_matches_official_outputs(prefix,edge):
    pytest.importorskip('monai')
    from phase_3.fdk_unet.swin_prior import SwinPriorEncoder
    if edge==128 and not torch.cuda.is_available():pytest.skip('128^3 GPU prefix acceptance unavailable')
    device=torch.device('cuda' if edge==128 else 'cpu')
    prior = SwinPriorEncoder(execute_through=prefix).to(device).eval()
    counts = {3: 0, 4: 0}
    handles = []
    for i in counts:
        handles.append(getattr(prior.swinViT,f'layers{i}')[0].register_forward_hook(
            lambda m,x,y,j=i: counts.__setitem__(j,counts[j]+1)))
    image = torch.randn(1,1,edge,edge,edge,device=device)
    with torch.no_grad(): actual = prior(image)
    assert counts[4] == 0
    assert counts[3] == (1 if prefix == 'layers3' else 0)
    for h in handles: h.remove()
    with torch.no_grad(): expected = prior.swinViT(image,normalize=True)
    for i in actual:
        torch.testing.assert_close(actual[i],expected[i],atol=1e-6,rtol=1e-5)
        assert actual[i].shape[1] == {1:96,2:192,3:384}[i]


def test_missing_monai_fails_explicitly():
    code = '''
import builtins
old = builtins.__import__
def guarded(name,*a,**k):
    if name.split('.')[0]=='monai': raise ImportError('blocked')
    return old(name,*a,**k)
builtins.__import__=guarded
from phase_3.fdk_unet.swin_prior import SwinPriorEncoder
try: SwinPriorEncoder(execute_through='layers3')
except ImportError as e: assert 'MONAI' in str(e)
else: raise AssertionError('silent fallback')
'''
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.monai
def test_random_prior_never_opens_ssl_file(monkeypatch):
    pytest.importorskip('monai')
    from phase_3.fdk_unet.swin_prior import SwinPriorEncoder
    def forbidden(*a,**k): raise AssertionError('random prior read weights')
    monkeypatch.setattr(torch,'load',forbidden)
    assert SwinPriorEncoder(execute_through='layers3').swinViT is not None
