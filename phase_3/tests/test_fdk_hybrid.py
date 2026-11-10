import torch
from phase_3.fdk_unet.plain_unet import PlainUNet3D
from phase_3.tests.fdk_fixtures import make_test_hybrid


def test_zero_alpha_matches_plain_for_all_h_variants():
    torch.manual_seed(2026)
    plain = PlainUNet3D().eval()
    x = torch.randn(1,1,32,32,32)
    with torch.no_grad():
        expected = plain(x)
        for name in ('H00','H10','H01','C'):
            model = make_test_hybrid(name, plain.state_dict())
            torch.testing.assert_close(model(x,x), expected, atol=1e-6, rtol=1e-5)
            model.train()
            assert not model.prior.training
            assert all(not p.requires_grad for p in model.prior.parameters())


def test_shared_modules_have_identical_initial_tensors():
    state = PlainUNet3D().state_dict()
    reference = make_test_hybrid('C',state).state_dict()
    for name in ('B','H00','H10','H01'):
        for key, value in make_test_hybrid(name,state).state_dict().items():
            torch.testing.assert_close(value,reference[key],atol=0,rtol=0)


def test_g3_reaches_logits_without_rewriting_encoder():
    model = make_test_hybrid('C',PlainUNet3D().state_dict())
    with torch.no_grad(): model.bottleneck.alpha.fill_(0.3)
    x = torch.randn(1,1,32,32,32)
    with torch.no_grad():
        features = model.unet.encode(x)
        before = [f.clone() for f in features]
        a = model.unet.decode(features, bottleneck_transform=lambda f: model.bottleneck(f,torch.ones(1,384,2,2,2)))
        b = model.unet.decode(features, bottleneck_transform=lambda f: model.bottleneck(f,torch.zeros(1,384,2,2,2)))
        assert not torch.allclose(a,b)
        for f, old in zip(features,before): torch.testing.assert_close(f,old,atol=0,rtol=0)
