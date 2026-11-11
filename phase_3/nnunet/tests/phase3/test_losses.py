import pytest
import torch

from phase_3.fdk_unet.losses import foreground_dice_bce
from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.losses import MaskedForegroundDiceBCE, Phase3CompositeLoss
from .fixtures import make_config


def _config(projection=False, ds=True):
    return resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                      cnn__batch_size=1, training__max_optimizer_steps=4,
                                      training__updates_per_epoch=2,
                                      projection__enabled=projection,
                                      projection__weight=.1 if projection else 0.,
                                      deep_supervision__enabled=ds))


def test_two_logit_loss_matches_old_value_and_foreground_gradient():
    z = torch.randn((2, 1, 4, 5, 6), requires_grad=True)
    target = torch.randint(0, 2, z.shape).float()
    valid = torch.ones_like(target)
    valid[0, :, 1, 2, 3] = 0
    old = foreground_dice_bce(z, target, valid)
    old_grad = torch.autograd.grad(old["loss"], z)[0]
    two = torch.cat((torch.zeros_like(z), z.detach()), dim=1).requires_grad_()
    encoded = target.long().masked_fill(valid == 0, 2)
    result = MaskedForegroundDiceBCE()(two, encoded)
    result.total.backward()
    assert torch.allclose(result.total, old["loss"], atol=1e-6)
    assert torch.allclose(result.dice, old["dice"], atol=1e-6)
    assert torch.allclose(result.bce, old["bce"], atol=1e-6)
    assert torch.allclose(two.grad[:, 1:2], old_grad, atol=1e-6)
    assert torch.allclose(two.grad[:, :1], -old_grad, atol=1e-6)


def test_auxiliary_empty_valid_support_contributes_zero_without_reweighting():
    loss = Phase3CompositeLoss(_config())
    outputs = [torch.zeros((2, 2, n, n, n), requires_grad=True) for n in (8, 4, 2, 2)]
    targets = [torch.zeros((2, 1, n, n, n), dtype=torch.long) for n in (8, 4, 2, 2)]
    targets[1][0] = 2
    targets[2][0] = 2
    result = loss.breakdown(outputs, targets)
    assert result.valid_sample_counts == (2, 1, 1, 2)
    assert torch.allclose(result.segmentation, sum(w * part for w, part in zip((4/7, 2/7, 1/7), result.segmentation_scales)))
    result.total.backward()
    assert outputs[0].grad is not None
    assert outputs[3].grad is None
    with pytest.raises(ValueError, match="valid"):
        loss.breakdown(outputs, [torch.full_like(targets[0], 2), *targets[1:]])


def test_projection_is_called_once_on_final_logits_and_zero_weight_skips_it(monkeypatch):
    loss = Phase3CompositeLoss(_config(projection=True))
    calls = []
    def spy(logits, target):
        calls.append(logits)
        return logits.square().mean()
    monkeypatch.setattr(loss.projection, "forward", spy)
    outputs = [torch.randn((1, 2, n, n, n), requires_grad=True) for n in (8, 4, 2, 2)]
    targets = [torch.zeros((1, 1, n, n, n), dtype=torch.long) for n in (8, 4, 2, 2)]
    result = loss.breakdown(outputs, targets)
    assert len(calls) == 1 and calls[0] is outputs[0]
    assert torch.allclose(result.total, result.segmentation + .1 * result.projection)
    result.total.backward()
    assert outputs[0].grad is not None
    assert Phase3CompositeLoss(_config(projection=False)).projection is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_rejects_nonfinite_logits(bad):
    logits = torch.zeros((1, 2, 2, 2, 2))
    logits[0, 0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        MaskedForegroundDiceBCE()(logits, torch.zeros((1, 1, 2, 2, 2)))


def test_rejects_wrong_class_and_head_count():
    loss = Phase3CompositeLoss(_config())
    with pytest.raises(ValueError):
        MaskedForegroundDiceBCE()(torch.zeros((1, 1, 2, 2, 2)), torch.zeros((1, 1, 2, 2, 2)))
    with pytest.raises(ValueError):
        loss([torch.zeros((1, 2, 2, 2, 2))], [])
