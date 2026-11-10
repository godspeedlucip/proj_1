import math
import pytest
import torch


def test_masked_loss_hand_value_padding_and_equal_case_weight():
    from phase_3.fdk_unet.losses import foreground_dice_bce
    z=torch.zeros(2,1,1,1,2,requires_grad=True)
    y=torch.tensor([0.,0.,1.,1.]).reshape_as(z);mask=torch.ones_like(z)
    r=foreground_dice_bce(z,y,mask)
    assert r['bce'].item()==pytest.approx(math.log(2))
    assert r['dice'].item()==pytest.approx((.5+.25)/2)
    pad=lambda t:torch.nn.functional.pad(t,(1,2),value=0)
    torch.testing.assert_close(r['loss'],foreground_dice_bce(pad(z),pad(y),pad(mask))['loss'])
    r['loss'].backward();assert torch.isfinite(z.grad).all()
    with pytest.raises(ValueError):foreground_dice_bce(z,y,mask*0)


def test_extreme_finite_logits_and_empty_target():
    from phase_3.fdk_unet.losses import foreground_dice_bce
    z=torch.tensor([-1000.,1000.]).reshape(1,1,1,1,2).requires_grad_()
    result=foreground_dice_bce(z,torch.zeros_like(z),torch.ones_like(z))
    result['loss'].backward();assert torch.isfinite(z.grad).all()
