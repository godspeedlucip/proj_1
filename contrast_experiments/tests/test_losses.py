import torch
from stent_repro.eulig2021.losses import soft_dice_A
from stent_repro.voth2022.losses import soft_dice_B

def test_a_dice_uses_batch_sum_and_laplace():
    p = torch.tensor([[[[0.2,0.8]]],[[[0.5,0.1]]]], requires_grad=True)
    t = torch.tensor([[[[0.,1.]]],[[[1.,0.]]]])
    assert torch.allclose(soft_dice_A(p,t), 1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1))
    soft_dice_A(p,t).backward()
    assert torch.isfinite(p.grad).all()

def test_b_dice_is_finite_for_empty_and_nonempty():
    for target in (torch.zeros(1,1,2,2,2), torch.ones(1,1,2,2,2)):
        p = torch.full_like(target, 0.2, requires_grad=True)
        loss = soft_dice_B(p,target)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
