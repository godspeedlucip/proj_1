"""Per-case masked foreground soft Dice + BCE, accumulated in float32."""
import torch
from torch.nn import functional as F


def foreground_dice_bce(logits,target,valid_mask):
    if logits.ndim!=5 or logits.shape[1]!=1 or logits.shape!=target.shape or logits.shape!=valid_mask.shape:
        raise ValueError('Loss expects matching [B,1,D,H,W] tensors')
    if not all(torch.isfinite(x).all() for x in (logits,target,valid_mask)):
        raise ValueError('Nonfinite loss input')
    if not torch.all((target==0)|(target==1)) or not torch.all((valid_mask==0)|(valid_mask==1)):
        raise ValueError('Target and valid_mask must be binary')
    z,y,m=logits.float(),target.float(),valid_mask.float();axes=(1,2,3,4)
    count=m.sum(axes)
    if torch.any(count==0):raise ValueError('No valid voxels in sample')
    p=torch.sigmoid(z)
    bce=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*m).sum(axes)/count
    dice=1-(2*(m*p*y).sum(axes)+1)/((m*p).sum(axes)+(m*y).sum(axes)+1)
    return {'loss':(bce+dice).mean(),'bce':bce.mean(),'dice':dice.mean()}
