import torch
from torch import nn
import torch.nn.functional as F


def _block(cin, cout):
    groups = min(8, cout)
    while cout % groups:
        groups -= 1
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True))


class MarkerUNet2D(nn.Module):
    def __init__(self, in_channels=1, widths=(32, 64, 128, 256)):
        super().__init__()
        if len(widths) != 4: raise ValueError("widths must contain four levels")
        w0,w1,w2,w3=map(int,widths)
        self.enc1=_block(in_channels,w0); self.enc2=_block(w0,w1); self.enc3=_block(w1,w2); self.enc4=_block(w2,w3)
        self.pool=nn.MaxPool2d(2)
        self.up3=nn.ConvTranspose2d(w3,w2,2,2); self.dec3=_block(w2*2,w2)
        self.up2=nn.ConvTranspose2d(w2,w1,2,2); self.dec2=_block(w1*2,w1)
        self.up1=nn.ConvTranspose2d(w1,w0,2,2); self.dec1=_block(w0*2,w0)
        self.head=nn.Conv2d(w0,1,1)

    def forward(self, x):
        h,w=x.shape[-2:]
        ph=(-h)%8; pw=(-w)%8
        if ph or pw: x=F.pad(x,(0,pw,0,ph),mode="replicate")
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        d3=self.dec3(torch.cat([self.up3(e4),e3],1)); d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.head(d1)[...,:h,:w]
