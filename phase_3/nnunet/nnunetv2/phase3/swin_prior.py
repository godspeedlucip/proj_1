"""MONAI 1.3.0 Swin prefix; normalized outputs never feed the next stage."""
import torch
from torch import nn


class SwinPriorEncoder(nn.Module):
    def __init__(self, *, execute_through, use_checkpoint=False):
        super().__init__()
        if execute_through not in ('layers2','layers3'):
            raise ValueError('execute_through must be layers2 or layers3')
        try:
            import monai
            from monai.networks.nets.swin_unetr import SwinTransformer
        except (ImportError, AttributeError) as exc:
            raise ImportError('MONAI 1.3.0 in a compatible Python environment is required for fusion') from exc
        if monai.__version__ != '1.3.0':
            raise ImportError('The validated implementation contract requires MONAI 1.3.0')
        self.execute_through = execute_through
        self.swinViT = SwinTransformer(
            in_chans=1, embed_dim=48, window_size=(7,7,7), patch_size=(2,2,2),
            depths=(2,2,2,2), num_heads=(3,6,12,24), mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
            patch_norm=False, use_checkpoint=use_checkpoint, spatial_dims=3,
            downsample='merging', use_v2=False)
        self.set_trainable_stages(())

    def set_trainable_stages(self, stages):
        if any(s != 'layers3' for s in stages) or (stages and self.execute_through != 'layers3'):
            raise ValueError('Only executed layers3 may be explicitly unfrozen')
        self.trainable_stages = tuple(stages)
        self.swinViT.requires_grad_(False)
        for stage in stages:
            getattr(self.swinViT,stage).requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.swinViT.eval()
        for stage in getattr(self,'trainable_stages',()):
            getattr(self.swinViT,stage).train(mode)
        return self

    def forward(self, image):
        if image.ndim != 5 or image.shape[1] != 1 or any(s < 32 or s % 32 for s in image.shape[2:]):
            raise ValueError('Swin input must be [B,1,D,H,W], edges divisible by 32')
        x = self.swinViT.pos_drop(self.swinViT.patch_embed(image))
        outputs = {}
        for i in range(1, int(self.execute_through[-1])+1):
            x = getattr(self.swinViT,f'layers{i}')[0](x.contiguous())
            outputs[i] = self.swinViT.proj_out(x, normalize=True)
        return outputs

    def parameter_counts(self):
        stop = int(self.execute_through[-1])
        return {'loaded': sum(p.numel() for p in self.parameters()),
                'executed': sum(p.numel() for name,p in self.swinViT.named_parameters()
                                if not any(name.startswith(f'layers{i}.') for i in range(stop+1,5))),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)}
