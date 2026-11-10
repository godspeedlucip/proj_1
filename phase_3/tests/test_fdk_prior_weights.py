import os
from pathlib import Path
import pytest
import torch
from torch import nn


def tiny_encoder():
    root = nn.Module()
    root.swinViT = nn.Module()
    root.swinViT.patch_embed = nn.Linear(2,3)
    root.swinViT.attn = nn.Module()
    root.swinViT.attn.relative_position_bias_table = nn.Parameter(torch.ones(3,2))
    root.swinViT.attn.register_buffer('relative_position_index',torch.arange(3))
    return root


def test_synthetic_ssl_mapping_and_buffer_report(tmp_path):
    from phase_3.fdk_unet.prior_weights import load_ssl_weights
    model = tiny_encoder()
    source = {'module.swin_vit.'+k:v.clone() for k,v in model.swinViT.state_dict().items()}
    source['rotation_head.weight'] = torch.ones(2,2)
    source['module.norm.weight'] = torch.ones(768)
    source['module.convTrans3d.weight'] = torch.ones(1,1,2,2,2)
    p = tmp_path/'synthetic.pt'
    torch.save({'state_dict':source},p)
    report = load_ssl_weights(model,p)
    assert report.loaded_parameter_fraction == 1.0
    assert report.discarded_heads == ['module.convTrans3d.weight','module.norm.weight','rotation_head.weight']
    assert report.missing == report.unexpected == []


@pytest.mark.parametrize('damage',['missing','bias','shape','collision','hash','buffer','unexpected'])
def test_loader_rejects_missing_bias_shape_collision_and_wrong_hash(tmp_path,damage):
    from phase_3.fdk_unet.prior_weights import load_ssl_weights
    model = tiny_encoder()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    source = {'module.'+k:v.clone() for k,v in model.swinViT.state_dict().items()}
    if damage == 'missing': del source['module.patch_embed.weight']
    if damage == 'bias': del source['module.attn.relative_position_bias_table']
    if damage == 'shape': source['module.patch_embed.weight'] = torch.ones(8)
    if damage == 'collision': source['swinViT.patch_embed.weight'] = source['module.patch_embed.weight']
    if damage == 'buffer': source['module.attn.relative_position_index'] += 1
    if damage == 'unexpected': source['module.unknown.weight'] = torch.ones(1)
    p = tmp_path/'bad.pt'; torch.save({'state_dict':source},p)
    with pytest.raises(ValueError):
        load_ssl_weights(model,p,expected_sha256='0'*64 if damage=='hash' else None)
    for key,value in model.state_dict().items(): torch.testing.assert_close(value,before[key],atol=0,rtol=0)


@pytest.mark.prior_artifact
@pytest.mark.monai
def test_actual_official_ssl_artifact():
    pytest.importorskip('monai')
    if not os.environ.get('FDK_SSL_WEIGHTS'): pytest.skip('FDK_SSL_WEIGHTS not provided: real SSL NOT verified')
    from phase_3.fdk_unet.swin_prior import SwinPriorEncoder
    from phase_3.fdk_unet.prior_weights import load_ssl_weights
    report = load_ssl_weights(SwinPriorEncoder(execute_through='layers3'),Path(os.environ['FDK_SSL_WEIGHTS']))
    assert report.loaded_parameter_fraction == 1.0
