import copy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch


def payload(model,cfg,stats=None):
    from phase_3.fdk_unet.manifest import json_digest
    return {'schema_version':'fdk_unet_checkpoint_v3','spec_version':'fdk_unet_swin_fusion_spec_v3',
            'model_id':cfg.model_id,'config':cfg.resolved,'config_digest':json_digest(cfg.resolved),
            'model_state':model.state_dict(),'stats':stats.to_dict() if stats else None,
            'manifest_snapshot':{},'split_snapshot':{'train':['g0','g1'],'validation':['g2'],'test':['g3']},
            'split_digest':json_digest({'train':['g0','g1'],'validation':['g2'],'test':['g3']}),
            'prior_load_report':getattr(model,'prior_load_report',None),'training_state':{}}


def test_initializer_does_not_shift_sampling_rng(tmp_path):
    from phase_3.fdk_unet.initialization import create_initialization_bundle,capture_rng_state,restore_rng_state
    from phase_3.fdk_unet.model_factory import build_model
    from phase_3.fdk_unet.config import get_experiment_config
    state=capture_rng_state();expected=torch.rand(4);restore_rng_state(state)
    bundle=create_initialization_bundle(seed=2026,output=tmp_path/'init.pt')
    torch.testing.assert_close(torch.rand(4),expected,atol=0,rtol=0)
    again=create_initialization_bundle(seed=2026,output=tmp_path/'init.pt')
    assert bundle['tensor_digest']==again['tensor_digest']
    a=build_model(get_experiment_config('A'),initialization=bundle)
    for key,v in a.state_dict().items():torch.testing.assert_close(v,bundle['unet'][key],atol=0,rtol=0)
    with pytest.raises(ValueError):create_initialization_bundle(seed=2027,output=tmp_path/'init.pt')


def test_plain_checkpoint_restores_bn_and_logits(tmp_path):
    from phase_3.fdk_unet.config import get_experiment_config
    from phase_3.fdk_unet.model_factory import build_model
    from phase_3.fdk_unet.checkpoint import save_checkpoint,load_run
    cfg=get_experiment_config('A');model=build_model(cfg);x=torch.randn(1,1,32,32,32)
    model(x);model.eval()
    with torch.no_grad():expected=model(x)
    path=tmp_path/'model.pt';save_checkpoint(path,payload(model,cfg))
    restored=load_run(path,device=torch.device('cpu'))
    assert not restored.model.training
    with torch.no_grad():torch.testing.assert_close(restored.model(x),expected,atol=0,rtol=0)


@pytest.mark.parametrize('damage',['schema','model','fusion','fold','config'])
def test_checkpoint_rejects_wrong_schema_model_fold_and_fusion_flags(tmp_path,damage):
    from phase_3.fdk_unet.config import get_experiment_config
    from phase_3.fdk_unet.model_factory import build_model
    from phase_3.fdk_unet.checkpoint import save_checkpoint,load_run
    cfg=get_experiment_config('A');p=payload(build_model(cfg),cfg)
    if damage=='schema':p['schema_version']='old'
    if damage=='model':p['model_id']='swin_prior_fdk_unet3d_v1'
    if damage=='fusion':p['config']['fusion']={'bottleneck_enabled':True}
    if damage=='config':p['config']['seed']=99
    path=tmp_path/'bad.pt';save_checkpoint(path,p)
    with pytest.raises(ValueError):load_run(path,device=torch.device('cpu'),expected_split_digest='wrong' if damage=='fold' else None)


def test_atomic_write_keeps_previous_checkpoint_on_failure(tmp_path,monkeypatch):
    from phase_3.fdk_unet.checkpoint import save_checkpoint
    p=tmp_path/'last.pt';save_checkpoint(p,{'value':1});before=p.read_bytes()
    def fail(payload,stream):stream.write(b'partial');raise OSError('interrupted')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):save_checkpoint(p,{'value':2})
    assert p.read_bytes()==before


@pytest.mark.monai
def test_offline_checkpoint_restores_model_and_preprocessing(tmp_path,monkeypatch):
    pytest.importorskip('monai')
    from phase_3.fdk_unet.config import get_experiment_config
    from phase_3.fdk_unet.model_factory import build_model
    from phase_3.fdk_unet.swin_prior import SwinPriorEncoder
    from phase_3.fdk_unet.prior_weights import file_sha256
    from phase_3.fdk_unet.prior_stats import PriorStats
    from phase_3.fdk_unet.checkpoint import save_checkpoint,load_run
    cfg=get_experiment_config('C');weights=tmp_path/'external.pt'
    torch.save(SwinPriorEncoder(execute_through='layers3').swinViT.state_dict(),weights)
    artifact=tmp_path/'prior.json';artifact.write_text(json.dumps({'schema_version':'fdk_prior_artifact_v1','init':'ssl','weights':str(weights),'sha256':file_sha256(weights)}))
    cfg.resolved['prior']['artifact']=str(artifact)
    model=build_model(cfg).eval();x=torch.randn(1,1,32,32,32)
    with torch.no_grad():model.bottleneck.alpha.fill_(.2);expected=model(x,x)
    stats=PriorStats(-2,3,2026,'linear',65536,131072,('g0','g1'),'test','test')
    p=tmp_path/'best.pt';save_checkpoint(p,payload(model,cfg,stats))
    weights.unlink();artifact.unlink()
    import urllib.request
    def forbidden(*a,**k):raise AssertionError('offline restore attempted download')
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    loaded=load_run(p,device=torch.device('cpu'))
    assert loaded.stats.lo==-2
    with torch.no_grad():torch.testing.assert_close(loaded.model(x,x),expected,atol=1e-6,rtol=1e-5)
