"""Opt-in real CUDA capacity acceptance; never substitutes a CPU/smaller run."""
import gc
import json
import os
from pathlib import Path
import time
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize('name',['A','B','H00','H10','H01','C'])
def test_actual_128_batch2_core_models_fit_and_preserve_contract(tmp_path,name):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable: 128^3 GPU acceptance NOT completed')
    pytest.importorskip('monai')
    from phase_3.fdk_unet.config import get_experiment_config
    from phase_3.fdk_unet.model_factory import build_model
    from phase_3.fdk_unet.data import SampleBatch
    from phase_3.fdk_unet.train import train_step,configure_determinism
    from phase_3.fdk_unet.initialization import tensor_digest
    from phase_3.fdk_unet.prior_weights import file_sha256
    from phase_3.fdk_unet.manifest import json_digest
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    determinism=configure_determinism(torch.device('cuda'))
    cfg=get_experiment_config(name);weights=os.environ.get('FDK_SSL_WEIGHTS')
    if name not in ('A','B'):
        if not weights:pytest.fail('FDK_SSL_WEIGHTS required for actual SSL GPU acceptance')
        artifact=tmp_path/'prior.json';artifact.write_text(json.dumps({'schema_version':'fdk_prior_artifact_v1','init':'ssl','weights':str(Path(weights).resolve()),'sha256':file_sha256(weights)}))
        cfg.resolved['prior']['artifact']=str(artifact)
    model=optimizer=batch=None;handles=[];shapes={};calls={3:0,4:0}
    report={'experiment_id':name,'patch_size':[128]*3,'batch_size':2,'amp_dtype':'float16',
            'device':torch.cuda.get_device_name(0),'config_digest':json_digest(cfg.resolved),
            'input':'synthetic standard normal, seed 2026; capacity only','status':'running','determinism':determinism}
    try:
        model=build_model(cfg).cuda().train();unet=model if name=='A' else model.unet
        for i,block in enumerate(unet.encoder):
            handles.append(block.register_forward_hook(lambda m,x,y,i=i:shapes.__setitem__(str(i),list(y.shape))))
        if name!='A':
            before=tensor_digest({'prior':model.prior.state_dict()})
            for i in calls:
                handles.append(getattr(model.prior.swinViT,f'layers{i}')[0].register_forward_hook(lambda m,x,y,i=i:calls.__setitem__(i,calls[i]+1)))
            report['prior_parameters']=model.prior.parameter_counts()
            report['prior_sha256']=getattr(model,'prior_load_report',None)
            if report['prior_sha256'] is not None:report['prior_sha256']=report['prior_sha256']['sha256']
        report['unet_parameters']=sum(p.numel() for p in unet.parameters())
        report['total_parameters']=sum(p.numel() for p in model.parameters())
        report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        report['fusion_parameters']=report['total_parameters']-report['unet_parameters']-(report.get('prior_parameters',{}).get('loaded',0))
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        torch.manual_seed(2026);x=torch.randn(2,1,128,128,128,device='cuda')
        batch=SampleBatch(x,x.sigmoid() if name!='A' else None,(x>1).float(),torch.ones_like(x),('synthetic0','synthetic1'),[])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        losses=train_step(model,batch,optimizer,torch.amp.GradScaler('cuda'),amp_dtype=torch.float16)
        torch.cuda.synchronize();report.update(update_seconds=time.perf_counter()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),losses=losses)
        assert shapes=={str(i):[2,c,128//2**i,128//2**i,128//2**i] for i,c in enumerate((16,32,64,128,256))}
        if name!='A':
            assert calls[4]==0 and calls[3]==(0 if name in ('H00','H01') else 1)
            assert tensor_digest({'prior':model.prior.state_dict()})==before
            assert not model.prior.training and all(p.grad is None for p in model.prior.parameters())
        assert all(block[0].weight.grad is not None and torch.isfinite(block[0].weight.grad).all() for block in unet.encoder)
        for h in handles:h.remove()
        handles=[];model.eval();torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            logits=model(x[:1],batch.image_swin[:1]) if name!='A' else model(x[:1])
        torch.cuda.synchronize();report['single_volume_seconds']=time.perf_counter()-start
        assert logits.shape==(1,1,128,128,128) and torch.isfinite(logits).all()
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        destination=Path(os.environ.get('FDK_GPU_REPORT','tmp/fdk_v3/gpu_resources.json'));destination.parent.mkdir(parents=True,exist_ok=True)
        rows=json.loads(destination.read_text()) if destination.exists() else {}
        rows[name]=report;destination.write_text(json.dumps(rows,indent=2),encoding='utf8')
        for h in handles:h.remove()
        del model,optimizer,batch;gc.collect();torch.cuda.empty_cache()


@pytest.mark.gpu
def test_cuda_resume_records_actual_difference(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA resume comparison unavailable')
    import importlib
    from phase_3.tests.test_fdk_train import inputs,assert_state_equal
    from phase_3.fdk_unet.checkpoint import load_run
    t=importlib.import_module('phase_3.fdk_unet.train')
    cfg,manifest,splits=inputs(tmp_path)
    args=dict(fold=0,device=torch.device('cuda'))
    continuous=t.train(cfg,manifest,splits,output=tmp_path/'continuous',**args)
    save=t.save_checkpoint
    def interrupted(path,payload):
        save(path,payload)
        if path.name=='last.pt' and payload['training_state']['global_step']==2:raise RuntimeError('interrupt')
    monkeypatch.setattr(t,'save_checkpoint',interrupted)
    with pytest.raises(RuntimeError,match='interrupt'):t.train(cfg,manifest,splits,output=tmp_path/'resumed',**args)
    monkeypatch.setattr(t,'save_checkpoint',save)
    resumed=t.train(cfg,manifest,splits,output=tmp_path/'resumed',resume=tmp_path/'resumed/last.pt',**args)
    a=load_run(continuous.last_checkpoint,device=torch.device('cpu'),for_training=True)
    b=load_run(resumed.last_checkpoint,device=torch.device('cpu'),for_training=True)
    assert_state_equal(a.training_state['patch_stream'],b.training_state['patch_stream'])
    diffs={key:float((value.to(torch.float64)-b.model.state_dict()[key].to(torch.float64)).abs().max()) for key,value in a.model.state_dict().items()}
    assert all(np_isfinite(v) for v in diffs.values())
    result={'max_absolute_tensor_difference':max(diffs.values()),'exact_in_this_run':all(v==0 for v in diffs.values()),
            'limitation':'Observed result only; CUDA MaxPool3d backward is nondeterministic','completed_steps':resumed.completed_steps}
    path=Path(os.environ.get('FDK_GPU_REPORT','tmp/fdk_v3/gpu_resources.json')).with_name('gpu_resume.json')
    path.write_text(json.dumps(result,indent=2))


def np_isfinite(value):
    import math
    return math.isfinite(value)
