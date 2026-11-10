"""Fixed optimizer-step budgets, group validation and exact boundary resume."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .checkpoint import save_checkpoint,load_run,environment_record
from .config import SPEC,resolve_config
from .data import PatchStream
from .evaluate import compute_case_metrics,aggregate_group_metrics
from .initialization import create_initialization_bundle,capture_rng_state,restore_rng_state,seed_for
from .losses import foreground_dice_bce
from .manifest import json_digest,validate_data,select_cases
from .model_factory import build_model
from .nifti_io import load_volume
from .predict import predict_volume,validate_device
from .prior_stats import PriorStats,validate_prior_stats


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path
    completed_steps: int
    best_validation_dice: float


def configure_determinism(device):
    cuda=device.type=='cuda'
    if cuda:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    # PyTorch 2.3.1 CUDA MaxPool3d backward is explicitly nondeterministic.
    # Preserve the specified architecture, surface warnings, and record this limit.
    torch.use_deterministic_algorithms(True,warn_only=cuda)
    return {'enabled':True,'warn_only':cuda,'limitation':'CUDA max_pool3d backward is not deterministic; no bitwise GPU resume claim' if cuda else None}


def train_step(model,batch,optimizer,scaler,*,amp_dtype):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=batch.image_unet.device.type,dtype=amp_dtype,enabled=amp_dtype is not None):
        logits=model(batch.image_unet,batch.image_swin) if batch.image_swin is not None else model(batch.image_unet)
        losses=foreground_dice_bce(logits,batch.target,batch.valid_mask)
    if not torch.isfinite(losses['loss']):raise ValueError('Nonfinite loss')
    scaled=scaler is not None and scaler.is_enabled()
    if scaled:
        scaler.scale(losses['loss']).backward();scaler.unscale_(optimizer)
    else:losses['loss'].backward()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('Nonfinite gradient: optimizer update rejected')
    if scaled:scaler.step(optimizer);scaler.update()
    else:optimizer.step()
    if any(not torch.isfinite(p).all() for group in optimizer.param_groups for p in group['params']):
        raise ValueError('Nonfinite parameter after optimizer update')
    return {key:float(value.detach()) for key,value in losses.items()}


def validate(model,manifest,split,config,stats,device):
    rows=[]
    for case in select_cases(manifest,split,'validation'):
        volume=load_volume(case,require_label=True)
        prediction=predict_volume(model,volume,config,stats,device=device)
        rows.append(compute_case_metrics(prediction,volume.target,case_id=case.case_id,group_id=case.group_id))
    return aggregate_group_metrics(rows)


def _write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')


def train(config,manifest,splits,*,fold,device,output,prior_artifact=None,initialization_artifact=None,resume=None):
    device=validate_device(device);config=resolve_config(config.resolved);output=Path(output).resolve()
    determinism=configure_determinism(device)
    if fold not in splits:raise ValueError('Unknown fold')
    if config.resolved['run_kind']=='formal' and device.type!='cuda':raise ValueError('Formal training requires validated CUDA capacity; use explicit smoke config on CPU')
    if prior_artifact is not None:
        if resume is not None or config.resolved['prior'] is None:raise ValueError('prior_artifact is only for a new fusion run')
        raw=config.resolved;raw['prior']['artifact']=str(Path(prior_artifact).resolve());config=resolve_config(raw)
    if resume is not None and initialization_artifact is not None:raise ValueError('Resume cannot override initialization')
    split=splits[fold];split_snapshot=split.to_dict();split_digest=json_digest(split_snapshot)
    audit=validate_data(manifest,splits,formal=config.resolved['run_kind']=='formal')
    manifest_snapshot=manifest.to_dict();manifest_digest=json_digest(manifest_snapshot)
    stats=None;step=0;best=-float('inf');loaded=None;prior_report=None
    if resume is not None:
        resume=Path(resume).resolve()
        if output!=resume.parent:raise ValueError('Resume must continue its original run directory')
        loaded=load_run(resume,device=device,for_training=True,expected_config=config,expected_split_digest=split_digest)
        if (loaded.payload.get('data_digest')!=audit['file_digest'] or loaded.payload.get('manifest_digest')!=manifest_digest
                or loaded.payload.get('fold')!=fold):raise ValueError('Resume data/fold changed')
        if loaded.payload.get('amp_dtype')!=('float16' if device.type=='cuda' else None):
            raise ValueError('Resume precision/device class differs; exact optimizer continuation unavailable')
        model=loaded.model;stats=loaded.stats;prior_report=loaded.prior_load_report
        step=loaded.training_state['global_step'];best=loaded.training_state['best_metric']
    else:
        if output.exists():raise FileExistsError(f'Run output already exists: {output}')
        if config.resolved['prior'] is not None:
            artifact_path=config.resolved['prior']['artifact']
            if not artifact_path:raise ValueError('Fusion training requires prepare-prior artifact with fold statistics')
            artifact=json.loads(Path(artifact_path).read_text(encoding='utf8'))
            if artifact.get('schema_version')!='fdk_prior_artifact_v1' or artifact.get('init')!=config.resolved['prior']['init']:
                raise ValueError('Prior artifact version/initialization mismatch')
            stats=PriorStats.from_dict(artifact['stats']);validate_prior_stats(stats,manifest,split)
        output.mkdir(parents=True,exist_ok=False)
        bundle=create_initialization_bundle(seed=config.seed,output=Path(initialization_artifact) if initialization_artifact else output/'initialization.pt')
        model=build_model(config,initialization=bundle).to(device);prior_report=getattr(model,'prior_load_report',None)
        training_seed=seed_for(config.seed,'training')
        random.seed(training_seed);np.random.seed(training_seed);torch.manual_seed(training_seed)
        _write_json(output/'config.resolved.json',config.resolved)
        _write_json(output/'manifest.snapshot.json',manifest_snapshot)
        _write_json(output/'splits.snapshot.json',{'schema_version':'fdk_stent_splits_v1','folds':{str(k):v.to_dict() for k,v in splits.items()}})
        _write_json(output/'data.audit.json',audit);_write_json(output/'environment.json',dict(environment_record(device),determinism=determinism))
        if prior_report is not None:_write_json(output/'prior_load_report.json',prior_report)
    if stats is not None:validate_prior_stats(stats,manifest,split)
    opt=config.resolved['optimizer']
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    stream=PatchStream(manifest,split,config,stats)
    logpath=output/'metrics.jsonl'
    if loaded is not None:
        optimizer.load_state_dict(loaded.training_state['optimizer']);scaler.load_state_dict(loaded.training_state['scaler'])
        stream.load_state_dict(loaded.training_state['patch_stream']);restore_rng_state(loaded.training_state['rng'])
        offset=loaded.training_state['log_bytes']
        if not logpath.is_file() or logpath.stat().st_size<offset:raise ValueError('Run log missing or shorter than checkpoint')
        with logpath.open('r+b') as log:log.truncate(offset)
    def log(record):
        with logpath.open('ab') as handle:
            handle.write((json.dumps(record,allow_nan=False)+'\n').encode());handle.flush();os.fsync(handle.fileno())
    budget=config.resolved['max_optimizer_steps']
    if step>budget:raise ValueError('Checkpoint exceeds fixed budget')
    for current in range(step+1,budget+1):
        model.train();batch=stream.next_batch().to(device)
        if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();metrics=train_step(model,batch,optimizer,scaler,amp_dtype=torch.float16 if device.type=='cuda' else None)
        if device.type=='cuda':torch.cuda.synchronize(device)
        record={'event':'train','step':current,**metrics,'lr':optimizer.param_groups[0]['lr'],
                'case_ids':batch.case_ids,'seconds':time.perf_counter()-start,
                'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
        if hasattr(model,'fusion_telemetry'):record['fusion']=model.fusion_telemetry()
        log(record);improved=False
        if current%config.resolved['validate_every']==0 or current==budget:
            summary=validate(model,manifest,split,config,stats,device);score=summary['mean']['dice']
            if not np.isfinite(score):raise ValueError('Nonfinite validation metric')
            improved=score>best
            if improved:best=score
            log({'event':'validation','step':current,'group_dice':score,'groups':summary['groups']})
        state={'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'global_step':current,
               'stage_step':current,'total_step':current,'best_metric':best,'rng':capture_rng_state(),
               'patch_stream':stream.state_dict(),'log_bytes':logpath.stat().st_size}
        payload={'schema_version':'fdk_unet_checkpoint_v3','spec_version':SPEC,'model_id':config.model_id,
                 'config':config.resolved,'config_digest':json_digest(config.resolved),'model_state':model.state_dict(),
                 'stats':stats.to_dict() if stats else None,'manifest_snapshot':manifest_snapshot,'manifest_digest':manifest_digest,
                 'split_snapshot':split_snapshot,'split_digest':split_digest,'fold':fold,'data_digest':audit['file_digest'],'data_audit':audit,
                 'prior_load_report':prior_report,'training_state':state,'amp_dtype':'float16' if device.type=='cuda' else None,
                 'determinism':determinism}
        if improved:save_checkpoint(output/'best.pt',payload)
        save_checkpoint(output/'last.pt',payload)
        step=current
    return TrainResult(output,output/'last.pt',output/'best.pt',step,float(best))
