"""Atomic self-contained run checkpoints. Load only locally trusted run files."""
from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import torch
from .config import SPEC,resolve_config
from .manifest import json_digest
from .model_factory import build_model
from .predict import validate_device
from .prior_stats import PriorStats


@dataclass
class LoadedRun:
    model: object
    config: object
    stats: object
    manifest_snapshot: dict
    split_snapshot: dict
    prior_load_report: dict | None
    training_state: dict
    payload: dict


def save_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=f'.{path.name}-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


def load_run(path,*,device,for_training=False,expected_config=None,expected_split_digest=None):
    device=validate_device(device)
    # RNG snapshots include Python/NumPy state; these are trusted local run artifacts.
    p=torch.load(Path(path),map_location='cpu',weights_only=False)
    required={'model_id','config','config_digest','model_state','stats','manifest_snapshot','split_snapshot',
              'split_digest','prior_load_report','training_state'}
    if p.get('schema_version')!='fdk_unet_checkpoint_v3' or p.get('spec_version')!=SPEC or not required<=set(p):
        raise ValueError('Unsupported or incomplete checkpoint')
    config=resolve_config(p['config'])
    if p['model_id']!=config.model_id or p['config_digest']!=json_digest(config.resolved):raise ValueError('Checkpoint model/config mismatch')
    if expected_config and config.resolved!=expected_config.resolved:raise ValueError('Resume config overrides are forbidden')
    if p['split_digest']!=json_digest(p['split_snapshot']):raise ValueError('Corrupt split snapshot')
    if expected_split_digest and p['split_digest']!=expected_split_digest:raise ValueError('Wrong fold/split digest')
    stats=PriorStats.from_dict(p['stats']) if p['stats'] is not None else None
    if config.resolved['prior'] is not None and stats is None:raise ValueError('Fusion checkpoint missing preprocessing')
    if for_training:
        state_keys={'optimizer','scaler','global_step','stage_step','total_step','best_metric','rng','patch_stream'}
        if not state_keys<=set(p['training_state']):raise ValueError('Incomplete training state')
    model=build_model(config,model_state=p['model_state']).to(device)
    model.train(for_training)
    return LoadedRun(model,config,stats,p['manifest_snapshot'],p['split_snapshot'],p['prior_load_report'],p['training_state'],p)


def environment_record(device):
    root=Path(__file__).resolve().parents[2]
    def git(*args):
        result=subprocess.run(['git',*args],cwd=root,text=True,encoding='utf8',errors='replace',capture_output=True)
        return result.stdout.strip() if result.returncode==0 else None
    status=git('status','--porcelain')
    versions={}
    for name in ('torch','numpy','monai','einops','nibabel','scipy','scikit-image'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':versions,
            'cuda':torch.version.cuda,'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'source_commit':git('rev-parse','HEAD'),'dirty':bool(status),'dirty_summary':status,
            'diff_digest':json_digest(git('diff','--binary'))}
