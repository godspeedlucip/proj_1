"""Command line entry for precomputed paired-data experiments."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from stent_repro.training.data import PreparedDataset, read_training_manifest
from stent_repro.training.engine import train_prepared


def main(argv=None):
    p = argparse.ArgumentParser(description='Train on prepared paired samples (not strict paper reproduction).')
    p.add_argument('--method',choices=['A','B'],required=True)
    p.add_argument('--stage',choices=['dte','dtr'],required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['prepared-surrogate','strict'],required=True)
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--device',default='cuda')
    p.add_argument('--slice-channels',type=int,choices=[1,3,5],default=5)
    p.add_argument('--input-channels',type=int,choices=[1,3],default=3)
    p.add_argument('--resume')
    p.add_argument('--online-dte',action='store_true',help='resample DTE augmentation from generated source patches during training')
    args = p.parse_args(argv)
    if args.online_dte and args.stage != 'dte': raise ValueError('--online-dte only applies to DTE')
    if args.mode == 'strict':
        raise RuntimeError('strict reproduction is unavailable: original data, paper-specific simulation, production geometry validation and full-shape CUDA memory preflight remain unavailable')
    if args.method=='A':
        from stent_repro.eulig2021.config import RunConfig
        from stent_repro.eulig2021.models import A_DTE, A_DTR
    else:
        from stent_repro.voth2022.config import RunConfig
        from stent_repro.voth2022.models import B_DTE, B_DTR
    cfg=RunConfig.load(args.config)
    params=cfg.training[args.stage]
    epochs=args.epochs if args.epochs is not None else params.get('epochs')
    batch=args.batch_size if args.batch_size is not None else params['batch_size']
    if epochs is None or epochs < 1: raise ValueError('provide --epochs; the paper did not specify it')
    if batch < 1: raise ValueError('batch size must be positive')
    if args.method=='B' and args.stage=='dtr':
        if cfg.profile=='B-one-channel-ablation' and args.input_channels!=1: raise ValueError('one-channel ablation requires --input-channels 1')
        if cfg.profile!='B-one-channel-ablation' and args.input_channels!=3: raise ValueError('one-channel model requires its separate ablation profile')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; use --device cpu only for an explicitly labeled prepared experiment')
    manifest_path=Path(args.manifest).resolve()
    preprocessing = None
    provenance_path = manifest_path.parent / 'provenance.json'
    if provenance_path.exists():
        candidate = json.loads(provenance_path.read_text(encoding='utf-8'))
        hashes = candidate.get('generated_manifests_sha256', {})
        if manifest_path.name in hashes:
            if hashes[manifest_path.name] != hashlib.sha256(manifest_path.read_bytes()).hexdigest(): raise ValueError('generated manifest does not match preprocessing provenance')
            if candidate.get('method') != args.method or candidate.get('profile') != cfg.profile: raise ValueError('preprocessing method/profile mismatch')
            preprocessing = candidate
    rows=read_training_manifest(manifest_path,args.method,args.stage,profile=cfg.profile)
    kw={'slice_channels':args.slice_channels,'input_channels':args.input_channels}
    train=PreparedDataset(rows,'train',args.method,args.stage,augment=args.method=='B' and args.stage=='dtr',online_dte=args.online_dte,**kw)
    val=PreparedDataset(rows,'validation',args.method,args.stage,**kw)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if args.method=='A': model=A_DTE() if args.stage=='dte' else A_DTR(args.slice_channels)
    else: model=B_DTE() if args.stage=='dte' else B_DTR(args.input_channels)
    info={'preprocessing':preprocessing,'online_dte':args.online_dte,'mode':args.mode,'config':asdict(cfg),'manifest_path':str(manifest_path),
          'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'train_samples':len(train),'validation_samples':len(val),
          'slice_channels':args.slice_channels if args.method=='A' and args.stage=='dtr' else None,
          'input_channels':args.input_channels if args.method=='B' and args.stage=='dtr' else None,
          'augmentation':'random z flip and right-angle z rotation' if args.method=='B' and args.stage=='dtr' else 'none',
          'data_note':'precomputed samples; source and alignment supplied by user; no strict paper claims'}
    result=train_prepared(train,val,model,args.output,method=args.method,stage=args.stage,epochs=epochs,batch_size=batch,
                          learning_rate=params['learning_rate'],seed=cfg.seed,device=args.device,resume=args.resume,run_info=info)
    print(f"completed {len(result['history'])} epochs; best validation loss: {result['best_validation_loss']:.6g}")
    return result
