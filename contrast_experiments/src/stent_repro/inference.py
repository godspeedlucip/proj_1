"""Projection-only inference. Ground truth is deliberately absent from the API."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from stent_repro.configuration import resolve_config,validate_projection_metadata,validate_case_angles
from stent_repro.geometry import Grid,circular_view,fdk,backproject_each
from stent_repro.io import load_projections,save_projections,save_volume
from stent_repro.model_registry import build_model,model_spec
from stent_repro.provenance import sha256,freeze_or_verify_manifest,verify_fingerprints,fingerprint_files


def _padded_forward(model,x,stride):
    shape=x.shape[2:];padding=[]
    for size in reversed(shape): padding.extend((0,(-size)%stride))
    y=model(F.pad(x,padding,mode='replicate') if any(padding) else x)
    y=y[(slice(None),slice(None),*[slice(0,n) for n in shape])]
    if y.shape!=(x.shape[0],1,*shape) or not torch.isfinite(y).all(): raise ValueError('network output shape or finite values invalid')
    return y


def _configured_forward(model,x,stride,config,stage,device):
    tiling=config['inference']['tiling'];shape=tiling[stage+'_shape']
    if tiling['enabled'] and shape is not None:
        from stent_repro.runtime import tiled_predict
        overlap=[int(n*tiling['overlap_fraction']//stride)*stride for n in shape]
        return tiled_predict(model,x,shape,overlap,stride)
    return _padded_forward(model,x.to(device),stride),None


def predict_geometric(geometric,config,model,device='cpu',return_coverage=False):
    """DTR-only inference, also used on the fixed validation-volume panel."""
    model=model.to(device).eval();stride=2**(len(model_spec(config,'dtr')[2])-1)
    with torch.inference_mode():
        if config['method']=='A':
            v=(geometric-float(geometric.mean()))/max(float(geometric.std()),1e-6)
            chunks=[];coverages=[];radius=config['network']['slice_channels']//2;batch=config['inference']['slice_batch']
            for first in range(0,len(v),batch):
                x=np.stack([v[np.clip(np.arange(z-radius,z+radius+1),0,len(v)-1)] for z in range(first,min(first+batch,len(v)))])
                y,coverage=_configured_forward(model,torch.from_numpy(x.copy()),stride,config,'dtr',device)
                chunks.append(y[:,0].cpu().numpy())
                if coverage is not None:coverages.extend([coverage.cpu().numpy()]*len(x))
            value=np.concatenate(chunks);cov=np.stack(coverages) if coverages else None
            return (value,cov) if return_coverage else value
        x=geometric.sum(axis=0,keepdims=True) if config['network']['input_channels']==1 else geometric
        y,coverage=_configured_forward(model,torch.from_numpy(x.copy())[None],stride,config,'dtr',device)
        value=y[0,0].cpu().numpy();cov=coverage.cpu().numpy() if coverage is not None else None
        return (value,cov) if return_coverage else value


def reconstruct(projections,config,dte,dtr,*,normalization=None,device='cpu',diagnostic_ideal=False):
    c=resolve_config(config);g=c['geometry'];data=np.asarray(projections,dtype=np.float32)
    if data.shape!=(len(g['angles_deg']),*g['detector_shape']) or not np.isfinite(data).all(): raise ValueError('projection shape and finite values must match geometry')
    if math.prod(g['grid']['shape'])>c['inference']['max_volume_voxels']: raise ValueError('volume exceeds configured memory limit; explicit resource adaptation required')
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if not diagnostic_ideal and dte is None: raise ValueError('DTE model required unless diagnostic_ideal is explicit')
    if c['method']=='A' and not diagnostic_ideal:
        if normalization is None or not np.isfinite([normalization.get('mean',np.nan),normalization.get('std',np.nan)]).all() or normalization['std']<=0:
            raise ValueError('A DTE requires frozen training normalization')
    if dte is not None:dte=dte.to(device).eval()
    dtr=dtr.to(device).eval()
    def stamp():
        if device.type=='cuda':torch.cuda.synchronize(device)
        return time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    start=stamp();stage=start
    with torch.inference_mode():
        extracted=[];dte_coverage=[]
        stride=2**(len(model_spec(c,'dte')[2])-1)
        for projection in data:
            if diagnostic_ideal:extracted.append(projection.copy());continue
            x=torch.from_numpy(projection.copy())[None,None]
            if c['method']=='A':x=(x-normalization['mean'])/normalization['std']
            y,coverage=_configured_forward(dte,x,stride,c,'dte',device)
            if coverage is not None:dte_coverage.append(coverage.cpu().numpy())
            if (y<0).any():raise ValueError('DTE output must be nonnegative line integrals')
            extracted.append(y[0,0].cpu().numpy())
        extracted=np.stack(extracted);end=stamp();times={'dte_seconds':end-stage};stage=end
        grid=Grid(tuple(g['grid']['shape']),tuple(g['grid']['spacing_mm']),tuple(g['grid']['origin_mm']))
        views=[circular_view(a,g['sdd_mm'],g['sid_mm'],g['detector_shape'],g['pixel_size_mm']) for a in g['angles_deg']]
        geometric=fdk(extracted,views,grid,g['angles_deg']) if c['method']=='A' else backproject_each(extracted,views,grid)
        end=stamp();times['geometry_seconds']=end-stage;stage=end
        probability,dtr_coverage=predict_geometric(geometric,c,dtr,device,return_coverage=True)
        if (probability<0).any() or (probability>1).any():raise ValueError('DTR output must be probability')
        end=stamp();times['dtr_seconds']=end-stage;times['total_seconds']=end-start
    return {'dte':extracted,'geometric':geometric,'probability':probability,'binary':(probability>=c['inference']['threshold']).astype(np.uint8),
            'grid':g['grid'],'timings':times,'diagnostic_ideal':diagnostic_ideal,
            'coverage':{'dte':np.stack(dte_coverage) if dte_coverage else None,'dtr':dtr_coverage},
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'timing_scope':'cold single-case execution includes CPU geometry and host/device transfers; excludes checkpoint loading, file I/O; not a warmed benchmark'}


def load_trained_model(checkpoint,config,stage):
    c=resolve_config(config);state=torch.load(checkpoint,map_location='cpu',weights_only=True);contract=state.get('contract',{})
    if contract.get('method')!=c['method'] or contract.get('stage')!=stage:raise ValueError('checkpoint method/stage mismatch')
    trained=contract.get('run_info',{}).get('resolved_config')
    if trained is None:raise ValueError('checkpoint lacks resolved_config; legacy checkpoint requires explicit migration')
    trained=resolve_config(trained)
    if trained['geometry']!=c['geometry']:raise ValueError('checkpoint geometry mismatch; regenerate and retrain the matching profile')
    fields=['architecture','bn_eps','bn_momentum','conv_bias','initialization',f'{stage}_activation']
    if stage=='dtr':fields+=['slice_channels' if c['method']=='A' else 'input_channels','dice_smoothing']
    if any(trained['network'][k]!=c['network'][k] for k in fields):raise ValueError('checkpoint network configuration mismatch')
    model=build_model(c,stage);model.load_state_dict(state['model'],strict=True)
    return model,state.get('normalization')


def benchmark_pipeline(projections,config,checkpoints,*,device='cpu',warmup=2,repeats=5):
    if type(warmup) is not int or type(repeats) is not int or warmup<1 or repeats<1:raise ValueError('positive warmup/repeats required')
    c=resolve_config(config);dte,norm=load_trained_model(checkpoints['dte'],c,'dte');dtr,_=load_trained_model(checkpoints['dtr'],c,'dtr')
    records=[];peak=[]
    for index in range(warmup+repeats):
        result=reconstruct(projections,c,dte,dtr,normalization=norm,device=device)
        if index>=warmup:records.append(result['timings']);peak.append(result['peak_cuda_bytes'])
    return {'device':str(device),'warmup':warmup,'repeats':repeats,'projection_shape':list(projections.shape),'grid':c['geometry']['grid'],
            'precision':'float32','samples':records,'medians':{key:float(np.median([row[key] for row in records])) for key in records[0]},
            'peak_cuda_bytes':max((v for v in peak if v is not None),default=None),
            'scope':'warmed synchronized full pipeline; includes CPU geometry and CPU/GPU stage transfers, excludes checkpoint loading and file I/O',
            'checkpoint_sha256':{key:sha256(path) for key,path in checkpoints.items()}}


def run_inference(manifest_path,config,checkpoints,output,device='cpu',frozen_inputs=None):
    c=resolve_config(config);path=Path(manifest_path).resolve();doc=json.loads(path.read_text(encoding='utf-8'));rows=doc.get('cases',[])
    if not rows or len({r.get('id') for r in rows})!=len(rows):raise ValueError('unique inference cases required')
    for row in rows:
        unknown=set(row)-{'id','entity_id','projections'}
        if unknown:raise ValueError(f'unknown inference fields (truth is forbidden): {sorted(unknown)}')
        if not isinstance(row.get('id'),str) or not row['id'] or not isinstance(row.get('projections'),str):raise ValueError('case id and projection path required')
        if 'entity_id' in row and (not isinstance(row['entity_id'],str) or not row['entity_id']):raise ValueError('entity_id must be a nonempty string')
    source_fingerprints=freeze_or_verify_manifest(path,frozen_inputs,required=c['provenance']['purpose']!='g0')
    checkpoint_fingerprints=fingerprint_files(checkpoints.values())
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('inference output must be empty')
    dte,norm=load_trained_model(checkpoints['dte'],c,'dte');dtr,_=load_trained_model(checkpoints['dtr'],c,'dtr')
    output.mkdir(parents=True,exist_ok=True);results=[]
    for index,row in enumerate(rows):
        entry={'id':row['id'],'entity_id':row.get('entity_id',row['id']),
               'entity_id_source':'manifest' if 'entity_id' in row else 'case_id_fallback'}
        try:
            projection_path=path.parent/row['projections'];p,metadata=load_projections(projection_path)
            if metadata.get('units') not in ('line_integral','synthetic_line_integral'):raise ValueError('projection units must be line_integral')
            angles=metadata.get('angles_deg');validate_case_angles(c,angles)
            validate_projection_metadata(metadata,c,angles=angles,require_calibration=c['provenance']['purpose']!='g0')
            relevant={str(path),str(projection_path.resolve()),str(projection_path.resolve())+'.json'}
            verify_fingerprints({name:digest for name,digest in source_fingerprints.items() if name in relevant})
            case_config=deepcopy(c);case_config['geometry']['angles_deg']=angles
            result=reconstruct(p,case_config,dte,dtr,normalization=norm,device=device)
            folder=output/f'case_{index:05d}';folder.mkdir()
            save_volume(folder/'probability.npz',result['probability'],{**result['grid'],'units':'probability'})
            save_volume(folder/'binary.npz',result['binary'],{**result['grid'],'units':'binary'})
            if result['coverage']['dte'] is not None:save_projections(folder/'coverage_dte.npz',result['coverage']['dte'],{'units':'tile_count','angles_deg':angles})
            if result['coverage']['dtr'] is not None:save_volume(folder/'coverage_dtr.npz',result['coverage']['dtr'],{**result['grid'],'units':'tile_count'})
            if c['inference']['retain_intermediates']:
                save_projections(folder/'dte.npz',result['dte'],{'units':'line_integral','angles_deg':angles})
                volumes=result['geometric'][None] if c['method']=='A' else result['geometric']
                for i,v in enumerate(volumes):save_volume(folder/f'geometric_{i}.npz',v,result['grid'])
            entry.update({'status':'ok','probability':str((folder/'probability.npz').resolve()),'timings':result['timings'],
                          'timing_scope':result['timing_scope'],'peak_cuda_bytes':result['peak_cuda_bytes']})
        except (ValueError,RuntimeError,OSError) as exc:
            entry.update({'status':'pipeline_failure','error':str(exc)})
        results.append(entry)
    verify_fingerprints(source_fingerprints);verify_fingerprints(checkpoint_fingerprints)
    report={'cases':results,'resolved_config':c,'source_fingerprints':source_fingerprints,'checkpoint_sha256':{s:sha256(p) for s,p in checkpoints.items()}}
    (output/'inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
