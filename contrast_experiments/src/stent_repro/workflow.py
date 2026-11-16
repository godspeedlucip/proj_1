"""Version 2 reproducible workflow CLI; legacy prepared training stays separate."""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import random
import numpy as np
import torch
import yaml
from stent_repro.configuration import resolve_config,validate_projection_metadata,validate_case_angles
from stent_repro.evaluation import validate_grid,segmentation_metrics,evaluate_manifest
from stent_repro.io import load_volume,load_projections
from stent_repro.model_registry import build_model,architecture_report,model_spec
from stent_repro.preprocessing import prepare_samples
from stent_repro.provenance import manifest_fingerprints,verify_fingerprints,environment_record,write_json
from stent_repro.training.data import PreparedDataset,read_training_manifest
from stent_repro.training.online import OnlineDTEDataset
from stent_repro.training.engine import train_prepared


def _eligibility(c):
    if c['provenance']['purpose']!='g0':
        if not c['provenance']['geometry_calibrated']:raise ValueError('formal training requires calibrated geometry; example geometry is G0 only')
        if c['provenance']['dataset_version'] in ('unspecified','REPLACE_WITH_FROZEN_DATA_VERSION'):raise ValueError('formal training requires frozen dataset_version')
        if not c['geometry']['calibration_id']:raise ValueError('formal training requires a traceable geometry calibration_id')


def prepare_workflow(config,manifest,output):
    c=resolve_config(config);path=Path(manifest).resolve();doc=json.loads(path.read_text(encoding='utf8'));g=c['geometry'];p=c['preprocessing']
    fingerprints=manifest_fingerprints(path)
    rows=doc.get('cases',[])
    if not rows:raise ValueError('raw cases required')
    for row in rows:
        truth,meta=load_volume(path.parent/row['truth'])
        for key,value in row.get('truth_grid',{}).items():
            if key in meta and not np.allclose(meta[key],value):raise ValueError('raw grid declarations conflict')
            meta[key]=value
        actual=validate_grid({**meta,'shape':list(truth.shape)})
        if actual!=validate_grid(g['grid']):raise ValueError('raw truth grid does not match resolved geometry grid')
        bg,bm=load_projections(path.parent/row['background'])
        background_angles=row.get('background_angles_deg',bm.get('angles_deg'))
        validate_projection_metadata({**bm,'angles_deg':background_angles},c,angles=background_angles,require_calibration=c['provenance']['purpose']!='g0')
        if row.get('background_units',bm.get('units'))!='line_integral':raise ValueError('v2 requires calibrated line_integral backgrounds; use explicit calibration before preparation')
        if tuple(bg.shape[1:])!=tuple(g['detector_shape']):raise ValueError('background detector shape mismatch')
        validate_case_angles(c,row.get('angles_deg'))
    patch=c['training']['dte']['patch_shape']
    if patch[0]!=patch[1]:raise ValueError('reference preparation currently requires square DTE patches')
    params={k:g[k] for k in ('sdd_mm','sid_mm','pixel_size_mm','detector_shape','calibration_id')}
    params.update({k:p[k] for k in ('attenuation_per_mm','integration_step_mm','blur_sigma_px','min_background','false_negative_rate','false_positive_rate','false_positive_scale')})
    params.update({k:p[k] for k in ('spectral','error_model','correlation_sigma_px')})
    params.update({'photons':float(np.mean(p['photons_range'])),'scatter_fraction':float(np.mean(p['scatter_range'])),'patch_size':patch[0],'seed':c['seed']})
    if p['dtr_crop_shape'] is not None:params['dtr_crop_shape']=p['dtr_crop_shape']
    all_background=all(r['projection_role']=='patient_without_tool' for r in rows)
    result=prepare_samples(path,output,method=c['method'],profile=c['profile'],params=params,export_pools=all_background)
    output=Path(output);(output/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf8')
    write_json(output/'workflow_provenance.json',{'source_fingerprints':fingerprints,'resolved_config':c,
        'environment':environment_record(),'data_mode':'online_independent_pools' if all_background else 'paired_measured_or_mixed_surrogate',
        'physics':{'projection_source':'native_continuous_tool' if all(r.get('ideal') for r in rows) else 'projected_truth_or_mixed',
                   'spectrum':p['spectral'],'b_error_model':p['error_model'] if c['method']=='B' else None},
        'limitations':['not author data or numerical reproduction','native source calibration and synthetic intensity mapping remain source-specific']})
    return result


class DTRValidationPanel:
    def __init__(self,rows,c):self.rows=[r for r in rows if r['split']=='validation'];self.config=c
    def evaluate(self,model,stats,device):
        from stent_repro.inference import predict_geometric
        scores=[]
        for row in self.rows:
            target,meta=load_volume(row['target'])
            geometric=load_volume(row['input'])[0] if self.config['method']=='A' else np.stack([load_volume(p)[0] for p in row['input']])
            prediction=predict_geometric(geometric,self.config,model,device)
            metrics=segmentation_metrics(prediction>=self.config['inference']['threshold'],target,{**meta,'shape':list(target.shape)})
            # Empty truth is unsuitable for the positive-device selection panel.
            if metrics['dice'] is None:raise ValueError('DTR validation panel needs nonempty device labels')
            scores.append(metrics['dice'])
        if not scores:raise ValueError('empty fixed validation panel')
        return float(np.mean(scores))


def train_workflow(config,manifest,output,stage,device='cpu',resume=None,preflight_report=None):
    c=resolve_config(config)
    if c['provenance']['purpose']!='g0' and preflight_report is None:raise ValueError('formal training requires a matching measured preflight report before model allocation')
    _eligibility(c)
    if stage not in ('dte','dtr'):raise ValueError('invalid stage')
    resource_evidence=None
    if preflight_report is not None:
        from stent_repro.preflight import validate_resource_report
        resource_evidence=validate_resource_report(c,stage,device,preflight_report)
    p=Path(manifest).resolve();out=Path(output);t=c['training'][stage]
    if resume:
        old=json.loads((out/'provenance.json').read_text(encoding='utf8'));verify_fingerprints(old['source_fingerprints'])
    elif out.exists() and any(out.iterdir()):raise FileExistsError('training output must be empty')
    if any(t[k] is None for k in ('epochs','train_pairs','validation_pairs')):raise ValueError('explicit epochs, train_pairs and validation_pairs required')
    fingerprints=manifest_fingerprints(p);doc=json.loads(p.read_text(encoding='utf8'))
    if (p.parent/'resolved_config.yaml').exists():
        prepared=resolve_config(p.parent/'resolved_config.yaml')
        if prepared['method']!=c['method'] or prepared['geometry']!=c['geometry'] or prepared['preprocessing']!=c['preprocessing']:
            raise ValueError('prepared data geometry/method/preprocessing differs from resolved config')
    selection=None
    if stage=='dte' and 'pools' in doc:
        rows=deepcopy(doc['pools'])
        for row in rows:
            row['path']=str((p.parent/row['path']).resolve());_,metadata=load_projections(row['path'])
            if metadata.get('units')!='line_integral':raise ValueError('online pool units must be line_integral')
        params={**c['preprocessing'],'patch_shape':t['patch_shape']}
        train=OnlineDTEDataset(rows,'train',params,c['seed']);val=OnlineDTEDataset(rows,'validation',params,c['seed'])
        data_mode='online_independent_pools'
    else:
        rows=read_training_manifest(p,c['method'],stage,profile=c['profile'] if stage=='dtr' else None)
        if stage=='dtr':
            for row in rows:
                target,meta=load_volume(row['target']);grid=validate_grid({**meta,'shape':list(target.shape)})
                if grid['spacing_mm']!=c['geometry']['grid']['spacing_mm']:raise ValueError('DTR grid spacing mismatch')
                for filename in row['input'] if isinstance(row['input'],list) else [row['input']]:
                    v,vm=load_volume(filename)
                    if validate_grid({**vm,'shape':list(v.shape)})!=grid:raise ValueError('DTR physical grid mismatch')
            selection=DTRValidationPanel(rows,c)
        kw={'slice_channels':c['network']['slice_channels'],'input_channels':c['network']['input_channels'],'patch_shape':t['patch_shape']}
        train=PreparedDataset(rows,'train',c['method'],stage,augment=c['method']=='B' and stage=='dtr',random_crop=True,**kw)
        val=PreparedDataset(rows,'validation',c['method'],stage,**kw)
        data_mode='prepared_volumes' if stage=='dtr' else 'paired_surrogate_not_online_pools'
    stride=2**(len(model_spec(c,stage)[2])-1)
    if any(n<stride or n%stride for n in t['patch_shape']):raise ValueError('training patch dimensions must be multiples of network stride')
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    model=build_model(c,stage)
    out.mkdir(parents=True,exist_ok=True)
    info={'resolved_config':c,'source_fingerprints':fingerprints,'data_mode':data_mode,'mode':'g0' if c['provenance']['purpose']=='g0' else 'adapted_or_reference_reimplementation'}
    record={**info,'environment':environment_record(),'resource_evidence':resource_evidence,'completion_claim':'training_only; no automatic G1/G2 certification'}
    if not resume:
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf8')
        write_json(out/'provenance.json',record);write_json(out/'split_manifest.json',{'source':str(p),'rows':rows})
        write_json(out/'architecture.json',architecture_report(c,stage))
    result=train_prepared(train,val,model,out,method=c['method'],stage=stage,epochs=t['epochs'],batch_size=t['batch_size'],
        learning_rate=t['learning_rate'],seed=c['seed'],device=device,resume=resume,run_info=info,
        train_pairs=t['train_pairs'],validation_pairs=t['validation_pairs'],normalization_mode='running' if c['method']=='A' and stage=='dte' else 'none',
        selection_dataset=selection,selection_metric=t['selection_metric'],dice_smoothing=c['network']['dice_smoothing'],
        precision=t['precision'],activation_checkpointing=t['activation_checkpointing'])
    if resume:
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf8')
        write_json(out/'provenance.json',record)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('validate','prepare','train','infer','preflight','simulate','benchmark','geometry-check'):
        s=sub.add_parser(name);s.add_argument('--config',required=True)
        if name in ('prepare','train','infer'):s.add_argument('--manifest',required=True)
        if name not in ('validate','preflight'):s.add_argument('--output',required=True)
        if name in ('train','infer','preflight','benchmark'):s.add_argument('--device',default='cpu')
        if name=='train':s.add_argument('--stage',choices=['dte','dtr'],required=True);s.add_argument('--resume');s.add_argument('--preflight-report')
        if name=='infer':s.add_argument('--dte',required=True);s.add_argument('--dtr',required=True);s.add_argument('--frozen-inputs')
        if name=='preflight':s.add_argument('--stage',choices=['dte','dtr']);s.add_argument('--measure',action='store_true');s.add_argument('--output')
        if name=='simulate':s.add_argument('--counts',nargs=3,type=int,default=[2,1,1])
        if name=='benchmark':
            s.add_argument('--stage',choices=['dte','dtr'],required=True);s.add_argument('--tile-shape',nargs='+',type=int)
            s.add_argument('--overlap',nargs='+',type=int);s.add_argument('--warmup',type=int,default=2);s.add_argument('--repeats',type=int,default=5)
    s=sub.add_parser('evaluate');s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('freeze');s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('matrix');s.add_argument('--config-a',required=True);s.add_argument('--config-b',required=True);s.add_argument('--seeds',nargs='+',type=int,default=[17,42,73]);s.add_argument('--output',required=True)
    s.add_argument('--reference',action='store_true')
    s=sub.add_parser('run-matrix');s.add_argument('--matrix',required=True);s.add_argument('--manifests',required=True);s.add_argument('--output',required=True);s.add_argument('--execute',action='store_true')
    s=sub.add_parser('summarize-matrix');s.add_argument('--directory',required=True)
    s=sub.add_parser('evaluate-dte');s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('calibrate-errors');s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('framework-probe');s.add_argument('--output',required=True);s.add_argument('--require-tensorflow',action='store_true')
    s=sub.add_parser('import-nerve');s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('smoke');s.add_argument('--output',required=True);s.add_argument('--seed',type=int,default=42)
    a=parser.parse_args(argv)
    if a.command=='validate':result=resolve_config(a.config)
    elif a.command=='prepare':result=prepare_workflow(a.config,a.manifest,a.output)
    elif a.command=='train':result=train_workflow(a.config,a.manifest,a.output,a.stage,a.device,a.resume,a.preflight_report)
    elif a.command=='infer':
        from stent_repro.inference import run_inference
        result=run_inference(a.manifest,a.config,{'dte':a.dte,'dtr':a.dtr},a.output,a.device,a.frozen_inputs)
    elif a.command=='evaluate':result=evaluate_manifest(a.manifest,a.output)
    elif a.command=='evaluate-dte':
        from stent_repro.evaluation import evaluate_dte_manifest
        result=evaluate_dte_manifest(a.manifest,a.output)
    elif a.command=='calibrate-errors':
        from stent_repro.projection_errors import ProjectionResidualPair,calibrate_correlated_projection_errors
        from stent_repro.provenance import fingerprint_files
        path=Path(a.manifest).resolve();doc=json.loads(path.read_text(encoding='utf8'));pairs=[];sources=[path]
        for row in doc['cases']:
            clean=path.parent/row['clean'];observed=path.parent/row['observed'];sources.extend([clean,observed])
            pairs.append(ProjectionResidualPair(row['split'],load_projections(clean)[0],load_projections(observed)[0]))
        fit=calibrate_correlated_projection_errors(pairs,support_threshold=doc.get('support_threshold',0.))
        result={'parameters':fit.parameters,'metadata':fit.metadata,'source_fingerprints':fingerprint_files(sources)};write_json(a.output,result)
    elif a.command=='framework-probe':
        from stent_repro.framework_probe import run_framework_probe
        result=run_framework_probe(require_tensorflow=a.require_tensorflow);write_json(a.output,result)
    elif a.command=='import-nerve':
        from stent_repro.nerve_adapter import NerveCase,import_nerve_dataset
        path=Path(a.manifest).resolve();doc=json.loads(path.read_text(encoding='utf8'));cases=[]
        for source in doc['cases']:
            row=dict(source)
            for key in ('head_dir','stent_projection_dir','stack_dir','model_dir'):row[key]=path.parent/row[key]
            cases.append(NerveCase(**row))
        result={k:str(v) for k,v in import_nerve_dataset(cases,a.output,angles_deg=doc['angles_deg'],target_grid=doc.get('target_grid'),engineering_holdout=doc.get('engineering_holdout',False)).items()}
    elif a.command=='geometry-check':
        from stent_repro.geometry_checks import geometry_suite
        result=geometry_suite(a.config);write_json(a.output,result)
    elif a.command=='benchmark':
        from stent_repro.preflight import benchmark_stage
        result=benchmark_stage(a.config,stage=a.stage,device=a.device,tile_shape=a.tile_shape,overlap=a.overlap,warmup=a.warmup,repeats=a.repeats);write_json(a.output,result)
    elif a.command=='freeze':
        if Path(a.output).exists():raise FileExistsError('frozen input contract already exists; use a new version')
        result=manifest_fingerprints(a.manifest);write_json(a.output,result)
    elif a.command=='simulate':
        from stent_repro.simulation import write_simulated_dataset
        result={'manifest':str(write_simulated_dataset(resolve_config(a.config),a.output,a.counts))}
    elif a.command=='preflight':
        from stent_repro.preflight import preflight
        result=preflight(resolve_config(a.config),stage=a.stage,device=a.device,measure=a.measure)
        if a.output:write_json(a.output,result)
    elif a.command=='matrix':
        from stent_repro.experiments import experiment_matrix,reference_matrix
        result=(reference_matrix if a.reference else experiment_matrix)(resolve_config(a.config_a),resolve_config(a.config_b),a.seeds);write_json(a.output,result)
    elif a.command=='run-matrix':
        from stent_repro.experiments import run_matrix
        result=run_matrix(json.loads(Path(a.matrix).read_text(encoding='utf8')),json.loads(Path(a.manifests).read_text(encoding='utf8')),a.output,execute=a.execute)
    elif a.command=='summarize-matrix':
        from stent_repro.experiments import summarize_matrix
        result=summarize_matrix(a.directory)
    else:
        from stent_repro.experiments import smoke
        result=smoke(a.output,a.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result


if __name__=='__main__':main()
