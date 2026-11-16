"""Independent experiment identities, shared held-out cases, and an honest G0 run."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from stent_repro.configuration import resolve_config
from stent_repro.provenance import write_json,sha256,manifest_fingerprints,verify_fingerprints


def dte_cache_key(config,source_fingerprints):
    network=deepcopy(config['network'])
    for key in ('slice_channels','input_channels','dtr_activation','dice_smoothing'):network.pop(key,None)
    contract={'method':config['method'],'seed':config['seed'],'network':network,'geometry':config['geometry'],
              'training':config['training']['dte'],'preprocessing':config['preprocessing'],'sources':source_fingerprints}
    return hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()


def experiment_matrix(config_a,config_b,seeds):
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):raise ValueError('unique nonnegative seeds required')
    a=resolve_config(config_a);b=resolve_config(config_b)
    if a['method']!='A' or b['method']!='B':raise ValueError('config_a/config_b methods must match')
    jobs=[]
    for seed in seeds:
        for source,key,values in ((a,'slice_channels',(1,3,5)),(b,'input_channels',(3,1))):
            for value in values:
                c=deepcopy(source);c['seed']=seed;c['network'][key]=value
                if c['method']=='B':
                    reference='original' in c['profile'] or c['profile']=='B-one-channel-ablation'
                    c['profile']=('B-one-channel-ablation' if value==1 else 'B-original-3view') if reference else ('B-adapted-one-channel' if value==1 else 'B-adapted-3view')
                c=resolve_config(c);run=f"{c['method']}_{'C' if c['method']=='A' else 'channels'}{value}_seed{seed}"
                jobs.append({'run_id':run,'config':c,'dte_reuse_key':f"{c['method']}_seed{seed}",'stages':['dte','dtr','inference','evaluation']})
    return jobs


def reference_matrix(config_a,config_b,seeds):
    a=resolve_config(config_a);b=resolve_config(config_b)
    a['profile']='A-original-4view';a['geometry']['angles_deg']=[0,45,90,135]
    b['profile']='B-original-3view';b['geometry']['angles_deg']=[0,60,120];b['network']['input_channels']=3
    jobs=experiment_matrix(a,b,seeds)
    for job in jobs:job['run_id']='REF_'+job['run_id']
    from stent_repro.configuration import ANGLE_PATTERNS
    for seed in seeds:
        for name,angles in ANGLE_PATTERNS.items():
            if not name.startswith('A-reference-'):continue
            c=deepcopy(a);c['seed']=seed;c['network']['slice_channels']=5;c['profile']=name;c['geometry']['angles_deg']=angles
            jobs.append({'run_id':f'REF_{name}_seed{seed}','config':resolve_config(c),'stages':['dte','dtr','inference','evaluation']})
    return jobs


def summarize_matrix(matrix_directory):
    """Matched entity/seed comparisons, never treating acquisitions as subjects."""
    from stent_repro.statistics import compare_training_seeds
    root=Path(matrix_directory);jobs=json.loads((root/'matrix.json').read_text(encoding='utf8'))
    saved=json.loads((root/'matrix_report.json').read_text(encoding='utf8'))
    results={r['run_id']:r for r in saved['jobs']}
    groups={};roster=dict(saved.get('expected_case_entities',{}))
    acquisition=None
    for job in jobs:
        c=job['config'];key=f"A_C{c['network']['slice_channels']}" if c['method']=='A' else f"B_channels{c['network']['input_channels']}"
        geometry=c['geometry'];current={k:geometry[k] for k in ('sdd_mm','sid_mm','detector_shape','pixel_size_mm','angles_deg')}
        if acquisition is not None and current!=acquisition:
            raise ValueError('matrix summary requires one acquisition geometry; summarize reference angle groups separately')
        acquisition=current
        if c['seed'] in groups.get(key,{}):raise ValueError('duplicate method/channel/seed in matrix summary')
        result=results.get(job['run_id'],{});rows=result.get('evaluation',{}).get('cases',[])
        if not rows and (root/job['run_id']/'evaluation.json').exists():
            planned=json.loads((root/job['run_id']/'evaluation.json').read_text(encoding='utf8'))['cases']
            rows=[{'case_id':r['id'],'entity_id':r.get('entity_id',r['id']),'status':'pipeline_failure'} for r in planned]
        groups.setdefault(key,{})[c['seed']]=rows
        for row in rows:
            case=row['case_id'];entity=row.get('entity_id',case)
            if case in roster and roster[case]!=entity:raise ValueError('matrix entity identity conflict')
            roster[case]=entity
    comparisons={}
    for ref,candidate in [('A_C5','B_channels3'),('A_C1','A_C3'),('A_C1','A_C5'),('B_channels1','B_channels3')]:
        if ref in groups and candidate in groups:
            comparisons[f'{candidate}_minus_{ref}']=compare_training_seeds(groups[ref],groups[candidate],expected_case_entities=roster)
    report={'scope':'paired entity effects; scientific interpretation depends on dataset/model provenance','comparisons':comparisons}
    write_json(root/'entity_comparisons.json',report);return report


def _entity_roster(cases,truths):
    """Reconcile independently supplied identities before freezing or evaluation."""
    if len({r['id'] for r in cases})!=len(cases) or len({r['id'] for r in truths})!=len(truths):
        raise ValueError('unique inference and truth case IDs required')
    truth_by_id={r['id']:r for r in truths}
    if {r['id'] for r in cases}!=set(truth_by_id):raise ValueError('inference and truth case IDs differ')
    roster={}
    for row in cases:
        case=row['id'];truth=truth_by_id[case]
        declared=None if row.get('entity_id_source')=='case_id_fallback' else row.get('entity_id')
        expected=truth.get('entity_id')
        for value in (declared,expected):
            if value is not None and (not isinstance(value,str) or not value):raise ValueError('entity_id must be a nonempty string')
        if declared is not None and expected is not None and declared!=expected:
            raise ValueError(f'entity identity conflict for case {case}')
        roster[case]=expected or declared or case
    return roster


def _common_test_signature(inference_path,truth_path):
    ip=Path(inference_path).resolve();tp=Path(truth_path).resolve()
    cases=json.loads(ip.read_text(encoding='utf8'))['cases'];truth_rows=json.loads(tp.read_text(encoding='utf8'))['cases']
    roster=_entity_roster(cases,truth_rows);truths={r['id']:r for r in truth_rows}
    signature=[]
    for row in sorted(cases,key=lambda r:r['id']):
        from stent_repro.provenance import fingerprint_files
        images=fingerprint_files([ip.parent/row['projections']]);truth=fingerprint_files([tp.parent/truths[row['id']]['truth']])
        signature.append([row['id'],roster[row['id']],sorted(str(v) for v in images.values()),sorted(str(v) for v in truth.values())])
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def _evaluation_document(inference_result,truth_path,threshold,allow_empty=False):
    p=Path(truth_path).resolve();rows=json.loads(p.read_text(encoding='utf8'))['cases'];truths={r['id']:r for r in rows}
    roster=_entity_roster(inference_result['cases'],rows)
    cases=[]
    for prediction in inference_result['cases']:
        row={k:v for k,v in prediction.items() if k in ('id','entity_id','status','error','probability')}
        row['entity_id']=roster[prediction['id']]
        row['truth']=str((p.parent/truths[prediction['id']]['truth']).resolve());cases.append(row)
    return {'threshold':threshold,'allow_empty_truth':allow_empty,'cases':cases}


def run_matrix(matrix,manifests,output,execute=False):
    from stent_repro.workflow import train_workflow
    from stent_repro.inference import run_inference
    from stent_repro.evaluation import evaluate_manifest
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('matrix output must be empty')
    if not matrix or len({j['run_id'] for j in matrix})!=len(matrix):raise ValueError('unique jobs required')
    frozen={}
    if execute:
        signatures={_common_test_signature(m['inference'],m['truth']) for m in manifests.values()}
        if len(signatures)!=1:raise ValueError('methods must share identical held-out projections, truth, and case identities')
        geometries=[resolve_config(j['config'])['geometry'] for j in matrix]
        for g in geometries[1:]:
            if any(g[k]!=geometries[0][k] for k in ('sdd_mm','sid_mm','detector_shape','pixel_size_mm','angles_deg')):
                raise ValueError('task comparison must share acquisition geometry')
        frozen={method:{key:manifest_fingerprints(m[key]) for key in ('inference','truth')} for method,m in manifests.items()}
    output.mkdir(parents=True,exist_ok=True);write_json(output/'matrix.json',matrix)
    if execute:write_json(output/'frozen_test_inputs.json',frozen)
    planned={}
    if execute:
        first=next(iter(manifests.values()))
        rows=json.loads(Path(first['inference']).read_text(encoding='utf8'))['cases']
        truth_rows=json.loads(Path(first['truth']).read_text(encoding='utf8'))['cases']
        planned=_entity_roster(rows,truth_rows)
    report={'executed':execute,'jobs':[],'expected_case_entities':planned,'formal_reproduction_claim':False};cache={}
    if not execute:return report
    for job in matrix:
        c=resolve_config(job['config']);m=manifests[c['method']];folder=output/job['run_id'];folder.mkdir()
        item={'run_id':job['run_id']}
        try:
            for expected in frozen[c['method']].values():verify_fingerprints(expected)
            cache_key=dte_cache_key(c,manifest_fingerprints(m['dte']))
            if cache_key not in cache:
                train_workflow(c,m['dte'],folder/'dte','dte',m.get('device','cpu'),preflight_report=m.get('preflight',{}).get('dte'));cache[cache_key]=folder/'dte'/'best.pt'
            train_workflow(c,m['dtr'],folder/'dtr','dtr',m.get('device','cpu'),preflight_report=m.get('preflight',{}).get(job['run_id'],{}).get('dtr'))
            inference=run_inference(m['inference'],c,{'dte':cache[cache_key],'dtr':folder/'dtr'/'best.pt'},folder/'inference',m.get('device','cpu'),frozen_inputs=frozen[c['method']]['inference'])
            verify_fingerprints(frozen[c['method']]['truth'])
            write_json(folder/'evaluation.json',_evaluation_document(inference,m['truth'],c['inference']['threshold'],c['evaluation']['allow_empty_truth']))
            score=evaluate_manifest(folder/'evaluation.json',folder/'evaluation')
            verify_fingerprints(frozen[c['method']]['truth'])
            item.update({'status':'completed','evaluation':score,'dte_checkpoint_sha256':sha256(cache[cache_key])})
        except (ValueError,RuntimeError,OSError,KeyError) as exc:item.update({'status':'pipeline_failure','error':str(exc)})
        report['jobs'].append(item);write_json(output/'matrix_report.json',report)
    return report


def smoke(output,seed=42):
    from stent_repro.workflow import prepare_workflow,train_workflow
    from stent_repro.simulation import write_simulated_dataset
    from stent_repro.inference import run_inference,load_trained_model,reconstruct
    from stent_repro.evaluation import evaluate_manifest,segmentation_metrics
    from stent_repro.io import load_projections,load_volume,save_volume
    from stent_repro.preflight import preflight
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('smoke output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    report={'scope':'G0_compact_synthetic_workflow','g1_complete':False,'g2_complete':False,'methods':{},
            'limitations':['compact networks and small synthetic data','no useful accuracy or generalization claim','not original patient/model data','no full B-DTE training or GPU benchmark']}
    common_inference=None;common_truth=None
    for method in ('A','B'):
        c=resolve_config({'method':method,'profile':f'{method}-adapted-3view','seed':seed,
            'geometry':{'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[16,16],'angles_deg':[0,30,60],
                        'grid':{'shape':[16]*3,'spacing_mm':[1]*3,'origin_mm':[-7.5]*3}},
            'network':{'architecture':'compact-g0'},
            'training':{stage:{'epochs':2,'batch_size':2,'train_pairs':8,'validation_pairs':4,
                         'patch_shape':[16]*(3 if method=='B' and stage=='dtr' else 2)} for stage in ('dte','dtr')},
            'preprocessing':{'blur_sigma_px':.3},'provenance':{'purpose':'g0','dataset_version':'synthetic-g0-v1'}})
        folder=output/method;folder.mkdir()
        raw=write_simulated_dataset(c,folder/'data',[3,1,1]);prep=prepare_workflow(c,raw,folder/'prepared')
        if common_inference is None:common_inference=raw.parent/'inference.json';common_truth=raw.parent/'test_truth.json'
        training={};resources={}
        for stage in ('dte','dtr'):
            resources[stage]=preflight(c,stage=stage,device='cpu',measure=True)
            train_manifest=prep['dte_pool_manifest'] if stage=='dte' else prep['dtr_manifest']
            training[stage]=train_workflow(c,train_manifest,folder/stage,stage,'cpu')
        checkpoints={s:folder/s/'best.pt' for s in ('dte','dtr')}
        inference=run_inference(common_inference,c,checkpoints,folder/'inference')
        write_json(folder/'evaluation.json',_evaluation_document(inference,common_truth,.5))
        evaluated=evaluate_manifest(folder/'evaluation.json',folder/'evaluation')
        # Separate diagnostic bypasses DTE and cannot replace measured-input results.
        dtr,_=load_trained_model(checkpoints['dtr'],c,'dtr');diagnostics=[]
        for row in json.loads(common_truth.read_text(encoding='utf8'))['cases']:
            ideal,_=load_projections(common_truth.parent/row['ideal']);truth,meta=load_volume(common_truth.parent/row['truth'])
            result=reconstruct(ideal,c,None,dtr,diagnostic_ideal=True)
            save_volume(folder/'ideal_diagnostic_probability.npz',result['probability'],result['grid'])
            diagnostics.append({'case_id':row['id'],**segmentation_metrics(result['binary'],truth,{**meta,'shape':list(truth.shape)})})
        report['methods'][method]={'training':training,'inference_case_count':len(inference['cases']),
            'pipeline_failure_count':sum(r['status']!='ok' for r in inference['cases']),'evaluation':evaluated,
            'ideal_projection_diagnostic':diagnostics,'resource_checks':{s:{k:v for k,v in r.items() if k!='architecture'} for s,r in resources.items()}}
        write_json(output/'report.json',report)
    report['common_test_signature']=_common_test_signature(common_inference,common_truth)
    report['software_chain_passed']=all(r['pipeline_failure_count']==0 for r in report['methods'].values())
    write_json(output/'report.json',report)
    return report
