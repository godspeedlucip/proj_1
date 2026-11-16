import json
from pathlib import Path
import numpy as np
import pytest
from stent_repro.configuration import resolve_config
from stent_repro.simulation import write_simulated_dataset
from stent_repro.workflow import prepare_workflow
from stent_repro.io import load_projections,save_projections


def reference_config(workflow_config):
    c=workflow_config();c['profile']='A-original-4view';c['geometry']['angles_deg']=[0,45,90,135]
    c['simulation']={'angular_offset_mode':'random_uniform','start_angle_range_deg':[0,180],
                     'dte_views':'a_17x10','material_density_range_g_cm3':None}
    return resolve_config(c)


def test_reference_random_angles_and_separate_17_view_pool(tmp_path,workflow_config):
    c=reference_config(workflow_config)
    p=write_simulated_dataset(c,tmp_path/'data',[2,1,1]);rows=json.loads(p.read_text(encoding='utf8'))['cases']
    assert len({round(r['angles_deg'][0],8) for r in rows})==len(rows)
    assert all(np.allclose(np.diff(r['angles_deg']),45) for r in rows)
    bg,meta=load_projections(p.parent/rows[0]['background'])
    assert len(bg)==17 and np.allclose(np.diff(meta['angles_deg']),10)
    result=prepare_workflow(c,p,tmp_path/'prepared')
    pools=json.loads(Path(result['dte_pool_manifest']).read_text(encoding='utf8'))['pools']
    assert len(load_projections(Path(result['dte_pool_manifest']).parent/pools[0]['path'])[0])==17
    p2=write_simulated_dataset(c,tmp_path/'repeat',[2,1,1])
    assert [r['angles_deg'] for r in rows]==[r['angles_deg'] for r in json.loads(p2.read_text(encoding='utf8'))['cases']]


@pytest.mark.parametrize('profile,angles',[
    ('A-reference-2view90',[12,102]),('A-reference-3view60',[12,72,132]),
    ('A-reference-4view30',[12,42,72,102]),('A-reference-4view22_5',[12,34.5,57,79.5])])
def test_reference_angle_ablation_configs_are_consumed(tmp_path,workflow_config,profile,angles):
    c=workflow_config();c['profile']=profile;c['geometry']['angles_deg']=angles;c=resolve_config(c)
    p=write_simulated_dataset(c,tmp_path/'data',[1,1,1])
    result=prepare_workflow(c,p,tmp_path/'prep')
    rows=json.loads(Path(result['dtr_manifest']).read_text(encoding='utf8'))['samples']
    assert all(row['angles_deg']==angles for row in rows)


def test_python310_file_fingerprint_fallback(tmp_path,monkeypatch):
    import hashlib
    from stent_repro.provenance import sha256
    p=tmp_path/'x';p.write_bytes(b'content')
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    assert sha256(p)==hashlib.sha256(b'content').hexdigest()


def test_native_tool_targets_are_not_recomputed_with_uniform_attenuation(tmp_path,workflow_config):
    c=workflow_config();p=write_simulated_dataset(c,tmp_path/'data',[1,1,1])
    doc=json.loads(p.read_text(encoding='utf8'))
    for i,row in enumerate(doc['cases']):
        filename=f'native_{i}.npz'
        save_projections(p.parent/filename,np.full((3,16,16),.7,np.float32),
                         {'units':'line_integral','angles_deg':[0,30,60],'geometry':c['geometry']})
        row.update(ideal=filename,registration_confirmed=True)
    p.write_text(json.dumps(doc),encoding='utf8')
    result=prepare_workflow(c,p,tmp_path/'prep')
    targets=json.loads(Path(result['dte_manifest']).read_text(encoding='utf8'))['samples']
    y,_=load_projections(Path(result['dte_manifest']).parent/targets[0]['target'])
    assert np.allclose(y,.7)


def test_matrix_statistics_preserve_planned_cases_when_every_run_fails(tmp_path,workflow_config):
    from stent_repro.experiments import experiment_matrix,summarize_matrix
    jobs=experiment_matrix(workflow_config(),workflow_config('B'),[17,42,73])
    (tmp_path/'matrix.json').write_text(json.dumps(jobs),encoding='utf8')
    (tmp_path/'matrix_report.json').write_text(json.dumps({'expected_case_entities':{'x':'stent1','y':'stent2'},
        'jobs':[{'run_id':j['run_id'],'status':'pipeline_failure'} for j in jobs]}),encoding='utf8')
    result=summarize_matrix(tmp_path)
    summary=result['comparisons']['B_channels3_minus_A_C5']
    assert summary['fixed_seed_average']['expected_entity_count']==2


def test_native_head_source_is_frozen(tmp_path):
    from stent_repro.provenance import manifest_fingerprints,verify_fingerprints
    head=tmp_path/'head.npy';np.save(head,np.ones((1,2,2)))
    manifest=tmp_path/'raw.json';manifest.write_text(json.dumps({'cases':[{'head_background':'head.npy'}]}))
    frozen=manifest_fingerprints(manifest)
    np.save(head,np.zeros((1,2,2)))
    with pytest.raises(ValueError,match='changed'):verify_fingerprints(frozen)


def test_random_acquisition_angles_follow_saved_dte(tmp_path,workflow_config,monkeypatch):
    import stent_repro.inference as module
    c=reference_config(workflow_config)
    raw=write_simulated_dataset(c,tmp_path/'data',[1,1,1])
    manifest=raw.parent/'inference.json'
    row=json.loads(manifest.read_text())['cases'][0]
    _,expected=load_projections(manifest.parent/row['projections'])
    checkpoint=tmp_path/'weights';checkpoint.write_bytes(b'test fixture')
    monkeypatch.setattr(module,'load_trained_model',lambda *args:(None,None))
    def reconstruction(p,case_config,*args,**kwargs):
        shape=tuple(c['geometry']['grid']['shape'])
        return {'probability':np.zeros(shape),'binary':np.zeros(shape),'geometric':np.zeros(shape),
                'dte':p,'grid':c['geometry']['grid'],'timings':{},'timing_scope':'fixture',
                'peak_cuda_bytes':None,'coverage':{'dte':None,'dtr':None}}
    monkeypatch.setattr(module,'reconstruct',reconstruction)
    report=module.run_inference(manifest,c,{'dte':checkpoint,'dtr':checkpoint},tmp_path/'inference')
    assert report['cases'][0]['status']=='ok'
    _,actual=load_projections(tmp_path/'inference/case_00000/dte.npz')
    assert actual['angles_deg']==expected['angles_deg']
