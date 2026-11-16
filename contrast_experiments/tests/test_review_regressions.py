import json
from pathlib import Path
import numpy as np
import pytest
import torch
from stent_repro.io import save_projections,save_volume
from stent_repro.provenance import fingerprint_files,manifest_fingerprints,sha256
from stent_repro.inference import run_inference
from stent_repro.evaluation import evaluate_manifest
from stent_repro.training.engine import train_prepared
from stent_repro.workflow import train_workflow


class ZeroDTE(torch.nn.Module):
    def forward(self,x):return torch.zeros_like(x[:,:1])


def inference_input(tmp_path,c,extra=None):
    meta={'units':'line_integral','angles_deg':[0,30,60],**(extra or {})}
    save_projections(tmp_path/'projection.npy',np.zeros((3,16,16),np.float32),meta)
    p=tmp_path/'infer.json';p.write_text(json.dumps({'cases':[{'id':'c1','projections':'projection.npy'}]}),encoding='utf8')
    return p


def fake_loader(monkeypatch):
    monkeypatch.setattr('stent_repro.inference.load_trained_model',lambda *args:(ZeroDTE(),{'mean':0,'std':1}))


def test_inference_rejects_acquisition_metadata_conflict(tmp_path,workflow_config,monkeypatch):
    c=workflow_config('B');p=inference_input(tmp_path,c,{'geometry':{'sdd_mm':9999},'pixel_spacing_mm':[999,999]})
    fake_loader(monkeypatch)
    report=run_inference(p,c,{'dte':p,'dtr':p},tmp_path/'out')
    assert report['cases'][0]['status']=='pipeline_failure'
    assert 'geometry' in report['cases'][0]['error'] or 'spacing' in report['cases'][0]['error']


def test_frozen_inference_contract_detects_replaced_projection(tmp_path,workflow_config,monkeypatch):
    c=workflow_config('B');p=inference_input(tmp_path,c);frozen=manifest_fingerprints(p)
    fake_loader(monkeypatch)
    np.save(tmp_path/'projection.npy',np.ones((3,16,16),np.float32))
    with pytest.raises(ValueError,match='changed'):
        run_inference(p,c,{'dte':p,'dtr':p},tmp_path/'out',frozen_inputs=frozen)


@pytest.mark.parametrize('extension',['npy','npz'])
def test_corrupt_array_keeps_evaluation_denominator(tmp_path,extension):
    g={'spacing_mm':[1]*3,'origin_mm':[0]*3,'direction':np.eye(3).tolist()}
    gt=np.zeros((3,3,3),np.uint8);gt[1,1,1]=1
    save_volume(tmp_path/'gt.npz',gt,g)
    bad=tmp_path/f'bad.{extension}'
    if extension=='npy':bad.write_bytes(b'')
    else:bad.write_bytes(b'PK\x03\x04broken')
    p=tmp_path/'eval.json';p.write_text(json.dumps({'cases':[{'id':'ok','probability':'gt.npz','truth':'gt.npz'},{'id':'bad','probability':bad.name,'truth':'gt.npz'}]}),encoding='utf8')
    r=evaluate_manifest(p,tmp_path/'out')
    assert r['case_count']==2 and r['failure_count']==1


def test_legacy_contract_migration_preserves_best_checkpoint(tmp_path):
    class DS:
        def __len__(self):return 2
        def __getitem__(self,i):return torch.zeros(1,4,4),torch.zeros(1,4,4)
    def model():
        m=torch.nn.Conv2d(1,1,1);torch.nn.init.zeros_(m.weight);torch.nn.init.zeros_(m.bias);return m
    kw=dict(method='B',stage='dte',batch_size=2,learning_rate=.001,seed=1,device='cpu')
    train_prepared(DS(),DS(),model(),tmp_path,epochs=1,**kw)
    state=torch.load(tmp_path/'last.pt',weights_only=True)
    for key in ('train_pairs','validation_pairs','normalization_mode','selection_metric','dice_smoothing'):state['contract'].pop(key)
    state.pop('best_selection');torch.save(state,tmp_path/'last.pt')
    before=sha256(tmp_path/'best.pt')
    train_prepared(DS(),DS(),model(),tmp_path,epochs=2,resume=tmp_path/'last.pt',**kw)
    assert sha256(tmp_path/'best.pt')==before


def test_formal_training_requires_resource_evidence_before_read_or_allocate(tmp_path,workflow_config):
    c=workflow_config('B');c['network']['architecture']='paper';c['provenance']['purpose']='adapted'
    c['provenance']['geometry_calibrated']=True
    with pytest.raises(ValueError,match='preflight'):
        train_workflow(c,tmp_path/'missing.json',tmp_path/'out','dte','cpu')


def test_cache_identity_changes_with_dte_training_or_source(workflow_config):
    from stent_repro.experiments import dte_cache_key
    c=workflow_config();a=dte_cache_key(c,{'x':'digest-a'})
    c['network']['slice_channels']=1
    assert dte_cache_key(c,{'x':'digest-a'})==a
    c['training']['dte']['train_pairs']+=1
    assert dte_cache_key(c,{'x':'digest-a'})!=a
    assert dte_cache_key(c,{'x':'digest-b'})!=dte_cache_key(c,{'x':'digest-a'})


def test_resource_contract_is_stage_specific_and_rejects_shape_changes(tmp_path,workflow_config):
    from stent_repro.preflight import resource_contract,validate_resource_report
    c=workflow_config()
    report={'resource_contract':resource_contract(c,'dte','cpu'),'full_step_measured':True,
            'gradient_finite':True,'optimizer_state_tensors':10,'formal_task_ready':True}
    path=tmp_path/'resource.json';path.write_text(json.dumps(report),encoding='utf8')
    c['network']['slice_channels']=1
    # One DTE resource measurement remains valid for the C=1/3/5 DTR arms.
    validate_resource_report(c,'dte','cpu',path)
    c['training']['dte']['batch_size']+=1
    with pytest.raises(ValueError,match='preflight report does not match'):
        validate_resource_report(c,'dte','cpu',path)
    with pytest.raises(ValueError,match='preflight report does not match'):
        validate_resource_report(c,'dtr','cpu',path)


@pytest.mark.parametrize('extension',['npy','npz'])
def test_corrupt_projection_is_a_reported_inference_failure(tmp_path,extension,workflow_config,monkeypatch):
    c=workflow_config('B');p=inference_input(tmp_path,c);fake_loader(monkeypatch)
    bad=tmp_path/f'bad.{extension}';bad.write_bytes(b'' if extension=='npy' else b'PK\x03\x04broken')
    p.write_text(json.dumps({'cases':[{'id':'corrupt','projections':bad.name}]}),encoding='utf8')
    r=run_inference(p,c,{'dte':p,'dtr':p},tmp_path/'out')
    assert len(r['cases'])==1 and r['cases'][0]['status']=='pipeline_failure'
    assert 'corrupt' in r['cases'][0]['error']


def test_prepare_rejects_declared_acquisition_conflict_before_writing(tmp_path,workflow_config):
    from stent_repro.workflow import prepare_workflow
    c=workflow_config()
    save_volume(tmp_path/'truth.npz',np.ones((16,16,16),np.uint8),c['geometry']['grid'])
    save_projections(tmp_path/'background.npz',np.ones((3,16,16),np.float32),
                     {'angles_deg':[0,30,60],'units':'line_integral','geometry':{'sid_mm':999}})
    manifest=tmp_path/'raw.json'
    manifest.write_text(json.dumps({'cases':[{'id':'one','truth':'truth.npz','background':'background.npz',
        'angles_deg':[0,30,60],'projection_role':'patient_without_tool'}]}),encoding='utf8')
    with pytest.raises(ValueError,match='geometry conflict'):
        prepare_workflow(c,manifest,tmp_path/'out')
    assert not (tmp_path/'out').exists()
