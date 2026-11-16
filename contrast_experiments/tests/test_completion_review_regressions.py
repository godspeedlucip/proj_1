import json
from pathlib import Path
import pytest
from stent_repro.experiments import reference_matrix,summarize_matrix,_evaluation_document,_common_test_signature


def test_mixed_reference_angles_cannot_overwrite_channel_comparison(tmp_path,workflow_config):
    jobs=reference_matrix(workflow_config(),workflow_config('B'),[17])
    (tmp_path/'matrix.json').write_text(json.dumps(jobs),encoding='utf8')
    rows=[{'run_id':j['run_id'],'evaluation':{'cases':[{'case_id':'x','entity_id':'stent',
        'status':'ok','dice':.9 if '22_5' in j['run_id'] else .1}]}} for j in jobs]
    (tmp_path/'matrix_report.json').write_text(json.dumps({'jobs':rows}),encoding='utf8')
    with pytest.raises(ValueError,match='acquisition|duplicate'):
        summarize_matrix(tmp_path)


def _truth(tmp_path):
    path=tmp_path/'truth.json'
    (tmp_path/'truth.npz').write_bytes(b'identical truth fixture')
    path.write_text(json.dumps({'cases':[{'id':x,'entity_id':'shared-stent','truth':'truth.npz'} for x in ('a','b')]}),encoding='utf8')
    return path


def test_truth_identity_prevents_repeat_acquisition_pseudoreplication(tmp_path):
    truth=_truth(tmp_path)
    inference={'cases':[{'id':x,'entity_id':x,'entity_id_source':'case_id_fallback','status':'ok'} for x in ('a','b')]}
    result=_evaluation_document(inference,truth,.5)
    assert [r['entity_id'] for r in result['cases']]==['shared-stent']*2


def test_explicit_conflicting_entity_identity_is_rejected(tmp_path):
    truth=_truth(tmp_path)
    inference={'cases':[{'id':x,'entity_id':'other-stent','status':'ok'} for x in ('a','b')]}
    with pytest.raises(ValueError,match='entity'):_evaluation_document(inference,truth,.5)


def test_common_test_signature_uses_validated_truth_entity(tmp_path):
    truth=_truth(tmp_path);(tmp_path/'projections.npz').write_bytes(b'same projections fixture')
    implicit=tmp_path/'implicit.json';explicit=tmp_path/'explicit.json'
    cases=[{'id':x,'projections':'projections.npz'} for x in ('a','b')]
    implicit.write_text(json.dumps({'cases':cases}),encoding='utf8')
    explicit.write_text(json.dumps({'cases':[{**r,'entity_id':'shared-stent'} for r in cases]}),encoding='utf8')
    assert _common_test_signature(implicit,truth)==_common_test_signature(explicit,truth)


def test_run_matrix_failure_roster_uses_truth_identity(tmp_path,workflow_config,monkeypatch):
    from stent_repro.experiments import experiment_matrix,run_matrix
    import stent_repro.workflow as workflow
    truth=_truth(tmp_path);(tmp_path/'projections.npz').write_bytes(b'fixed projections')
    inf=tmp_path/'inference.json';inf.write_text(json.dumps({'cases':[{'id':x,'projections':'projections.npz'} for x in ('a','b')]}),encoding='utf8')
    empty=tmp_path/'prepared.json';empty.write_text('{}',encoding='utf8')
    def fail(*a,**k):raise ValueError('controlled training failure')
    monkeypatch.setattr(workflow,'train_workflow',fail)
    manifests={m:{'inference':str(inf),'truth':str(truth),'dte':str(empty),'dtr':str(empty)} for m in ('A','B')}
    report=run_matrix(experiment_matrix(workflow_config(),workflow_config('B'),[17]),manifests,tmp_path/'matrix',execute=True)
    assert report['expected_case_entities']=={'a':'shared-stent','b':'shared-stent'}
