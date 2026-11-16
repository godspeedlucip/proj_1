import json
from stent_repro.experiments import smoke


def test_synthetic_train_infer_evaluate_chain(tmp_path):
    r=smoke(tmp_path/'g0',seed=29)
    assert r['scope']=='G0_compact_synthetic_workflow'
    assert r['g1_complete'] is False and r['g2_complete'] is False
    for method in ('A','B'):
        assert r['methods'][method]['inference_case_count']==1
        assert r['methods'][method]['pipeline_failure_count']==0
        assert r['methods'][method]['evaluation']['case_count']==1
        assert r['methods'][method]['evaluation']['metrics']['dice']['n']==1
    assert (tmp_path/'g0'/'report.json').exists()
    json.dumps(r,allow_nan=False)
