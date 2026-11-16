import json
import numpy as np
import pytest
from stent_repro.evaluation import segmentation_metrics, resample_volume, summarize_cases, evaluate_manifest
from stent_repro.io import save_volume


def grid(shape=(5,5,5),spacing=(.6,.6,.6),origin=(0,0,0)):
    return {'shape':list(shape),'spacing_mm':list(spacing),'origin_mm':list(origin),'direction':np.eye(3).tolist()}


def test_binary_metrics_and_directed_physical_distance():
    gt=np.zeros((5,5,5),bool); gt[2,2,2]=1
    pred=np.roll(gt,1,axis=2)
    m=segmentation_metrics(pred,gt,grid())
    assert m['dice']==0 and m['precision']==0 and m['recall']==0
    assert m['pred_to_truth_mm']==pytest.approx(.6)
    assert m['truth_to_pred_mm']==pytest.approx(.6)
    assert segmentation_metrics(gt,gt,grid())['dice']==1


def test_empty_prediction_stays_a_failure_not_zero_distance():
    gt=np.zeros((5,5,5),bool); gt[2,2,2]=1
    m=segmentation_metrics(np.zeros_like(gt),gt,grid())
    assert m['status']=='empty_prediction' and m['dice']==0
    assert m['precision'] is None and m['pred_to_truth_mm'] is None
    assert segmentation_metrics(np.zeros_like(gt),np.zeros_like(gt),grid())['status']=='empty_truth'
    s=summarize_cases([{'case_id':'x',**m},{'case_id':'y','status':'pipeline_failure','error':'bad'}])
    assert s['case_count']==2 and s['failure_count']==2 and s['failure_rate']==1
    assert s['metrics']['pred_to_truth_mm']['n']==0
    json.dumps(s,allow_nan=False)


def test_resampling_is_physical_and_checks_coverage():
    source=np.zeros((5,5,5),np.float32); source[2,2,2]=1
    target=grid((3,3,3),origin=(.6,.6,.6))
    result=resample_volume(source,grid(),target)
    assert result[1,1,1]==pytest.approx(1)
    with pytest.raises(ValueError,match='cover'): resample_volume(source,grid(),grid(origin=(-5,0,0)))
    g=grid(); g['direction'][0][0]=-1
    with pytest.raises(ValueError,match='direction'): resample_volume(source,g,grid())


def test_manifest_reports_failed_cases_and_writes_artifacts(tmp_path):
    gt=np.zeros((5,5,5),np.uint8); gt[2,2,2]=1
    save_volume(tmp_path/'gt.npz',gt,grid())
    save_volume(tmp_path/'p.npz',gt.astype(np.float32),grid())
    manifest={'threshold':.5,'cases':[
        {'id':'ok','entity_id':'s1','probability':'p.npz','truth':'gt.npz'},
        {'id':'bad','entity_id':'s2','probability':'missing.npz','truth':'gt.npz'}]}
    path=tmp_path/'eval.json';path.write_text(json.dumps(manifest),encoding='utf8')
    out=tmp_path/'result'; report=evaluate_manifest(path,out)
    assert report['case_count']==2 and report['failure_count']==1
    assert (out/'metrics_per_case.csv').exists() and (out/'summary.json').exists()
