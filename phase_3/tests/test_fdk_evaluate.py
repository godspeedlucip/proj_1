from dataclasses import replace
import numpy as np
import nibabel as nib
import pytest


def prediction(mask):
    from phase_3.fdk_unet.nifti_io import SpatialMeta
    from phase_3.fdk_unet.predict import Prediction
    aff=np.array([[0,-.2,0,12],[.2,0,0,3],[0,0,.2,-9],[0,0,0,1.]])
    return Prediction(mask.astype(np.float32),mask.astype(np.uint8),SpatialMeta(tuple(reversed(mask.shape)),aff,nib.Nifti1Header(),'mm',aff),{})


def test_hand_computable_overlap_and_empty_rules():
    from phase_3.fdk_unet.evaluate import compute_case_metrics
    y=np.zeros((8,8,8),np.uint8);p=y.copy();y[1,1,1:4]=1;p[1,1,2:5]=1
    metric=compute_case_metrics(prediction(p),y,case_id='c',group_id='g')
    assert metric.dice==pytest.approx(2/3) and metric.precision==pytest.approx(2/3) and metric.recall==pytest.approx(2/3)
    for pp,yy,expected in ((p*0,y*0,(1,1,1,0,'both_empty')),(p*0,y,(0,0,0,None,'empty_prediction')),(p,y*0,(0,0,1,None,'empty_target'))):
        m=compute_case_metrics(prediction(pp),yy,case_id='c',group_id='g')
        assert (m.dice,m.precision,m.recall,m.skeleton_distance_mm,m.empty_status)==expected


def test_world_skeleton_distance_and_false_positive_volume():
    from phase_3.fdk_unet.evaluate import compute_case_metrics
    y=np.zeros((8,8,8),np.uint8);p=y.copy();y[3,3,1:7]=1;p[4,3,1:7]=1
    m=compute_case_metrics(prediction(p),y,case_id='c',group_id='g')
    assert m.skeleton_distance_mm==pytest.approx(.2)
    assert m.false_positive_mm3==pytest.approx(6*.008)
    assert m.pred_components==m.target_components==1


def test_group_macro_is_not_window_weighted():
    from phase_3.fdk_unet.evaluate import compute_case_metrics,aggregate_group_metrics
    y=np.zeros((8,8,8),np.uint8);y[2,2,2]=1
    good=compute_case_metrics(prediction(y),y,case_id='c0',group_id='a')
    bad=compute_case_metrics(prediction(y*0),y,case_id='c2',group_id='b')
    rows=[good,replace(good,case_id='c1'),bad]
    r=aggregate_group_metrics(rows)
    assert r['mean']['dice']==.5 and r['std']['dice']==.5
    assert r['distance_valid_cases']==2 and r['distance_failures']==1
    with pytest.raises(ValueError):aggregate_group_metrics(rows+[good])
