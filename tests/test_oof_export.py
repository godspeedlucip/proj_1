import json
import numpy as np

from phase_1.marker_pipeline.synthetic import make_synthetic_dataset
from phase_1.marker_pipeline.data import load_manifest
from phase_1.marker_pipeline.export import export_phase2_fold, validate_prediction_record, evaluate


def test_export_records_three_views_and_never_fits_predicted_group(tmp_path):
    manifest=make_synthetic_dataset(tmp_path/"data",seed=2)
    cases=load_manifest(manifest)
    payload=json.loads(manifest.read_text(encoding="utf-8")); split=payload["folds"]["fold_0"]
    out=tmp_path/"export"
    result=export_phase2_fold(manifest,split,{"epochs":1,"widths":[4,8,16,32],"batch_size":12,"seed":3,"top_k":5,"top_l":5},out)
    registry=json.loads((out/"source_registry.json").read_text(encoding="utf-8"))
    for case in cases:
        rec=json.loads((out/"predictions"/f"{case.case_id}.json").read_text(encoding="utf-8"))
        validate_prediction_record(rec)
        assert len(rec["views"])==3
        model=registry[rec["model_id"]]
        assert model["parameter_selection_groups"] == []
        assert model["parameter_selection_status"] == "externally_frozen_unverified"
        if rec["prediction_source"]=="group_oof":
            assert case.group_id not in model["fit_groups"]
            assert set(model["fit_groups"]) <= set(split["train"])
        else:
            assert set(model["fit_groups"]) == set(split["train"])
    assert result["prediction_count"]==len(cases)


def test_contract_accepts_null_qpair_and_invalid_qpair_zero():
    base={"schema_version":"phase1_marker_case_v1","case_id":"c","group_id":"g","model_id":"m","geometry_id":"sim_geometry_v1","calibration_id":None,"prediction_source":"held_out","fold_id":"fold_0","q_tolerance_px":3.0,"association_status":"resolved","views":[]}
    for i in range(3):
        base["views"].append({"view_id":i,"image_size":[16,16],"marker_xy":[[2.1,3.2],[10.0,11.0]],"error_radius_px":[5.,5.],"q_pair":None,"pair_status":"reliable","fusion_mode":"single_view_fallback"})
    validate_prediction_record(base)
    bad=json.loads(json.dumps(base)); bad["views"][0].update(marker_xy=None,error_radius_px=None,q_pair=0,pair_status="invalid",fusion_mode=None)
    validate_prediction_record(bad)


def test_unordered_evaluation_is_permutation_invariant():
    truth=np.array([[2.,3.],[10.,11.]])
    a=np.array([[2.2,3.],[9.8,11.]])
    b=a[::-1]
    assert evaluate._pair_metrics(truth,a,3)==evaluate._pair_metrics(truth,b,3)


def test_unordered_evaluation_requires_one_to_one_pair_matching():
    truth=np.array([[0.,0.],[0.2,0.]])
    one_distant_prediction=np.array([[0.,0.],[10.,0.]])
    error,ok=evaluate._pair_metrics(truth,one_distant_prediction,1.0)
    assert not ok
    assert error > 4.0
    assert evaluate._pair_metrics(truth[::-1],one_distant_prediction,1.0)==(error,ok)


def test_invalid_view_counts_as_failure_in_primary_pair_accuracy(tmp_path):
    from phase_1.tests.test_joint_pipeline import make_case
    case=make_case("c","g",0)
    views=[]
    for index,view in enumerate(case.views):
        views.append({"view_id":view.view_id,"image_size":[32,32],
                      "marker_xy":None if index==2 else view.marker_xy.tolist(),
                      "error_radius_px":None if index==2 else [1.,1.],
                      "q_pair":0 if index==2 else None,
                      "pair_status":"invalid" if index==2 else "reliable",
                      "fusion_mode":None if index==2 else "three_view"})
    record={"schema_version":"phase1_marker_case_v1","case_id":"c","group_id":"g",
            "model_id":"m","geometry_id":"fixed_cone","calibration_id":None,
            "prediction_source":"held_out","fold_id":"fold0","views":views}
    (tmp_path/"c.json").write_text(json.dumps(record),encoding="utf-8")
    result=evaluate([case],tmp_path,tau=1.0)
    assert result["PairAcc@tau"]==2/3
    assert result["conditional_PairAcc@tau"]==1.0
    assert result["three_view_all_success_rate"]==0.0
