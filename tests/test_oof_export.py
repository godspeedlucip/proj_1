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
