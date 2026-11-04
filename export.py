import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import load_manifest, validate_group_split
from .peaks import decode_heatmap
from .fusion import fuse_three_views
from .train import train_detector, predict_heatmap


def _finite_number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x)


def validate_prediction_record(record):
    if record.get("schema_version")!="phase1_marker_case_v1": raise ValueError("invalid schema_version")
    for key in ("case_id","group_id","model_id","geometry_id","fold_id"):
        if not isinstance(record.get(key),str) or not record[key]: raise ValueError(f"{key} is required")
    if record.get("prediction_source") not in ("group_oof","held_out"): raise ValueError("invalid prediction_source")
    views=record.get("views")
    if not isinstance(views,list) or len(views)!=3: raise ValueError("exactly three views are required")
    if len({v.get("view_id") for v in views})!=3: raise ValueError("view ids must be distinct")
    default_q=record.get("q_tolerance_px")
    if default_q is not None and (not _finite_number(default_q) or default_q<=0): raise ValueError("q_tolerance_px must be positive")
    for v in views:
        size=v.get("image_size")
        if not isinstance(size,list) or len(size)!=2 or any(not isinstance(n,int) or n<=0 for n in size): raise ValueError("image_size must be [H,W]")
        status=v.get("pair_status")
        if status not in ("reliable","uncertain","invalid"): raise ValueError("invalid pair_status")
        mode=v.get("fusion_mode")
        if status=="invalid":
            if v.get("marker_xy") is not None or v.get("error_radius_px") is not None or not _finite_number(v.get("q_pair")) or v.get("q_pair")!=0 or mode is not None:
                raise ValueError("invalid points require null coordinates/radii/mode and q_pair=0")
            continue
        xy=np.asarray(v.get("marker_xy"),dtype=float); radii=np.asarray(v.get("error_radius_px"),dtype=float)
        if xy.shape!=(2,2) or radii.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(radii).all() or np.any(radii<0): raise ValueError("valid pair coordinates and radii must be finite")
        h,w=size
        if np.any(xy[:,0]<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]<0) or np.any(xy[:,1]>=h): raise ValueError("prediction coordinates out of bounds")
        if mode not in ("three_view","single_view_fallback"): raise ValueError("valid pair requires fusion_mode")
        q=v.get("q_pair")
        if q is not None and (not _finite_number(q) or q<0 or q>1): raise ValueError("q_pair must be null or in [0,1]")
        if q is not None and (record.get("calibration_id") is None or not _finite_number(v.get("q_tolerance_px",default_q)) or v.get("q_tolerance_px",default_q)<=0): raise ValueError("calibrated q_pair requires calibration_id and positive tolerance")
    return True


def _pair_metrics(truth,prediction,tau):
    truth=np.asarray(truth,dtype=float); prediction=np.asarray(prediction,dtype=float)
    direct=np.linalg.norm(truth-prediction,axis=1); swapped=np.linalg.norm(truth-prediction[::-1],axis=1)
    err=np.minimum(direct,swapped)
    return (float(np.mean(err)),bool(np.all(err<=tau)))


def evaluate(cases, prediction_dir, tau=3.0):
    pred_dir=Path(prediction_dir); errors=[]; correct=[]; fallbacks=invalid=0; case_success=[]
    for case in cases:
        rec=json.loads((pred_dir/f"{case.case_id}.json").read_text(encoding="utf-8")); validate_prediction_record(rec)
        view_success=[]
        for view,pred in zip(case.views,rec["views"]):
            if pred["pair_status"]=="invalid": invalid+=1; view_success.append(False); continue
            err,ok=_pair_metrics(view.marker_xy,pred["marker_xy"],tau); errors.append(err); correct.append(ok); view_success.append(ok)
            fallbacks+=pred["fusion_mode"]=="single_view_fallback"
        case_success.append(bool(len(view_success)==3 and all(view_success)))
    return {"mean_unordered_point_error_px":float(np.mean(errors)) if errors else None,
            "PairAcc@tau":float(np.mean(correct)) if correct else None,"tau_px":float(tau),
            "three_view_all_success_rate":float(np.mean(case_success)) if case_success else 0.0,
            "fallback_rate":fallbacks/max(1,len(correct)+invalid),"invalid_rate":invalid/max(1,len(correct)+invalid),"valid_view_count":len(correct)}


evaluate._pair_metrics=_pair_metrics


def _model_id(checkpoint_path, config):
    digest=hashlib.sha256(Path(checkpoint_path).read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    return f"marker_unet_mv_v1_{digest}"


def _predict_case(case, checkpoint, config, model_id, source, fold_id):
    sets=[]
    for v in case.views:
        heat=predict_heatmap(checkpoint,v.image)
        sets.append(decode_heatmap(heat,config))
    fusion=fuse_three_views(sets,[v.P for v in case.views],config)
    views=[]
    radius=float(config.get("fallback_error_radius_px",10.0))
    for i,v in enumerate(case.views):
        xy=fusion.marker_xy[i]
        status=(fusion.view_pair_status[i] if fusion.view_pair_status else fusion.pair_status)
        invalid=xy is None or status=="invalid"
        selected_scores=[]
        if not invalid:
            for point in xy:
                selected_scores.append(float(min(sets[i],key=lambda c:(c.x-point[0])**2+(c.y-point[1])**2).score))
        item={"view_id":int(v.view_id),"image_size":[int(v.image.shape[0]),int(v.image.shape[1])],
              "marker_xy":None if invalid else [[float(x),float(y)] for x,y in xy],
              "error_radius_px":None if invalid else [radius,radius],"q_pair":0 if invalid else None,
              "pair_status":"invalid" if invalid else status,
              "fusion_mode":None if invalid else fusion.fusion_mode[i],
              "peak_score":None if invalid else selected_scores,
              "reprojection_error_px":fusion.diagnostics.get("reprojection_error_px")}
        views.append(item)
    record={"schema_version":"phase1_marker_case_v1","case_id":case.case_id,"group_id":case.group_id,"model_id":model_id,
            "geometry_id":case.metadata.get("geometry_id","sim_geometry_v1"),"calibration_id":None,"prediction_source":source,
            "fold_id":fold_id,"q_tolerance_px":float(config.get("q_tolerance_px",3.0)),"association_status":fusion.association_status,"views":views}
    validate_prediction_record(record)
    return record


def export_phase2_fold(manifest, split, config, out_dir):
    cases=load_manifest(manifest) if isinstance(manifest,(str,Path)) else manifest
    validate_group_split(cases,split)
    out=Path(out_dir); pred_dir=out/"predictions"; pred_dir.mkdir(parents=True,exist_ok=True)
    by_group={g:[c for c in cases if c.group_id==g] for g in sorted({c.group_id for c in cases})}
    registry={}; index=[]; fold_id=str(config.get("phase1_fold_id","phase1_group_oof"))
    records=[]
    # Outer validation/test predictions share one frozen model fitted only on train groups.
    train_cases=[c for c in cases if c.group_id in set(split["train"])]
    held_dir=out/"models"/"held_out"; held_result=train_detector(train_cases,config,held_dir)
    held_ckpt=Path(held_result["checkpoint"]); held_id=_model_id(held_ckpt,config)
    held_selection=set(config.get("parameter_selection_groups",[]))
    if not held_selection.issubset(set(split.get("train",[]))): raise ValueError("held-out parameter selection groups must be outer train groups")
    registry[held_id]={"checkpoint":str(held_ckpt.relative_to(out)),"fit_groups":held_result["fit_groups"],"parameter_selection_groups":sorted(held_selection),"parameter_selection_status":"verified" if held_selection else "externally_frozen_unverified","calibration_groups":[],"config":dict(config)}
    for group in split.get("val",[])+split.get("test",[]):
        for case in by_group[group]: records.append(_predict_case(case,held_ckpt,config,held_id,"held_out",fold_id))
    # Group-wise OOF predictions for the phase-2 training groups.
    for group in split.get("train",[]):
        fit=[c for c in train_cases if c.group_id!=group]
        if not fit: raise ValueError("group_oof requires at least two training groups")
        model_dir=out/"models"/f"oof_{group}"; result=train_detector(fit,config,model_dir); ckpt=Path(result["checkpoint"])
        selection_map=config.get("parameter_selection_groups_by_excluded_group",{})
        selection=set(selection_map.get(group,[])) if isinstance(selection_map,dict) else set()
        if not selection.issubset(set(result["fit_groups"])): raise ValueError(f"OOF parameter selection groups for {group} must be fitted groups excluding the predicted group")
        mid=_model_id(ckpt,config)
        registry[mid]={"checkpoint":str(ckpt.relative_to(out)),"fit_groups":result["fit_groups"],"parameter_selection_groups":sorted(selection),"parameter_selection_status":"verified" if selection else "externally_frozen_unverified","calibration_groups":[],"config":dict(config)}
        for case in by_group[group]: records.append(_predict_case(case,ckpt,config,mid,"group_oof",f"{fold_id}_excluding_{group}"))
    for record in records:
        path=pred_dir/f"{record['case_id']}.json"; path.write_text(json.dumps(record,indent=2),encoding="utf-8")
        for v in record["views"]: index.append({"case_id":record["case_id"],"view_id":v["view_id"],"group_id":record["group_id"],"prediction_path":str(path.relative_to(out))})
    (out/"source_registry.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    (out/"package.json").write_text(json.dumps({"phase2_fold_id":config.get("phase2_fold_id","fold_0"),"split":split,"schema_version":"phase1_marker_case_v1"},indent=2),encoding="utf-8")
    with (out/"case_view_index.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["case_id","view_id","group_id","prediction_path"]); writer.writeheader(); writer.writerows(index)
    return {"prediction_count":len(records),"out_dir":str(out),"model_count":len(registry)}
