import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from .data import load_manifest, validate_group_split
from .peaks import decode_heatmap
from .fusion import fuse_three_views
from .train import train_detector, predict_heatmap, predict_joint_heatmaps


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
    err=direct if np.sum(direct)<=np.sum(swapped) else swapped
    return (float(np.mean(err)),bool(np.all(direct<=tau) or np.all(swapped<=tau)))


def _candidate_pair_recall(truth,candidates,tau,top_l=None):
    pairs=list(combinations(candidates,2))
    if top_l is not None:
        pairs=sorted(pairs,key=lambda pair:-(pair[0]["score"]+pair[1]["score"]))[:top_l]
    return any(_pair_metrics(truth,[a["xy"],b["xy"]],tau)[1] for a,b in pairs)


def _diagnostic_path(pred_dir,case_id):
    for root in (pred_dir/"diagnostics",pred_dir.parent/"diagnostics"):
        path=root/f"{case_id}.json"
        if path.is_file(): return path
    return None


def evaluate(cases, prediction_dir, tau=3.0):
    pred_dir=Path(prediction_dir); errors=[]; correct=[]; fallbacks=invalid=0; case_success=[]
    diagnostics={"initial_topk_pair_recall":[],"refined_topk_pair_recall":[],
                 "refined_topl_pair_recall":[],"ablation_no_geometry_PairAcc@tau":[]}
    strata={}
    for case in cases:
        rec=json.loads((pred_dir/f"{case.case_id}.json").read_text(encoding="utf-8")); validate_prediction_record(rec)
        diagnostic_file=_diagnostic_path(pred_dir,case.case_id)
        diagnostic_payload=json.loads(diagnostic_file.read_text(encoding="utf-8")) if diagnostic_file else None
        diagnostic_views={v["view_id"]:v for v in diagnostic_payload["views"]} if diagnostic_payload and diagnostic_payload.get("model_id")==rec["model_id"] else {}
        view_success=[]
        predictions={view["view_id"]:view for view in rec["views"]}
        for view in case.views:
            pred=predictions[view.view_id]
            if pred["pair_status"]=="invalid": invalid+=1; view_success.append(False); continue
            err,ok=_pair_metrics(view.marker_xy,pred["marker_xy"],tau); errors.append(err); correct.append(ok); view_success.append(ok)
            fallbacks+=pred["fusion_mode"]=="single_view_fallback"
        for view in case.views:
            if view.view_id not in diagnostic_views: continue
            d=diagnostic_views[view.view_id]
            diagnostics["initial_topk_pair_recall"].append(_candidate_pair_recall(view.marker_xy,d["initial_candidates"],tau))
            diagnostics["refined_topk_pair_recall"].append(_candidate_pair_recall(view.marker_xy,d["refined_candidates"],tau))
            diagnostics["refined_topl_pair_recall"].append(_candidate_pair_recall(view.marker_xy,d["refined_candidates"],tau,int(d["top_l"])))
            ablated=d["no_geometry_marker_xy"]
            diagnostics["ablation_no_geometry_PairAcc@tau"].append(False if ablated is None else _pair_metrics(view.marker_xy,ablated,tau)[1])
        case_success.append(bool(len(view_success)==3 and all(view_success)))
        strata.setdefault(case.difficulty,[]).extend(view_success)
    total_views=len(correct)+invalid
    result={"mean_unordered_point_error_px":float(np.mean(errors)) if errors else None,
            "PairAcc@tau":sum(correct)/total_views if total_views else None,
            "conditional_PairAcc@tau":float(np.mean(correct)) if correct else None,"tau_px":float(tau),
            "three_view_all_success_rate":float(np.mean(case_success)) if case_success else 0.0,
            "fallback_rate":fallbacks/max(1,total_views),"invalid_rate":invalid/max(1,total_views),
            "valid_view_count":len(correct),
            "PairAcc@tau_by_difficulty":{name:float(np.mean(values)) for name,values in strata.items()}}
    result.update({name:float(np.mean(values)) if values else None for name,values in diagnostics.items()})
    result["diagnostic_view_count"]=len(diagnostics["refined_topk_pair_recall"])
    return result


evaluate._pair_metrics=_pair_metrics


def _model_id(checkpoint_path, config):
    digest=hashlib.sha256(Path(checkpoint_path).read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    return f"marker_unet_mv_v1_{digest}"


def _predict_case(case, checkpoint, config, model_id, source, fold_id, diagnostics_dir=None):
    checkpoint_format=torch.load(checkpoint,map_location="cpu",weights_only=False).get("format")
    if checkpoint_format=="marker_joint_epipolar_checkpoint_v1":
        initial,refined=predict_joint_heatmaps(checkpoint,case)
        initial_sets=[decode_heatmap(heat,config) for heat in initial]
        sets=[decode_heatmap(heat,config) for heat in refined]
    elif checkpoint_format=="marker_unet_checkpoint_v1":
        initial_sets=None
        sets=[decode_heatmap(predict_heatmap(checkpoint,v.image),config) for v in case.views]
    else:
        raise ValueError(f"unknown checkpoint format: {checkpoint_format}")
    if initial_sets is not None and diagnostics_dir is not None:
        diagnostic_root=Path(diagnostics_dir)
        diagnostic_root.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(diagnostic_root/f"{case.case_id}.npz",initial=initial,refined=refined)
        diagnostic_views=[]
        for index,view in enumerate(case.views):
            def serialize(candidates):
                return [{"xy":[float(c.x),float(c.y)],"score":float(c.score)} for c in candidates]
            top_two=sorted(sets[index],key=lambda c:-c.score)[:2]
            diagnostic_views.append({"view_id":int(view.view_id),
                                     "initial_candidates":serialize(initial_sets[index]),
                                     "refined_candidates":serialize(sets[index]),
                                     "top_l":max(1,int(config.get("top_l",config.get("L",5)))),
                                     "no_geometry_marker_xy":None if len(top_two)<2 else [[float(c.x),float(c.y)] for c in top_two]})
        (diagnostic_root/f"{case.case_id}.json").write_text(
            json.dumps({"case_id":case.case_id,"model_id":model_id,"views":diagnostic_views},indent=2),encoding="utf-8")
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
              "reprojection_error_px":fusion.diagnostics.get("reprojection_error_px"),
              "initial_candidate_count":None if initial_sets is None else len(initial_sets[i]),
              "refined_candidate_count":len(sets[i])}
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
        for case in by_group[group]: records.append(_predict_case(case,held_ckpt,config,held_id,"held_out",fold_id,out/"diagnostics"))
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
        for case in by_group[group]: records.append(_predict_case(case,ckpt,config,mid,"group_oof",f"{fold_id}_excluding_{group}",out/"diagnostics"))
    for model_id,entry in registry.items():
        ckpt=torch.load(out/entry["checkpoint"],map_location="cpu",weights_only=False)
        entry["architecture"]=ckpt["format"]
        if "reference_F" in ckpt:
            entry["geometry_id"]=ckpt["geometry_id"]
            entry["relative_geometry_sha256"]=hashlib.sha256(np.asarray(ckpt["reference_F"],dtype=np.float64).tobytes()).hexdigest()
    for record in records:
        path=pred_dir/f"{record['case_id']}.json"; path.write_text(json.dumps(record,indent=2),encoding="utf-8")
        for v in record["views"]: index.append({"case_id":record["case_id"],"view_id":v["view_id"],"group_id":record["group_id"],"prediction_path":str(path.relative_to(out))})
    (out/"source_registry.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    phase2_sources=[{"model_id":model_id,"model_artifact_id":model_id,
                     "pipeline_config_id":hashlib.sha256(json.dumps(entry["config"],sort_keys=True).encode()).hexdigest()[:16],
                     "fit_group_ids":entry["fit_groups"],
                     "selection_group_ids":entry["parameter_selection_groups"]}
                    for model_id,entry in registry.items()]
    package={"phase2_fold_id":config.get("phase2_fold_id","fold_0"),"split":split,
             "schema_version":"phase1_marker_case_v1","source_registry":phase2_sources,
             "cases":records}
    (out/"package.json").write_text(json.dumps(package,indent=2),encoding="utf-8")
    with (out/"case_view_index.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["case_id","view_id","group_id","prediction_path"]); writer.writeheader(); writer.writerows(index)
    return {"prediction_count":len(records),"out_dir":str(out),"model_count":len(registry)}
