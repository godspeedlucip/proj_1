import argparse
import json
from pathlib import Path
from .data import load_manifest, validate_group_split
from .synthetic import make_synthetic_dataset
from .train import train_detector
from .export import export_phase2_fold, evaluate, _predict_case, _model_id


def _config(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def main(argv=None):
    parser=argparse.ArgumentParser(prog="phase1-marker")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("make-synthetic"); p.add_argument("out_dir"); p.add_argument("--seed",type=int,default=0)
    p=sub.add_parser("validate"); p.add_argument("manifest"); p.add_argument("--split",default="fold_0")
    p=sub.add_parser("train"); p.add_argument("manifest"); p.add_argument("out_dir"); p.add_argument("--config")
    p=sub.add_parser("predict"); p.add_argument("manifest"); p.add_argument("checkpoint"); p.add_argument("out_dir"); p.add_argument("--config")
    p=sub.add_parser("evaluate"); p.add_argument("manifest"); p.add_argument("prediction_dir"); p.add_argument("--tau",type=float,default=3.0)
    p=sub.add_parser("export-fold"); p.add_argument("manifest"); p.add_argument("out_dir"); p.add_argument("--split",default="fold_0"); p.add_argument("--config")
    args=parser.parse_args(argv)
    if args.command=="make-synthetic":
        print(make_synthetic_dataset(args.out_dir,args.seed)); return 0
    if args.command=="validate":
        cases=load_manifest(args.manifest); payload=json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        split=payload.get("folds",{}).get(args.split)
        if split: validate_group_split(cases,split)
        print(json.dumps({"cases":len(cases),"groups":len({c.group_id for c in cases}),"split_valid":bool(split)},indent=2)); return 0
    if args.command=="train":
        cases=load_manifest(args.manifest); result=train_detector(cases,_config(args.config),args.out_dir)
        print(json.dumps(result,indent=2)); return 0
    if args.command=="predict":
        import torch
        from .train import predict_heatmap
        cases=load_manifest(args.manifest); cfg=_config(args.config); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
        ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False); cfg={**ckpt.get("config",{}),**cfg}
        mid=_model_id(args.checkpoint,cfg)
        for case in cases:
            rec=_predict_case(case,args.checkpoint,cfg,mid,"held_out","inference")
            (out/f"{case.case_id}.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
        print(json.dumps({"prediction_count":len(cases),"out_dir":str(out)},indent=2)); return 0
    if args.command=="evaluate":
        cases=load_manifest(args.manifest); print(json.dumps(evaluate(cases,args.prediction_dir,args.tau),indent=2)); return 0
    if args.command=="export-fold":
        payload=json.loads(Path(args.manifest).read_text(encoding="utf-8")); split=payload.get("folds",{}).get(args.split)
        if split is None: raise SystemExit(f"split not found: {args.split}")
        cfg=_config(args.config); cfg.setdefault("phase2_fold_id",args.split)
        print(json.dumps(export_phase2_fold(args.manifest,split,cfg,args.out_dir),indent=2)); return 0


if __name__=="__main__":
    raise SystemExit(main())
