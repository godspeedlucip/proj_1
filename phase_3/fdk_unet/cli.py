"""Explicit local CLI for the FDK v3 experiment pipeline."""
import argparse
import json
from pathlib import Path
import sys
import torch
from .config import load_config,resolve_config
from .manifest import load_manifest,load_splits,validate_data,import_nnunet,manifest_from_dict,FoldSplit,select_cases,json_digest
from .prior_weights import prepare_prior,file_sha256
from .checkpoint import load_run
from .train import train
from .predict import predict_volume,validate_device
from .nifti_io import load_volume,save_prediction
from .evaluate import evaluate_predictions


def parser():
    p=argparse.ArgumentParser(description='Independent FDK stent-strut segmentation v3')
    sub=p.add_subparsers(dest='command',required=True)
    imp=sub.add_parser('import-nnunet');imp.add_argument('--source',type=Path,required=True)
    imp.add_argument('--source-groups',type=Path,required=True);imp.add_argument('--output',type=Path,required=True)
    for name in ('validate-data','prepare-prior','train','predict','evaluate'):
        command=sub.add_parser(name)
        command.add_argument('--manifest',type=Path,required=name not in ('train',))
        command.add_argument('--splits',type=Path,required=name in ('validate-data','prepare-prior','evaluate'))
        command.add_argument('--fold',type=int)
        if name in ('validate-data','prepare-prior'):command.add_argument('--smoke',action='store_true')
        if name in ('train','predict'):command.add_argument('--device',default='cpu')
        if name!='validate-data':command.add_argument('--output',type=Path,required=name!='train')
        if name=='prepare-prior':
            command.add_argument('--init',choices=('ssl','random'),default='ssl');command.add_argument('--weights',type=Path)
            command.add_argument('--download-official',action='store_true');command.add_argument('--seed',type=int,default=2026)
        if name=='train':
            command.add_argument('--config',type=Path);command.add_argument('--resume',type=Path)
            command.add_argument('--seed',type=int);command.add_argument('--prior-artifact',type=Path)
            command.add_argument('--initialization-artifact',type=Path)
        if name=='predict':command.add_argument('--checkpoint',type=Path,required=True)
        if name in ('predict','evaluate'):command.add_argument('--split',choices=('train','validation','test'),default='test')
        if name=='evaluate':command.add_argument('--predictions',type=Path,required=True)
    return p


def _split(splits,fold):
    fold=0 if fold is None else fold
    if fold not in splits:raise ValueError(f'Unknown fold: {fold}')
    return splits[fold]


def main(argv=None):
    try:args=parser().parse_args(argv)
    except SystemExit as exc:return int(exc.code)
    try:
        if args.command=='import-nnunet':
            print(import_nnunet(args.source,args.source_groups,args.output));return 0
        if args.command=='train' and args.resume is not None:
            forbidden=('config','manifest','splits','fold','seed','prior_artifact','initialization_artifact','output')
            if any(getattr(args,k) is not None for k in forbidden):raise ValueError('Resume forbids experiment overrides; use only --resume and --device')
            loaded=load_run(args.resume,device=validate_device(args.device),for_training=True)
            manifest=manifest_from_dict(loaded.manifest_snapshot,Path.cwd())
            split=FoldSplit(**{k:tuple(v) for k,v in loaded.split_snapshot.items()});fold=loaded.payload['fold']
            result=train(loaded.config,manifest,{fold:split},fold=fold,device=torch.device(args.device),output=args.resume.resolve().parent,resume=args.resume)
            print(f'Completed optimizer steps: {result.completed_steps}');return 0
        if args.command=='train':
            if any(getattr(args,k) is None for k in ('config','manifest','splits','output')):
                raise ValueError('New train requires config, manifest, splits and output')
            cfg=load_config(args.config);raw=cfg.resolved
            if args.seed is not None:raw['seed']=args.seed
            cfg=resolve_config(raw)
            manifest=load_manifest(args.manifest,require_labels=True);splits=load_splits(args.splits)
            result=train(cfg,manifest,splits,fold=args.fold if args.fold is not None else 0,device=torch.device(args.device),
                         output=args.output,prior_artifact=args.prior_artifact,initialization_artifact=args.initialization_artifact)
            print(f'Completed optimizer steps: {result.completed_steps}');return 0
        manifest=load_manifest(args.manifest,require_labels=args.command!='predict')
        splits=load_splits(args.splits) if args.splits else None
        if args.command=='validate-data':
            print(json.dumps(validate_data(manifest,splits,formal=not args.smoke),indent=2));return 0
        if args.command=='prepare-prior':
            validate_data(manifest,splits,formal=not args.smoke)
            print(prepare_prior(manifest,_split(splits,args.fold),init=args.init,weights=args.weights,
                                download_official=args.download_official,output=args.output,seed=args.seed));return 0
        if args.command=='predict':
            device=validate_device(args.device);loaded=load_run(args.checkpoint,device=device)
            cases=select_cases(manifest,_split(splits,args.fold),args.split) if splits else manifest.cases
            if not cases:raise ValueError('No prediction cases')
            digest=file_sha256(args.checkpoint)
            provenance={'schema_version':'fdk_predictions_v1','mode':'experimental' if splits else 'external',
                        'checkpoint_sha256':digest,'checkpoint_fold':loaded.payload.get('fold'),
                        'split_digest':loaded.payload['split_digest'],'split_name':args.split if splits else None,
                        'case_groups':{c.case_id:c.group_id for c in cases},
                        'projection_sources':{c.case_id:c.projection_source for c in cases},'outputs':{},'label_sha256':{}}
            if splits:
                fold=0 if args.fold is None else args.fold
                if fold!=loaded.payload.get('fold') or json_digest(_split(splits,fold).to_dict())!=loaded.payload['split_digest']:
                    raise ValueError('Prediction fold/split does not match checkpoint; omit splits for explicit external inference')
                saved={c['case_id']:c for c in loaded.manifest_snapshot['cases']}
                audit={row['case_id']:row for row in loaded.payload.get('data_audit',{}).get('files',[])}
                for record in cases:
                    old=saved.get(record.case_id)
                    if old is None or record.case_id not in audit:raise ValueError('Checkpoint lacks experimental case provenance')
                    identity=(record.group_id,record.source_model_ids,record.patient_id,record.angles_deg,record.projection_source,record.upstream_run_id)
                    expected=(old['group_id'],tuple(old['source_model_ids']),old.get('patient_id'),tuple(old['angles_deg']),old['projection_source'],old.get('upstream_run_id'))
                    if identity!=expected or file_sha256(record.image)!=audit[record.case_id]['image_sha256']:
                        raise ValueError('Experimental prediction input/provenance differs from checkpoint')
                    provenance['label_sha256'][record.case_id]=audit[record.case_id]['label_sha256']
            args.output.mkdir(parents=True,exist_ok=False)
            for record in cases:
                volume=load_volume(record,require_label=False)
                prediction=predict_volume(loaded.model,volume,loaded.config,loaded.stats,device=device)
                prediction.telemetry['checkpoint_sha256']=digest
                prob,mask=save_prediction(prediction.probability,prediction.mask,volume.meta,case_id=record.case_id,output=args.output)
                provenance['outputs'][record.case_id]={'prob_sha256':file_sha256(prob),'mask_sha256':file_sha256(mask)}
                (args.output/f'{record.case_id}_telemetry.json').write_text(json.dumps(prediction.telemetry,indent=2),encoding='utf8')
            (args.output/'prediction_run.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
            print(f'Predicted {len(cases)} original-grid volumes');return 0
        summary=evaluate_predictions(manifest,_split(splits,args.fold),args.predictions,split_name=args.split,output=args.output,
                                     expected_fold=0 if args.fold is None else args.fold)
        print(json.dumps(summary,indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError,ImportError,RuntimeError) as exc:
        print(f'FDK error: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
