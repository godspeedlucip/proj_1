"""Resumable, locked acquisition experiment with automatic held-out evaluation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .acquisition import ARMS, read_json, write_json, sha256

PROJECT = Path('/data1/liuyang/stent/proj_1')
SOURCE = Path(__file__).resolve().parents[2]
PHASE3_PYTHON = '/data1/liuyang/conda/envs/phase3/bin/python'
TIGRE_PYTHON = '/data1/liuyang/conda/envs/nnunet/bin/python'
METRICS = ['dice', 'iou', 'precision', 'recall', 'false_positive_mm3', 'skeleton_distance_mm']


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Task already running: {path}') from exc
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid())); stream.flush()
        # Closing our descriptor keeps an inherited lock alive in orphaned stages.
        yield stream


def check_checkpoint_state(payload, *, config_digest, steps, updates):
    state = payload.get('phase3_state', {})
    expected = dict(global_step=steps, next_epoch=steps//updates, config_digest=config_digest,
                    max_optimizer_steps=steps, updates_per_epoch=updates)
    if any(state.get(key) != value for key, value in expected.items()) or payload.get('current_epoch') != steps//updates:
        raise ValueError('Checkpoint does not match the complete experiment budget/configuration')
    return True


def training_mode(fold):
    fold=Path(fold)
    if (fold/'checkpoint_latest.pth').is_file():return 'resume'
    if (fold/'checkpoint_final.pth').is_file():return 'complete'
    if fold.exists():
        fold.rename(fold.with_name(fold.name+f'.incomplete_{time.time_ns()}'))
    return 'fresh'


def check_result_identity(prediction, summary, rows, *, expected_checkpoint, expected_cases, split_name):
    identity={key:prediction.get(key) for key in ('checkpoint_sha256','split_digest','source_fold','split_name')}
    if (identity['checkpoint_sha256']!=expected_checkpoint or identity['split_name']!=split_name
            or identity['source_fold']!=0 or summary.get('provenance')!=identity
            or prediction.get('case_groups')!=expected_cases
            or len(rows)!=len(expected_cases) or summary.get('case_count')!=len(expected_cases)
            or {r['case_id']:r['group_id'] for r in rows}!=expected_cases):
        raise ValueError('Cached results do not match this checkpoint, split and complete cohort')
    return True


def verify_controls(root, name, *, validate_dataset=True):
    from .config import resolve_config, config_digest
    from .acquisition import collect_acquisitions
    root=Path(root);info=read_json(root/'run.json');arm=root/name
    for key in ('source_manifest','phase2_mapping'):
        if sha256(info[key])!=info[key+'_sha256']:raise ValueError('Experiment source manifest changed')
    if config_digest(resolve_config(read_json(arm/'config.json')))!=info['config_digest']:
        raise ValueError('Training controls changed for an acquisition arm')
    span,views=ARMS[name]
    if (info['arms'][name]['span_deg'],info['arms'][name]['views'])!=(span,views):
        raise ValueError('Acquisition matrix changed')
    for path,digest in info['code_sha256'].items():
        if sha256(Path(info['code_root'])/path)!=digest:raise ValueError('Frozen experiment code changed')
    if not validate_dataset:return
    identity=read_json(arm/'raw/build_identity.json')
    if (identity['span_deg'],identity['views'])!=(span,views):raise ValueError('Dataset acquisition differs from arm')
    for path,digest in identity['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Dataset source model or geometry changed')
    expected,splits=collect_acquisitions(info['source_manifest'],info['phase2_mapping'],span=span,views=views)
    if read_json(arm/'raw/splits.json')!=splits or splits['folds']['0']!=info['split_groups']:
        raise ValueError('Dataset partition differs from the saved Phase 2 split')
    actual=read_json(arm/'raw/manifest.json')['cases']
    if len(actual)!=len(expected):raise ValueError('Dataset case inventory changed')
    by_id={r['case_id']:r for r in actual}
    provenance=read_json(arm/'raw/source_mapping.json')
    if len(by_id)!=len(expected) or len(provenance)!=len(expected):raise ValueError('Duplicate or missing acquisition IDs')
    mapped={r['case_id']:r for r in provenance}
    for row in expected:
        case=by_id.get(row['case_id']);origin=mapped.get(row['case_id'])
        if (case is None or origin is None or case['group_id']!=row['group_id'] or case['angles_deg']!=row['angles_deg']
                or any(origin.get(k)!=v for k,v in row.items())
                or case['projection_source']!='simulated_stent_projection' or case['input_kind']!='fdk_sparse'):
            raise ValueError('Dataset acquisition provenance differs from the paired experiment')


def paired_differences(reference, arm, *, metrics=METRICS):
    base = {row['case_id']: row for row in reference}
    other = {row['case_id']: row for row in arm}
    if len(base) != len(reference) or len(other) != len(arm) or set(base) != set(other):
        raise ValueError('Paired comparison requires identical distinct case IDs')
    rows = []
    for cid in sorted(base):
        if base[cid]['group_id'] != other[cid]['group_id']:
            raise ValueError('Paired group identities differ')
        row = dict(case_id=cid, group_id=base[cid]['group_id'])
        for key in metrics:
            a, b = base[cid].get(key), other[cid].get(key)
            row['delta_'+key] = float(b)-float(a) if a not in ('',None) and b not in ('',None) else None
        rows.append(row)
    return rows


def initialize(root):
    root = Path(root).resolve()
    if (root/'run.json').exists():
        raise FileExistsError(root/'run.json')
    if shutil.disk_usage(root).free < 250 * 2**30:
        raise OSError('Reserve at least 250 GiB for five full-grid datasets and results')
    template = PROJECT/'training_data/training/nnunet/phase3/Dataset603_Phase3FDKThree/results/ablation_gdl_2x2_70ep_20261114/gdl_u1_pt1/config.json'
    config = read_json(template)
    from .config import resolve_config, config_digest
    resolved = resolve_config(config)
    if (config['cnn']['patch_size'] != [192]*3 or config['cnn']['batch_size'] != 2
            or config['training']['max_optimizer_steps'] != 17500):
        raise ValueError('Unexpected source training template')
    from .prior_weights import prepare_prior_artifact
    prepare_prior_artifact(PROJECT/'model_outputs/models/phase3_nnunet/model_swinvit.pt', root/'prior_artifact.json')
    snapshot = root/'code/nnunet'
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ('nnunetv2', 'configs'):
        shutil.copytree(SOURCE/name, snapshot/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(SOURCE/'acquisition_launch.py', snapshot/'acquisition_launch.py')
    source_hashes = {str(p.relative_to(snapshot)): sha256(p) for p in snapshot.rglob('*') if p.is_file()}
    dataset = PROJECT/'training_data/training/generated/phase1_200'
    mapping = PROJECT/'training_data/training/nnunet/phase2/Dataset503_MarkerGuided_fold_0/raw/Dataset503_MarkerGuided_fold_0/case_mapping.json'
    manifest = dataset/'phase1/manifest.json'
    from .acquisition import collect_acquisitions
    records, splits = collect_acquisitions(manifest, mapping, span=60, views=3)
    for index, (name, (span, views)) in enumerate(ARMS.items()):
        arm = root/name; arm.mkdir(exist_ok=True)
        write_json(arm/'config.json', config)
        smoke = json.loads(json.dumps(config))
        smoke['run_kind'] = 'smoke'
        smoke['training'].update(max_optimizer_steps=2, updates_per_epoch=2)
        resolve_config(smoke)
        write_json(arm/'smoke_config.json', smoke)
    info = dict(schema_version='phase3_acquisition_run_v1', created_unix=time.time(),
                code_root=str(snapshot), code_sha256=source_hashes,
                phase3_python=PHASE3_PYTHON, tigre_python=TIGRE_PYTHON,
                source_root=str(dataset), source_manifest=str(manifest), phase2_mapping=str(mapping),
                source_manifest_sha256=sha256(manifest), phase2_mapping_sha256=sha256(mapping),
                config_digest=config_digest(resolved), source_template=str(template),
                arms={name:dict(span_deg=span,views=views,dataset_id=920+i,smoke_dataset_id=930+i)
                      for i,(name,(span,views)) in enumerate(ARMS.items())},
                split_case_counts={s:sum(r['split']==s for r in records) for s in ('train','validation','test')},
                split_groups=splits['folds']['0'], formal_steps=17500, epochs=70,
                interpretation='Ideal simulated implant projections; descriptive paired effects, two test stent models and one seed.')
    write_json(root/'run.json', info)
    print(json.dumps(info, indent=2))


def _environment(info, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=info['code_root'],
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               nnUNet_compile='false', nnUNet_n_proc_DA='0', PYTHONUNBUFFERED='1')
    return env


def _state(arm, stage, **values):
    write_json(arm/'status.json', dict(stage=stage, updated_unix=time.time(), worker_pid=os.getpid(), **values))


def _command(arm, name, args, env, *, cwd, gpu, lock_fds=()):
    _state(arm, name, state='running', physical_gpu=gpu, command=args)
    started = time.monotonic()
    env=dict(env,PHASE3_LOCK_FDS=','.join(str(fd) for fd in lock_fds))
    with (arm/f'{name}.log').open('a') as log:
        p = subprocess.Popen(args, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(lock_fds))
        (arm/f'{name}.pid').write_text(str(p.pid)+'\n')
        code = p.wait()
    result = dict(returncode=code, elapsed_seconds=time.monotonic()-started,
                  command=args, pid=p.pid, physical_gpu=gpu, finished_unix=time.time())
    write_json(arm/f'{name}_exit.json', result)
    if code:
        raise RuntimeError(f'{name} failed with exit code {code}; see {arm/name}.log')


def _model_folder(prepared_path):
    data = read_json(prepared_path)
    return Path(data['results_dir'])/f'nnUNetTrainerPhase3__{data["plans_name"]}__3d_fullres'


def _prepared_valid(path, config):
    if not path.is_file(): return False
    from .config import resolve_config, config_digest
    prepared = read_json(path)
    if not prepared['preprocessed_complete']: return False
    if prepared['config_digest'] != config_digest(resolve_config(read_json(config))):
        raise ValueError('Prepared configuration changed')
    return True


def prepare_child(root, name, smoke=False):
    import torch
    torch.set_num_threads(2)
    from .config import resolve_config
    from .data_conversion import prepare_raw_fold, PreparedFold
    from .manifest import load_manifest, load_splits, validate_data
    from .planning import prepare_training_fold
    from .autoplan import plan_for_prepared
    root, info = Path(root), read_json(Path(root)/'run.json')
    arm = root/name
    output = arm/('smoke/prepared' if smoke else 'prepared')
    config_path = arm/('smoke_config.json' if smoke else 'config.json')
    if _prepared_valid(output/'prepared.json', config_path):
        return
    mp, sp = arm/'raw/manifest.json', arm/'raw/splits.json'
    if smoke:
        raw = read_json(mp)
        representatives = {}
        for case in raw['cases']:
            representatives.setdefault(case['group_id'], case)
        raw['cases'] = list(representatives.values())
        mp = arm/'smoke/manifest.json'
        write_json(mp, raw)
        sp = arm/'smoke/splits.json'; write_json(sp, read_json(arm/'raw/splits.json'))
    manifest, splits = load_manifest(mp, require_labels=True), load_splits(sp)
    audit = validate_data(manifest, splits, formal=True)
    write_json(output.parent/('smoke_data_validation.json' if smoke else 'data_validation.json'), audit)
    if output.exists():
        # Preserve failed partial preparation rather than overwriting its files.
        output.rename(output.with_name(output.name+f'.incomplete_{time.time_ns()}'))
    prepared = prepare_raw_fold(manifest, splits, source_fold=0,
                               dataset_id=info['arms'][name]['smoke_dataset_id' if smoke else 'dataset_id'], output=output)
    config = resolve_config(read_json(config_path))
    if not smoke:
        plan, shape, spacing = plan_for_prepared(prepared, 24)
        write_json(arm/'native_plan_audit.json', dict(native_patch=[int(v) for v in plan['patch_size']],native_batch=int(plan['batch_size']),
                   source_shape=shape, source_spacing=spacing, used_patch=list(config.cnn.patch_size),
                   used_batch=config.cnn.batch_size, note='Fixed common model budget; native planner is advisory.'))
    completed = prepare_training_fold(prepared, config)
    print(json.dumps(dict(prepared=str(output/'prepared.json'), train=len(completed.train_ids),
                          validation=len(completed.validation_ids), test=len(completed.test_ids))), flush=True)


def verify_child(root, name, smoke=False):
    import torch
    from .config import resolve_config, config_digest
    arm = Path(root)/name
    config = resolve_config(read_json(arm/('smoke_config.json' if smoke else 'config.json')))
    model = _model_folder(arm/('smoke/prepared/prepared.json' if smoke else 'prepared/prepared.json'))
    payload = torch.load(model/'fold_0/checkpoint_final.pth', map_location='cpu', weights_only=False)
    check_checkpoint_state(payload, config_digest=config_digest(config),
                           steps=config.training.max_optimizer_steps, updates=config.training.updates_per_epoch)
    best = torch.load(model/'fold_0/checkpoint_best.pth', map_location='cpu', weights_only=False)
    if best['phase3_state']['config_digest'] != config_digest(config):
        raise ValueError('Best checkpoint configuration mismatch')
    report = dict(complete=True, final_steps=payload['phase3_state']['global_step'],
                  final_epoch=payload['current_epoch'], best_step=best['phase3_state']['global_step'],
                  config_digest=config_digest(config), checkpoint_sha256=sha256(model/'fold_0/checkpoint_best.pth'))
    write_json(arm/('smoke/checkpoint_audit.json' if smoke else 'checkpoint_audit.json'), report)


def predict_child(root, name, smoke=False):
    import torch
    torch.set_num_threads(2)
    from . import cli
    arm = Path(root)/name
    prefix = arm/'smoke' if smoke else arm
    mp = prefix/'manifest.json' if smoke else arm/'raw/manifest.json'
    sp = prefix/'splits.json' if smoke else arm/'raw/splits.json'
    model = _model_folder(prefix/'prepared/prepared.json')
    output = prefix/'predictions'
    if (output/'prediction_run.json').exists():
        _validate_predictions(root,name,smoke)
        return
    if output.exists():
        output.rename(output.with_name(output.name+f'.incomplete_{time.time_ns()}'))
    timings = []
    original = cli.predict_volume
    def timed(network, volume, **kwargs):
        prediction = original(network, volume, **kwargs)
        timings.append(dict(case_id=volume.case_id, group_id=volume.group_id, **prediction.telemetry))
        write_json(prefix/'inference_timings.json', timings)
        return prediction
    cli.predict_volume = timed
    try:
        cli.main(['predict','--model-dir',str(model),'--mode','experimental','--manifest',str(mp),
                  '--splits',str(sp),'--source-fold','0','--split-name','validation' if smoke else 'test',
                  '--device','cuda','--output',str(output)])
    finally:
        cli.predict_volume = original


def _validate_predictions(root,name,smoke):
    from .manifest import load_manifest,load_splits,select_cases
    from .provenance import validate_prediction_provenance
    arm=Path(root)/name;prefix=arm/'smoke' if smoke else arm
    mp=prefix/'manifest.json' if smoke else arm/'raw/manifest.json'
    sp=prefix/'splits.json' if smoke else arm/'raw/splits.json'
    manifest=load_manifest(mp,require_labels=True);split=load_splits(sp)[0]
    split_name='validation' if smoke else 'test'
    prediction=read_json(prefix/'predictions/prediction_run.json')
    best=_model_folder(prefix/'prepared/prepared.json')/'fold_0/checkpoint_best.pth'
    expected=sha256(best)
    if prediction['checkpoint_sha256']!=expected or Path(prediction['checkpoint_path']).resolve()!=best.resolve():
        raise ValueError('Prediction was not generated by this arm selected checkpoint')
    validate_prediction_provenance(prediction,manifest,split,source_fold=0,split_name=split_name)
    cases={c.case_id:c.group_id for c in select_cases(manifest,split,split_name)}
    return prediction,cases,expected


def results_child(root,name,smoke=False,*,require_audit=False):
    arm=Path(root)/name;prefix=arm/'smoke' if smoke else arm
    prediction,cases,checkpoint=_validate_predictions(root,name,smoke)
    summary=read_json(prefix/'metrics/summary.json')
    with (prefix/'metrics/metrics_per_case.csv').open() as stream:rows=list(csv.DictReader(stream))
    check_result_identity(prediction,summary,rows,expected_checkpoint=checkpoint,expected_cases=cases,
                          split_name='validation' if smoke else 'test')
    files=['predictions/prediction_run.json','metrics/summary.json','metrics/metrics_per_case.csv','metrics/metrics_per_group.json']
    hashes={p:sha256(prefix/p) for p in files}
    audit_path=prefix/'results_audit.json'
    if audit_path.exists():
        old=read_json(audit_path)
        if old['file_sha256']!=hashes or old['checkpoint_sha256']!=checkpoint:
            raise ValueError('Previously audited test metrics changed')
    elif require_audit:
        raise ValueError('Test metrics lack a completion/provenance audit')
    else:
        write_json(audit_path,dict(complete=True,checkpoint_sha256=checkpoint,file_sha256=hashes,
                                   expected_cases=cases,verified_unix=time.time()))
    return summary,rows


def _wait_for_free_gpu(arm, gpu):
    while True:
        used = int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        if used < 1024: return
        _state(arm, 'waiting_for_gpu', state='waiting', physical_gpu=gpu, used_memory_mib=used)
        time.sleep(30)


def worker(root, name, gpu, stop_after='complete'):
    root = Path(root).resolve(); info = read_json(root/'run.json'); arm = root/name
    with exclusive_lock(arm/'worker.lock') as worker_lock:
        (arm/'worker.pid').write_text(str(os.getpid())+'\n')
        env = _environment(info, gpu)
        entry = [info['phase3_python'],str(Path(info['code_root'])/'acquisition_launch.py')]
        common = ['--run-dir',str(root),'--arm',name]
        def run_stage(stage,args):
            _command(arm,stage,args,env,cwd=info['code_root'],gpu=gpu,lock_fds=(worker_lock.fileno(),))
        try:
            verify_controls(root,name,validate_dataset=False)
            # Existing externally started builders use the same lock.
            with (arm/'build.lock').open('a+') as build_lock:
                _state(arm, 'waiting_for_data_builder', state='waiting', physical_gpu=gpu)
                fcntl.flock(build_lock, fcntl.LOCK_EX)
                if not (arm/'raw/audit.json').is_file():
                    _wait_for_free_gpu(arm, gpu)
                    a = info['arms'][name]
                    args = [info['tigre_python'],'-u',str(Path(info['code_root'])/'nnunetv2/phase3/acquisition.py'),
                            '--phase1-root',info['source_root'],'--phase1-manifest',info['source_manifest'],
                            '--phase2-mapping',info['phase2_mapping'],'--span',str(a['span_deg']),
                            '--views',str(a['views']),'--gpu','0','--output',str(arm/'raw')]
                    if (arm/'raw').exists(): args.append('--resume')
                    _command(arm,'build',args,env,cwd=info['code_root'],gpu=gpu,
                             lock_fds=(worker_lock.fileno(),build_lock.fileno()))
            verify_controls(root,name)
            audit = read_json(arm/'raw/audit.json')
            if audit['split_counts'] != info['split_case_counts'] or audit['unique_counts'] != dict(train=70,validation=10,test=20):
                raise ValueError('Full Phase 2 cohort audit failed')
            if stop_after == 'build': return
            if shutil.disk_usage(root).free < 60*2**30:
                raise OSError('At least 60 GiB free required before preparing an arm')
            run_stage('prepare',entry+['prepare-child']+common)
            if stop_after == 'prepare': return
            run_stage('smoke_prepare',entry+['prepare-child']+common+['--smoke'])
            _wait_for_free_gpu(arm,gpu)
            for smoke in (True,False):
                prefix = arm/'smoke' if smoke else arm
                prepared = prefix/'prepared/prepared.json'
                config = arm/('smoke_config.json' if smoke else 'config.json')
                model = _model_folder(prepared)
                label = 'smoke_' if smoke else ''
                if not (model/'fold_0/checkpoint_final.pth').is_file():
                    args = [info['phase3_python'],'-m','nnunetv2.phase3.cli','train','--prepared',str(prepared),
                            '--config',str(config),'--device','cuda']
                    if training_mode(model/'fold_0')=='resume': args.append('--resume')
                    else: args += ['--prior-artifact',str(root/'prior_artifact.json')]
                    run_stage(label+'train',args)
                run_stage(label+'verify',entry+['verify-child']+common+(['--smoke'] if smoke else []))
                run_stage(label+'predict',entry+['predict-child']+common+(['--smoke'] if smoke else []))
                metric_dir = prefix/'metrics'
                if metric_dir.exists() and not (prefix/'results_audit.json').exists():
                    metric_dir.rename(metric_dir.with_name('metrics.unaudited_'+str(time.time_ns())))
                if not (metric_dir/'summary.json').exists():
                    if metric_dir.exists(): metric_dir.rename(metric_dir.with_name('metrics.incomplete_'+str(time.time_ns())))
                    mp = prefix/'manifest.json' if smoke else arm/'raw/manifest.json'
                    sp = prefix/'splits.json' if smoke else arm/'raw/splits.json'
                    args = [info['phase3_python'],'-m','nnunetv2.phase3.cli','evaluate','--predictions',str(prefix/'predictions'),
                            '--manifest',str(mp),'--splits',str(sp),'--source-fold','0',
                            '--split-name','validation' if smoke else 'test','--output',str(metric_dir)]
                    run_stage(label+'evaluate',args)
                run_stage(label+'results_audit',entry+['results-child']+common+(['--smoke'] if smoke else []))
                if smoke:
                    write_json(arm/'smoke/passed.json',dict(passed=True,checkpoint_reloaded=True,
                               full_grid_prediction=True,finished_unix=time.time()))
                    if stop_after == 'smoke':
                        _state(arm,'smoke_complete',state='complete',physical_gpu=gpu); return
                    if shutil.disk_usage(root).free < 25*2**30:
                        raise OSError('At least 25 GiB free required before formal training')
                    _wait_for_free_gpu(arm,gpu)
            _state(arm,'complete',state='complete',physical_gpu=gpu)
        except BaseException as exc:
            _state(arm,'failed',state='failed',physical_gpu=gpu,error=str(exc),traceback=traceback.format_exc())
            raise


def compare(root):
    root = Path(root); summaries, cases = {}, {}
    for name in ARMS:
        path = root/name/'metrics'
        if not (path/'summary.json').exists():
            raise ValueError(f'Formal test results are not complete for {name}')
        verify_controls(root,name)
        verify_child(root,name)
        summaries[name],cases[name]=results_child(root,name,require_audit=True)
        if len(cases[name]) != 20:
            raise ValueError('Expected every one of the 20 held-out acquisition cases')
    paired = {name:paired_differences(cases['a60_n3'],cases[name]) for name in ARMS if name != 'a60_n3'}
    write_json(root/'comparison.json', dict(reference='a60_n3', summaries=summaries,paired_differences=paired,
               interpretation=read_json(root/'run.json')['interpretation']))
    rows = [dict(arm=name,**row) for name, values in paired.items() for row in values]
    with (root/'paired_differences.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def supervise(root, gpus, stop_after):
    root=Path(root).resolve();info=read_json(root/'run.json')
    if len(gpus)!=len(ARMS) or len(set(gpus))!=len(gpus):
        raise ValueError('Provide five distinct physical GPU indices, in matrix order')
    with exclusive_lock(root/'supervisor.lock'):
        (root/'supervisor.pid').write_text(str(os.getpid())+'\n')
        workers={}
        for name,gpu in zip(ARMS,gpus):
            arm=root/name
            args=[info['phase3_python'],str(Path(info['code_root'])/'acquisition_launch.py'),'worker',
                  '--run-dir',str(root),'--arm',name,'--gpu',str(gpu),'--stop-after',stop_after]
            log=(arm/'worker.log').open('a')
            p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                               env=_environment(info,gpu),cwd=info['code_root'],start_new_session=True)
            log.close();workers[name]=(p,gpu)
        while True:
            states={name:dict(pid=p.pid,physical_gpu=gpu,returncode=p.poll()) for name,(p,gpu) in workers.items()}
            write_json(root/'supervisor_status.json',dict(updated_unix=time.time(),workers=states,stop_after=stop_after))
            if all(row['returncode'] is not None for row in states.values()):break
            time.sleep(20)
        if any(row['returncode'] for row in states.values()):
            raise RuntimeError('One or more arms failed; inspect each status.json and stage logs')
        if stop_after=='complete':compare(root)


def status(root):
    root=Path(root);result={}
    for name in ARMS:
        arm=root/name;row={}
        for key,file in [('worker','status.json'),('data','raw/build_status.json')]:
            if (arm/file).exists():row[key]=read_json(arm/file)
        prepared=arm/'prepared/prepared.json'
        if prepared.exists():
            p=read_json(prepared);row['preprocessed_complete']=p['preprocessed_complete']
            if p.get('plans_name'):
                fold=_model_folder(prepared)/'fold_0'
                row['saved_epochs']=len(list(fold.glob('checkpoint_epoch_*.pth')))
                row['formal_final_exists']=(fold/'checkpoint_final.pth').exists()
        result[name]=row
    print(json.dumps(result,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','run','worker','prepare-child','verify-child','predict-child','results-child','status','compare'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--arm',choices=list(ARMS))
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--gpus',nargs='+',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after',choices=['build','prepare','smoke','complete'],default='complete')
    args=parser.parse_args(argv)
    if args.command=='init':initialize(args.run_dir)
    elif args.command=='run':supervise(args.run_dir,args.gpus or [],args.stop_after)
    elif args.command=='status':status(args.run_dir)
    elif args.command=='compare':compare(args.run_dir)
    else:
        if args.arm is None:parser.error('--arm is required')
        if args.command=='worker':
            if args.gpu is None:parser.error('--gpu is required')
            worker(args.run_dir,args.arm,args.gpu,args.stop_after)
        elif args.command=='prepare-child':prepare_child(args.run_dir,args.arm,args.smoke)
        elif args.command=='verify-child':verify_child(args.run_dir,args.arm,args.smoke)
        elif args.command=='predict-child':predict_child(args.run_dir,args.arm,args.smoke)
        elif args.command=='results-child':results_child(args.run_dir,args.arm,args.smoke)
    return 0
