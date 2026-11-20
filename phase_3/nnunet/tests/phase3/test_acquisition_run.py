import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_exclusive_run_lock_blocks_duplicate_launch_and_releases(tmp_path):
    from nnunetv2.phase3.acquisition_run import exclusive_lock
    with exclusive_lock(tmp_path/'worker.lock'):
        with pytest.raises(RuntimeError, match='already running'):
            with exclusive_lock(tmp_path/'worker.lock'):
                pytest.fail('Duplicate worker acquired lock')
    with exclusive_lock(tmp_path/'worker.lock'):
        pass


def test_complete_checkpoint_requires_exact_budget_and_config():
    from nnunetv2.phase3.acquisition_run import check_checkpoint_state
    payload = dict(current_epoch=70, phase3_state=dict(next_epoch=70, global_step=17500,
        config_digest='wanted', max_optimizer_steps=17500, updates_per_epoch=250))
    assert check_checkpoint_state(payload, config_digest='wanted', steps=17500, updates=250)
    for key, value in [('global_step', 17250), ('config_digest', 'other'), ('next_epoch', 69)]:
        bad = json.loads(json.dumps(payload));bad['phase3_state'][key] = value
        with pytest.raises(ValueError):
            check_checkpoint_state(bad, config_digest='wanted', steps=17500, updates=250)


def test_pair_metrics_matches_case_ids_and_keeps_stent_groups():
    from nnunetv2.phase3.acquisition_run import paired_differences
    base = [dict(case_id='a', group_id='stent5', dice=.5), dict(case_id='b',group_id='w_stent',dice=.8)]
    arm = [dict(case_id='b', group_id='w_stent', dice=.7), dict(case_id='a',group_id='stent5',dice=.7)]
    rows = paired_differences(base, arm, metrics=['dice'])
    assert rows[0]['case_id'] == 'a'
    assert rows[0]['delta_dice'] == pytest.approx(.2)
    assert rows[1]['delta_dice'] == pytest.approx(-.1)
    with pytest.raises(ValueError):
        paired_differences(base, arm[:1], metrics=['dice'])
    arm[0]['group_id'] = 'stent5'
    with pytest.raises(ValueError):
        paired_differences(base, arm, metrics=['dice'])


def test_stage_holds_worker_lock_after_worker_is_killed(tmp_path):
    from nnunetv2.phase3.acquisition_run import exclusive_lock
    source = Path(__file__).resolve().parents[2]
    code = """
from pathlib import Path
import os,sys
from nnunetv2.phase3.acquisition_run import exclusive_lock, _command
arm=Path(sys.argv[1])
with exclusive_lock(arm/'worker.lock') as lock:
    _command(arm,'stage',[sys.executable,'-c','import time; time.sleep(20)'],
             os.environ.copy(),cwd=arm,gpu=0,lock_fds=(lock.fileno(),))
"""
    env = os.environ.copy();env['PYTHONPATH'] = str(source)
    parent = subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],env=env)
    child = None
    try:
        deadline = time.monotonic()+5
        while not (tmp_path/'stage.pid').exists() and parent.poll() is None and time.monotonic()<deadline:
            time.sleep(.02)
        assert (tmp_path/'stage.pid').exists(), 'Stage must start and hold an inherited lock'
        child = int((tmp_path/'stage.pid').read_text())
        parent.kill();parent.wait()
        with pytest.raises(RuntimeError,match='already running'):
            with exclusive_lock(tmp_path/'worker.lock'):
                pytest.fail('Orphaned stage lost exclusivity')
    finally:
        if parent.poll() is None:parent.kill();parent.wait()
        if child is not None:
            try:os.kill(child,15)
            except ProcessLookupError:pass


def test_training_retry_preserves_partial_fold_without_checkpoint(tmp_path):
    from nnunetv2.phase3.acquisition_run import training_mode
    fold=tmp_path/'model/fold_0';fold.mkdir(parents=True)
    (fold/'training_log.txt').write_text('failed before first optimizer update')
    assert training_mode(fold)=='fresh'
    assert not fold.exists()
    archived=list(fold.parent.glob('fold_0.incomplete_*'))
    assert len(archived)==1 and (archived[0]/'training_log.txt').is_file()
    fold.mkdir();(fold/'checkpoint_latest.pth').write_text('checkpoint')
    assert training_mode(fold)=='resume'
    assert (fold/'checkpoint_latest.pth').exists()


def test_cached_metric_identity_rejects_wrong_checkpoint_and_test_split():
    from nnunetv2.phase3.acquisition_run import check_result_identity
    prediction=dict(checkpoint_sha256='best',split_digest='split',source_fold=0,split_name='test',
                    case_groups={'one':'stent5','two':'w_stent'})
    summary=dict(case_count=2,provenance=dict(checkpoint_sha256='best',split_digest='split',source_fold=0,split_name='test'))
    rows=[dict(case_id='one',group_id='stent5'),dict(case_id='two',group_id='w_stent')]
    assert check_result_identity(prediction,summary,rows,expected_checkpoint='best',
                                 expected_cases={'one':'stent5','two':'w_stent'},split_name='test')
    for key,value in [('checkpoint_sha256','stale'),('split_name','validation')]:
        broken=json.loads(json.dumps(summary));broken['provenance'][key]=value
        with pytest.raises(ValueError):
            check_result_identity(prediction,broken,rows,expected_checkpoint='best',
                                  expected_cases={'one':'stent5','two':'w_stent'},split_name='test')


@pytest.mark.parametrize('damage',['config','source','angle'])
def test_control_audit_rejects_drift_before_execution(tmp_path, damage):
    from nnunetv2.phase3.acquisition_run import verify_controls
    from nnunetv2.phase3.acquisition import sha256
    from nnunetv2.phase3.config import config_digest, resolve_config
    from .fixtures import make_config
    source=tmp_path/'source.json';source.write_text('{}')
    mapping=tmp_path/'mapping.json';mapping.write_text('[]')
    arm=tmp_path/'a60_n3';arm.mkdir()
    config=make_config();(arm/'config.json').write_text(json.dumps(config))
    info=dict(source_manifest=str(source),phase2_mapping=str(mapping),source_manifest_sha256=sha256(source),
              phase2_mapping_sha256=sha256(mapping),config_digest=config_digest(resolve_config(config)),
              code_root=str(tmp_path),code_sha256={},arms={'a60_n3':dict(span_deg=60,views=3)})
    (tmp_path/'run.json').write_text(json.dumps(info))
    if damage=='config':
        config['training']['foreground_probability']=.1
        (arm/'config.json').write_text(json.dumps(config))
    if damage=='source':source.write_text('{"changed": true}')
    if damage=='angle':
        info['arms']['a60_n3']['span_deg']=45;(tmp_path/'run.json').write_text(json.dumps(info))
    with pytest.raises(ValueError):verify_controls(tmp_path,'a60_n3',validate_dataset=False)
