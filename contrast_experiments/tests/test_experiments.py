import pytest
from stent_repro.experiments import experiment_matrix, run_matrix


def test_ablation_matrix_has_independent_ids_and_seeds(workflow_config):
    jobs=experiment_matrix(workflow_config(),workflow_config('B'),[17,42])
    assert len(jobs)==10 and len({j['run_id'] for j in jobs})==10
    assert {j['config']['network']['slice_channels'] for j in jobs if j['config']['method']=='A'}=={1,3,5}
    assert {j['config']['network']['input_channels'] for j in jobs if j['config']['method']=='B'}=={1,3}
    assert all(j['config']['profile']=='B-adapted-one-channel' for j in jobs if j['config']['method']=='B' and j['config']['network']['input_channels']==1)
    with pytest.raises(ValueError,match='unique'):experiment_matrix(workflow_config(),workflow_config('B'),[1,1])


def test_dry_run_is_reviewable_and_refuses_overwrite(tmp_path,workflow_config):
    jobs=experiment_matrix(workflow_config(),workflow_config('B'),[17])
    r=run_matrix(jobs,{},tmp_path/'matrix',execute=False)
    assert r['executed'] is False and (tmp_path/'matrix'/'matrix.json').exists()
    with pytest.raises(FileExistsError):run_matrix(jobs,{},tmp_path/'matrix',execute=False)
