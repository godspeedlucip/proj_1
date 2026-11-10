import copy
import json

import pytest


def test_h_matrix_contract_and_unknown_fields():
    from phase_3.fdk_unet.config import get_experiment_config, resolve_config

    assert get_experiment_config('H11').experiment_id == 'C'
    for name in ('A', 'B', 'H00', 'H10', 'H01', 'C'):
        cfg = get_experiment_config(name)
        assert cfg.patch_size == (128, 128, 128)
        assert cfg.resolved['max_optimizer_steps'] == 10000
        if name == 'A':
            continue
        r = cfg.resolved
        bottleneck = name in ('B', 'H10', 'C')
        assert r['fusion']['bottleneck_enabled'] == bottleneck
        assert r['fusion']['decoder_conditioning'] == (name in ('B', 'H01', 'C'))
        assert r['fusion']['skip_gate_input_multiplier'] == 3
        assert r['fusion']['swin_indices'] == ([1, 2, 3] if bottleneck else [1, 2])
        assert r['prior']['execute_through'] == ('layers3' if bottleneck else 'layers2')
    original = get_experiment_config('C').resolved
    for mutate in (lambda r: r.update(typo=True),
                   lambda r: r['fusion']['swin_channels'].__setitem__(2, 768),
                   lambda r: r['prior'].update(execute_through='layers2'),
                   lambda r: r.update(stage='partial_finetune'),
                   lambda r: r.update(max_optimizer_steps=2),
                   lambda r: r['prior'].update(typo=1)):
        bad = copy.deepcopy(original)
        mutate(bad)
        with pytest.raises(ValueError):
            resolve_config(bad)


def test_explicit_smoke_paths_and_full_configs(tmp_path):
    from phase_3.fdk_unet.config import get_experiment_config, load_config, resolve_config

    r = get_experiment_config('C').resolved
    r.update(run_kind='smoke', experiment_id='C-smoke', patch_size=[32]*3, max_optimizer_steps=4, validate_every=2)
    r['prior']['artifact'] = 'prior/prior.json'
    p = tmp_path / 'smoke.json'
    p.write_text(json.dumps(r))
    cfg = load_config(p)
    assert cfg.resolved['prior']['artifact'] == str((tmp_path / 'prior/prior.json').resolve())
    r['experiment_id'] = 'C'
    with pytest.raises(ValueError):
        resolve_config(r)
    from pathlib import Path
    for p in (Path(__file__).parents[1] / 'fdk_unet/configs').rglob('*.json'):
        load_config(p)


@pytest.mark.parametrize('source,label',[('A','C'),('C','B'),('A','H11'),('H01','H00')])
def test_reserved_experiment_identity_cannot_mislabel_architecture(source,label):
    from phase_3.fdk_unet.config import get_experiment_config,resolve_config
    raw=get_experiment_config(source).resolved;raw['experiment_id']=label
    with pytest.raises(ValueError):resolve_config(raw)
