from pathlib import Path
import pytest
from stent_repro.voth2022.config import RunConfig, split_scenes, preflight_memory

ROOT = Path(__file__).resolve().parents[2]

def test_profiles():
    original = RunConfig.load(ROOT / 'configs/voth2022/original_3view_60step.yaml')
    adapted = RunConfig.load(ROOT / 'configs/voth2022/adapted_3view_0_30_60.yaml')
    assert original.angles_deg == [0, 60, 120]
    assert adapted.angles_deg == [0, 30, 60]
    assert original.geometry['sdd_mm'] == 1100
    assert original.training['dte']['batch_size'] == 24
    assert original.training['dtr']['batch_size'] == 8

def test_split_keeps_model_together():
    result = split_scenes(['s1','s2','s3','s4'], ['m1','m1','m2','m3'], (.7,.3), 42)
    train = {x['model_id'] for x in result['train']}
    val = {x['model_id'] for x in result['validation']}
    assert train.isdisjoint(val)
    assert sum(map(len, result.values())) == 4

def test_memory_preflight_does_not_shrink():
    report = preflight_memory(cuda_available=False)
    assert report['ready'] is False
    assert 'CUDA' in report['reason']

def test_multiple_models_connected_scenes_stay_together():
    result = split_scenes(['a','b','c','d'], [['stent1','wire1'],['stent2','wire1'],['stent3','wire2'],['stent4']], (.5,.5), 7)
    group = {row['scene_id']: name for name, rows in result.items() for row in rows}
    assert group['a'] == group['b']
    assert len(group) == 4
