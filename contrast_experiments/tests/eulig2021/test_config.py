from pathlib import Path
import pytest
from stent_repro.eulig2021.config import RunConfig, split_models, preflight_training

ROOT = Path(__file__).resolve().parents[2]

def test_profiles_and_roundtrip(tmp_path):
    original = RunConfig.load(ROOT / 'configs/eulig2021/original_4view.yaml')
    adapted = RunConfig.load(ROOT / 'configs/eulig2021/adapted_3view_0_30_60.yaml')
    assert original.angles_deg == [0, 45, 90, 135]
    assert adapted.angles_deg == [0, 30, 60]
    assert adapted.profile == 'A-adapted-3view'
    path = tmp_path / 'config.yaml'
    adapted.save(path)
    assert RunConfig.load(path) == adapted

def test_split_is_by_model():
    split = split_models(['a', 'b', 'a', 'c', 'd'], seed=1, ratios=(.5,.25,.25))
    assert set.union(*(set(v) for v in split.values())) == {'a','b','c','d'}
    assert sum(map(len, split.values())) == 4

def test_strict_training_rejects_cpu():
    with pytest.raises(RuntimeError, match='CUDA'):
        preflight_training(cuda_available=False)
