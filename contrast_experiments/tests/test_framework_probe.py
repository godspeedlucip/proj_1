import json
import sys

import pytest

from stent_repro.framework_probe import run_framework_probe


def test_numpy_and_pytorch_use_same_weights_and_loss_reduction(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tensorflow', None)
    report = run_framework_probe()

    assert report['status'] == 'passed'
    comparison = report['comparisons']['numpy_vs_pytorch']
    assert comparison['status'] == 'passed'
    assert comparison['checks']['conv']['shape'] == [2, 5, 7, 2]
    assert comparison['checks']['upsampled']['shape'] == [2, 10, 21, 2]
    for metric in ('conv', 'batch_norm', 'relu', 'upsampled', 'probabilities',
                   'soft_dice_loss_a', 'soft_dice_loss_b', 'mae_loss'):
        assert comparison['checks'][metric]['passed']
        assert comparison['checks'][metric]['max_abs_error'] < 2e-6
    assert report['device'] == 'cpu'
    assert report['comparisons']['tensorflow_vs_pytorch']['status'] == 'unavailable'
    json.dumps(report, allow_nan=False)


def test_required_tensorflow_marks_report_failed_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tensorflow', None)
    report = run_framework_probe(require_tensorflow=True)

    assert report['status'] == 'failed'
    assert report['comparisons']['tensorflow_vs_pytorch']['reason']
    assert report['comparisons']['numpy_vs_pytorch']['status'] == 'passed'


def test_tensorflow_matches_pytorch_at_each_stage_when_available():
    pytest.importorskip('tensorflow')
    report = run_framework_probe(require_tensorflow=True)
    comparison = report['comparisons']['tensorflow_vs_pytorch']

    assert report['status'] == 'passed'
    assert comparison['status'] == 'passed'
    assert comparison['checks'].keys() == report['comparisons']['numpy_vs_pytorch']['checks'].keys()
    assert all(check['passed'] for check in comparison['checks'].values())
