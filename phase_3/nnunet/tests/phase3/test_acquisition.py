import json
from pathlib import Path

import numpy as np
import pytest

from nnunetv2.phase3.manifest import load_manifest, load_splits, validate_data
from .fixtures import synthetic_dataset


@pytest.mark.parametrize('angles', [[0., 60.], [0., 15., 30., 45., 60.]])
def test_sparse_fdk_contract_accepts_actual_view_count(tmp_path, angles):
    mp, sp = synthetic_dataset(tmp_path)
    raw = json.loads(mp.read_text())
    for case in raw['cases']:
        case.update(input_kind='fdk_sparse', angles_deg=angles)
    mp.write_text(json.dumps(raw))
    manifest = load_manifest(mp, require_labels=True)
    assert manifest.cases[0].input_kind == 'fdk_sparse'
    assert manifest.cases[0].angles_deg == tuple(angles)
    assert validate_data(manifest, load_splits(sp), formal=True)['case_count'] == 4


@pytest.mark.parametrize('angles', [[], [30.], [0., 0.], [60., 0.], [0., float('nan')]])
def test_sparse_fdk_contract_rejects_missing_or_invalid_angles(tmp_path, angles):
    mp, _ = synthetic_dataset(tmp_path)
    raw = json.loads(mp.read_text())
    raw['cases'][0].update(input_kind='fdk_sparse', angles_deg=angles)
    mp.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_manifest(mp, require_labels=True)


@pytest.mark.parametrize('span,views,expected', [
    (45, 3, [7.5, 30, 52.5]), (75, 3, [-7.5, 30, 67.5]),
    (60, 2, [0, 60]), (60, 5, [0, 15, 30, 45, 60]),
])
def test_angles_keep_center_and_exact_endpoints(span, views, expected):
    from nnunetv2.phase3.acquisition import acquisition_angles
    np.testing.assert_array_equal(acquisition_angles(30, span, views), expected)


def cohort(tmp_path):
    cases, mapping = [], []
    for group, split in [('a', 'train'), ('b', 'val'), ('c', 'test')]:
        case = dict(case_id=f'head_{group}_v0_3_6', group_id=group,
                    metadata=dict(sample='head', stent_variant='001', source_split='test' if split == 'test' else 'train'),
                    views=[dict(view_id=i, angle_deg=a) for i, a in [(0, 0), (3, 30), (6, 60)]])
        cases.append(case)
        mapping.extend(dict(case_id=case['case_id'], view_id=v['view_id'],
                            angle_deg=v['angle_deg'], group_id=group, split=split)
                       for v in case['views'])
    mp, pp = tmp_path/'manifest.json', tmp_path/'mapping.json'
    mp.write_text(json.dumps(dict(cases=cases, folds=dict(fold_0=dict(train=['a'], val=['b'], test=['c'])))))
    pp.write_text(json.dumps(mapping))
    return mp, pp


def test_cohort_preserves_phase2_case_inventory_and_test_groups(tmp_path):
    from nnunetv2.phase3.acquisition import collect_acquisitions
    mp, pp = cohort(tmp_path)
    records, splits = collect_acquisitions(mp, pp, span=75, views=3)
    assert [r['split'] for r in records] == ['train', 'validation', 'test']
    assert [r['case_id'] for r in records] == ['head_a_v0_3_6', 'head_b_v0_3_6', 'head_c_v0_3_6']
    assert splits['folds']['0']['test'] == ['c']
    assert records[-1]['angles_deg'] == [-7.5, 30., 67.5]


@pytest.mark.parametrize('damage', ['missing', 'extra', 'wrong_split', 'wrong_angle', 'duplicate'])
def test_cohort_rejects_phase2_inventory_or_split_mismatch(tmp_path, damage):
    from nnunetv2.phase3.acquisition import collect_acquisitions
    mp, pp = cohort(tmp_path)
    rows = json.loads(pp.read_text())
    if damage == 'missing': rows.pop()
    if damage == 'extra': rows.append(dict(rows[-1], case_id='extra'))
    if damage == 'wrong_split': rows[-1]['split'] = 'train'
    if damage == 'wrong_angle': rows[-1]['angle_deg'] = 50
    if damage == 'duplicate': rows.append(rows[-1])
    pp.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        collect_acquisitions(mp, pp, span=60, views=2)
