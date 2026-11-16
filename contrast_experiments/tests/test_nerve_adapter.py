from dataclasses import replace
import json

import numpy as np
import pytest
import tifffile

from stent_repro.io import load_projections, load_volume
from stent_repro.nerve_adapter import NerveCase, import_nerve_dataset


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def _fixture(root, head='head01', family='family1', variant='001', source_split='train'):
    nib = pytest.importorskip('nibabel')
    head_dir = root / 'heads' / head
    stent_dir = root / 'projections' / family / variant
    model_dir = root / 'models' / family / variant
    stack_dir = root / 'stacks' / head / family / variant
    angles = [0, 30, 60]
    geometry = {'DSO_mm': 100, 'DSD_mm': 200, 'nDetector': [16, 16],
                'dDetector_mm': [1, 1], 'nVoxel': [7, 6, 5], 'dVoxel_mm': [1, 1, 1],
                'offOrigin_mm': [0, 0, 0], 'offDetector_mm': [0, 0]}
    common = {'split': source_split, 'angles_deg': angles, 'view_indices': [0, 1, 2], 'output_mode': '2D'}
    stent = {'stent_series': family, 'stent_variant': variant, 'stent_mu': 0.17, 'marker_mu': 1.02}
    _write_json(head_dir / 'preparation.json', {'sample': head, 'split': source_split})
    _write_json(head_dir / 'projection.json', {**common, 'geometry': geometry, 'sample': head})
    _write_json(stent_dir / 'projection.json', {**common, **stent, 'geometry': geometry})
    _write_json(stack_dir / 'stack.json', {**common, **stent, 'sample': head,
                 'head_geometry': geometry, 'stent_geometry': geometry, 'noise_std': 0.,
                 'implant_sigma_px': 0., 'formula': 'image = head + Gaussian(combined implant projection)'})
    raw = np.zeros((5, 6, 7), dtype=np.uint8)
    raw[3, 2, 3] = 1
    raw[3, 2, 1] = 1
    raw[3, 2, 5] = 1
    model_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(raw, np.eye(4)), model_dir / 'stent_real_grid.nii.gz')
    _write_json(model_dir / 'stent.json', {'series': family, 'variant': int(variant),
                 'split': source_split, 'volume_shape': [5, 6, 7], 'voxel_spacing_mm': [1, 1, 1],
                 'combined_voxels': 3})
    _write_json(model_dir / 'marker.json', {'markers': [
        {'center_voxel_ijk': [3, 2, 1], 'voxels_ijk': [[3, 2, 1]]},
        {'center_voxel_ijk': [3, 2, 5], 'voxels_ijk': [[3, 2, 5]]},
    ]})
    for idx, angle in enumerate(angles):
        a = np.deg2rad(angle)
        depth = 100 - np.cos(a) + 0.5 * np.sin(a)
        col = 7.5 + 200 * (-0.5 * np.cos(a) - np.sin(a)) / depth
        rows = [7.5 - 400 / depth, 7.5 + 400 / depth]
        _write_json(stent_dir / 'marker_metadata' / f'{idx:03}.json', {
            'angle_deg': angle, 'view_index': idx, 'coordinate_order': 'row_column', 'index_base': 0,
            'marker_centers_rc': [[row, col] for row in rows],
        })
        ideal = np.zeros((16, 16), np.float32)
        ideal[5:10, 7] = idx + 0.25
        head_image = np.full_like(ideal, 2)
        for directory, kind, frame in [(head_dir, 'head', head_image), (stent_dir, 'implant', ideal),
                                       (stack_dir, 'image', head_image + ideal)]:
            path = directory / 'projections' / kind / f'{angle:03}.tif'
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, frame, photometric='minisblack')
    return NerveCase(head_dir, stent_dir, stack_dir, model_dir, source_split)


def test_import_aligns_true_volume_and_preserves_native_projection_values(tmp_path):
    case = _fixture(tmp_path / 'source', source_split='test')
    output = tmp_path / 'imported'
    result = import_nerve_dataset([case], output, angles_deg=[0, 30, 60])
    raw = json.loads(result['raw'].read_text())['cases'][0]
    inference = json.loads(result['inference'].read_text())['cases'][0]
    truth = json.loads(result['test_truth'].read_text())['cases'][0]
    assert inference.keys() == {'id', 'entity_id', 'projections'}
    assert raw['model_ids'] == ['family1'] and raw['patient_id'] == 'head01'
    assert truth['entity_id'] == 'family1' and raw['id'] == inference['id'] == truth['id']
    assert raw['projection_role'] == 'measured_with_tool' and raw['registration_confirmed']
    image, meta = load_projections(output / inference['projections'])
    implant, _ = load_projections(output / truth['ideal'])
    volume, vm = load_volume(output / truth['truth'])
    np.testing.assert_array_equal(image[:, 5, 7], [2.25, 3.25, 4.25])
    np.testing.assert_array_equal(implant[:, 5, 7], [0.25, 1.25, 2.25])
    np.testing.assert_array_equal(np.argwhere(volume), [[1, 1, 2], [3, 1, 2], [5, 1, 2]])
    assert vm['origin_mm'] == [-2.5, -2.0, -3.0]
    assert meta['calibration_id'].startswith('nerve-tigre:')
    assert meta['source_kind'] == 'synthetic_topbrain_overlay'


def test_fixed_grid_uses_physical_origin_and_nearest_neighbor(tmp_path):
    case = _fixture(tmp_path / 'source')
    target = {'shape': [5, 3, 4], 'spacing_mm': [1, 1, 1], 'origin_mm': [-1.5, -1, -2],
              'direction': np.eye(3).tolist()}
    result = import_nerve_dataset([case], tmp_path / 'imported', angles_deg=[0], target_grid=target)
    raw = json.loads(result['raw'].read_text())['cases'][0]
    volume, meta = load_volume(result['raw'].parent / raw['truth'])
    assert meta['origin_mm'] == target['origin_mm']
    assert meta['source_foreground_voxels'] == meta['output_foreground_voxels'] == 3
    np.testing.assert_array_equal(np.argwhere(volume), [[0, 0, 1], [2, 0, 1], [4, 0, 1]])
    assert json.loads(result['inference'].read_text())['cases'] == []


def test_rejects_family_reuse_across_splits_even_with_distinct_variants(tmp_path):
    train = _fixture(tmp_path / 'train', variant='001')
    test = _fixture(tmp_path / 'test', head='head02', variant='002', source_split='test')
    with pytest.raises(ValueError, match='family.*multiple splits'):
        import_nerve_dataset([train, test], tmp_path / 'imported', angles_deg=[0])
    assert not (tmp_path / 'imported').exists()


def test_rejects_geometry_disagreement_and_missing_angles(tmp_path):
    case = _fixture(tmp_path / 'source')
    with pytest.raises(ValueError, match='available angles'):
        import_nerve_dataset([case], tmp_path / 'imported', angles_deg=[45])
    metadata = case.head_dir / 'projection.json'
    doc = json.loads(metadata.read_text())
    doc['geometry']['offDetector_mm'] = [1, 0]
    _write_json(metadata, doc)
    with pytest.raises(ValueError, match='offset'):
        import_nerve_dataset([case], tmp_path / 'imported', angles_deg=[0])


def test_legacy_holdout_requires_explicit_split_and_override(tmp_path):
    source = _fixture(tmp_path / 'source')
    for directory, name in [(source.head_dir, 'projection.json'), (source.stent_projection_dir, 'projection.json'),
                            (source.head_dir, 'preparation.json'), (source.stack_dir, 'stack.json'),
                            (source.model_dir, 'stent.json')]:
        doc = json.loads((directory / name).read_text())
        doc.pop('split')
        if name == 'stent.json':
            doc['mesh'] = r'ply\train\family1.ply'
        _write_json(directory / name, doc)
    case = replace(source, split='test')
    with pytest.raises(ValueError, match='legacy_source_split'):
        import_nerve_dataset([case], tmp_path / 'imported', angles_deg=[0], engineering_holdout=True)
    case = replace(case, legacy_source_split='train')
    with pytest.raises(ValueError, match='engineering_holdout'):
        import_nerve_dataset([case], tmp_path / 'imported', angles_deg=[0])
    result = import_nerve_dataset([case], tmp_path / 'imported', angles_deg=[0], engineering_holdout=True)
    evidence = json.loads(result['provenance'].read_text())['cases'][0]
    assert evidence['source_split'] == 'train' and evidence['split'] == 'test'


def test_rejects_composite_that_does_not_match_source_head_and_implant(tmp_path):
    source = _fixture(tmp_path / 'source')
    image_path = source.stack_dir / 'projections' / 'image' / '000.tif'
    tifffile.imwrite(image_path, tifffile.imread(image_path) + 0.01, photometric='minisblack')
    with pytest.raises(ValueError, match='disagree with.*overlay'):
        import_nerve_dataset([source], tmp_path / 'imported', angles_deg=[0])
    assert not (tmp_path / 'imported').exists()


def test_rejects_head_reuse_across_splits_even_with_distinct_families(tmp_path):
    train = _fixture(tmp_path / 'train')
    test = _fixture(tmp_path / 'test', family='another_family', source_split='test')
    with pytest.raises(ValueError, match='head.*multiple splits'):
        import_nerve_dataset([train, test], tmp_path / 'imported', angles_deg=[0])
