import numpy as np
import pytest
from stent_repro.io import load_projections, save_projections, load_volume, save_volume

def test_projection_npz_roundtrip_preserves_values_and_geometry(tmp_path):
    pixels = np.arange(24, dtype=np.float32).reshape(3,2,4) / 10
    metadata = {'angles_deg':[0,30,60], 'pixel_spacing_mm':[0.3,0.3], 'geometry':{'sdd_mm':1100,'sid_mm':600}, 'units':'line_integral'}
    path = tmp_path / 'projections.npz'
    save_projections(path, pixels, metadata)
    loaded, info = load_projections(path)
    np.testing.assert_array_equal(loaded, pixels)
    assert info == metadata

def test_single_projection_npy_roundtrip(tmp_path):
    pixels = np.array([[0,256],[65535,42]], dtype=np.uint16)
    path = tmp_path / 'image.npy'
    save_projections(path, pixels, {'angles_deg':[45]})
    result, metadata = load_projections(path)
    np.testing.assert_array_equal(result, pixels[None])
    assert metadata['angles_deg'] == [45]

def test_volume_roundtrip_preserves_axes_and_physical_grid(tmp_path):
    arr = np.arange(24,dtype=np.float32).reshape(2,3,4)
    metadata = {'spacing_mm':[0.6,0.7,0.8], 'origin_mm':[-1,2,3], 'direction':[[1,0,0],[0,1,0],[0,0,1]], 'units':'probability'}
    for suffix in ('.npy','.npz'):
        path = tmp_path / ('volume'+suffix)
        save_volume(path, arr, metadata)
        actual, info = load_volume(path)
        np.testing.assert_array_equal(actual, arr)
        assert info == metadata

def test_reject_mismatched_angles_and_bad_grid(tmp_path):
    with pytest.raises(ValueError, match='angles'):
        save_projections(tmp_path/'bad.npz', np.zeros((3,2,2)), {'angles_deg':[0,30]})
    with pytest.raises(ValueError, match='spacing'):
        save_volume(tmp_path/'bad.npz', np.zeros((2,2,2)), {'spacing_mm':[1,0,1]})

def test_read_external_npy_without_inventing_geometry(tmp_path):
    path = tmp_path / 'external.npy'
    np.save(path, np.ones((2,3),dtype=np.float32))
    result, metadata = load_projections(path)
    assert result.shape == (1,2,3)
    assert metadata == {}

def test_refuse_object_arrays_and_invalid_dimensions(tmp_path):
    with pytest.raises(ValueError):
        save_volume(tmp_path/'x.npz', np.array([[[object()]]]), {})
    with pytest.raises(ValueError):
        save_volume(tmp_path/'x.npz', np.zeros((2,2)), {})

def test_manifest_loads_pairs_relative_to_manifest(tmp_path):
    import json
    from stent_repro.io import load_manifest, load_case
    save_projections(tmp_path/'p.npz', np.ones((3,2,4)), {'angles_deg':[0,30,60]})
    save_volume(tmp_path/'v.npz', np.ones((2,3,4)), {'spacing_mm':[.6,.6,.6]})
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'id':'case1','projections':'p.npz','volume':'v.npz','split':'train'}]}), encoding='utf-8')
    cases = load_manifest(manifest)
    case = load_case(cases[0])
    assert case['id'] == 'case1'
    assert case['projections'].shape == (3,2,4)
    assert case['volume'].shape == (2,3,4)
    assert case['projection_metadata']['angles_deg'] == [0,30,60]
    assert case['volume_metadata']['spacing_mm'] == [.6,.6,.6]

def test_manifest_rejects_duplicate_cases_and_invalid_split(tmp_path):
    import json
    from stent_repro.io import load_manifest
    manifest = tmp_path/'manifest.json'
    row = {'id':'a', 'projections':'x.npy','volume':'y.npy','split':'train'}
    manifest.write_text(json.dumps({'cases':[row,row]}), encoding='utf-8')
    with pytest.raises(ValueError,match='duplicate'): load_manifest(manifest)
    row['split'] = 'whatever'
    manifest.write_text(json.dumps({'cases':[row]}), encoding='utf-8')
    with pytest.raises(ValueError,match='split'): load_manifest(manifest)

def test_save_manifest_roundtrip(tmp_path):
    from stent_repro.io import load_manifest, save_manifest
    path = tmp_path/'cases.json'
    save_manifest(path, [{'id':'s','projections':'p.npy','volume':'v.npy','split':'test','model_ids':['m1']}])
    cases = load_manifest(path)
    assert cases[0]['model_ids'] == ['m1']
    assert cases[0]['projections'] == tmp_path/'p.npy'

def test_manifest_rejects_models_shared_between_splits(tmp_path):
    from stent_repro.io import save_manifest
    cases = [{'id':'a','projections':'a.npy','volume':'a3.npy','split':'train','model_ids':['s1','w1']}, {'id':'b','projections':'b.npy','volume':'b3.npy','split':'test','model_ids':['s2','w1']}]
    with pytest.raises(ValueError, match='model.*split'):
        save_manifest(tmp_path/'manifest.json', cases)

def test_tiff_projection_and_volume_roundtrip(tmp_path):
    projections = np.arange(60,dtype=np.uint16).reshape(3,4,5)
    volume = np.arange(60,dtype=np.float32).reshape(3,4,5) / 10
    save_projections(tmp_path/'projections.tiff', projections, {'angles_deg':[0,30,60]})
    save_volume(tmp_path/'volume.tif', volume, {'spacing_mm':[.1,.2,.3]})
    a, am = load_projections(tmp_path/'projections.tiff')
    b, bm = load_volume(tmp_path/'volume.tif')
    np.testing.assert_array_equal(a, projections)
    np.testing.assert_array_equal(b, volume)
    assert am['angles_deg'] == [0,30,60]
    assert bm['spacing_mm'] == [.1,.2,.3]

def test_tiff_sequence_saves_and_reads_order_and_metadata(tmp_path):
    from stent_repro.io import save_tiff_sequence, load_tiff_sequence
    x = np.arange(36,dtype=np.uint16).reshape(3,3,4)
    save_tiff_sequence(tmp_path/'sequence', x, {'angles_deg':[0,30,60]})
    actual, metadata = load_tiff_sequence(tmp_path/'sequence')
    np.testing.assert_array_equal(actual,x)
    assert metadata == {'angles_deg':[0,30,60]}

def test_tiff_sequence_user_specified_order(tmp_path):
    import tifffile
    from stent_repro.io import load_tiff_sequence
    tifffile.imwrite(tmp_path/'a.tif', np.full((2,3), 1, dtype=np.uint16))
    tifffile.imwrite(tmp_path/'b.tif', np.full((2,3), 2, dtype=np.uint16))
    result, _ = load_tiff_sequence([tmp_path/'b.tif',tmp_path/'a.tif'], {'angles_deg':[60,0]})
    assert result[:,0,0].tolist() == [2,1]

def test_tiff_rejects_color(tmp_path):
    import tifffile
    image = np.zeros((4,5,3),dtype=np.uint8)
    tifffile.imwrite(tmp_path/'rgb.tif',image,photometric='rgb')
    with pytest.raises(ValueError, match='grayscale'):
        load_projections(tmp_path/'rgb.tif')

def test_manifest_accepts_projection_sequence_directory(tmp_path):
    from stent_repro.io import save_tiff_sequence, save_manifest, load_manifest, load_case
    save_tiff_sequence(tmp_path/'views', np.ones((3,2,2),dtype=np.uint16), {'angles_deg':[0,30,60]})
    save_volume(tmp_path/'truth.tif', np.ones((2,2,2),dtype=np.uint8), {})
    save_manifest(tmp_path/'cases.json',[{'id':'x','projections':'views','volume':'truth.tif','split':'train'}])
    result = load_case(load_manifest(tmp_path/'cases.json')[0])
    assert result['projections'].shape == (3,2,2)
    assert result['projection_metadata']['angles_deg'] == [0,30,60]

def test_reads_multipage_tiff_with_separate_series(tmp_path):
    import tifffile
    path = tmp_path/'pages.tif'
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.full((3,4),1,dtype=np.uint16))
        writer.write(np.full((3,4),2,dtype=np.uint16))
    data, _ = load_projections(path)
    assert data.shape == (2,3,4)
    assert data[:,0,0].tolist() == [1,2]

def test_single_slice_volume_tiff_roundtrip(tmp_path):
    x = np.ones((1,3,4),dtype=np.uint16)
    path = tmp_path/'single.tif'
    save_volume(path,x,{'spacing_mm':[1,1,1]})
    actual, _ = load_volume(path)
    np.testing.assert_array_equal(actual,x)
