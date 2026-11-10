import json
import shutil
import pytest
from phase_3.tests.fdk_fixtures import synthetic_dataset


@pytest.mark.parametrize('damage',['transitive','duplicate_case','path','content','upstream_fit','upstream_selection','upstream_calibration'])
def test_transitive_sources_and_upstream_leakage_rejected(tmp_path,damage):
    from phase_3.fdk_unet.manifest import load_manifest,load_splits,validate_data
    mp,sp=synthetic_dataset(tmp_path)
    r=json.loads(mp.read_text());cases=r['cases']
    if damage=='transitive':
        cases[1]['source_model_ids']=cases[0]['source_model_ids'];cases[1]['patient_id']=cases[2]['patient_id']='patient'
    if damage=='duplicate_case':cases[1]['case_id']=cases[0]['case_id']
    if damage=='path':cases[3]['image']=cases[0]['image']
    if damage=='content':shutil.copyfile(tmp_path/cases[0]['image'],tmp_path/cases[3]['image'])
    if damage.startswith('upstream'):
        cases[3].update(projection_source='predicted_mask',upstream_run_id='run')
        r['upstream_registry']['run']={k:[] for k in ('fit_group_ids','selection_group_ids','calibration_group_ids')}
        r['upstream_registry']['run'][damage.split('_')[1]+'_group_ids']=['g3']
    mp.write_text(json.dumps(r))
    with pytest.raises(ValueError):validate_data(load_manifest(mp,require_labels=True),load_splits(sp),formal=True)


def test_valid_splits_and_source_evidence(tmp_path):
    from phase_3.fdk_unet.manifest import load_manifest,load_splits,validate_data
    mp,sp=synthetic_dataset(tmp_path)
    assert validate_data(load_manifest(mp,require_labels=True),load_splits(sp),formal=False)['case_count']==4
    r=json.loads(mp.read_text());r['group_provenance']={};mp.write_text(json.dumps(r))
    with pytest.raises(ValueError):validate_data(load_manifest(mp,require_labels=True),load_splits(sp),formal=True)
    s=json.loads(sp.read_text());s['folds']['0']['train'].append('g3');sp.write_text(json.dumps(s))
    with pytest.raises(ValueError):validate_data(load_manifest(mp,require_labels=True),load_splits(sp),formal=False)


def test_import_requires_source_mapping(tmp_path):
    from phase_3.fdk_unet.manifest import import_nnunet,load_manifest
    root=tmp_path/'source';mp,_=synthetic_dataset(root)
    (root/'imagesTr').mkdir();(root/'labelsTr').mkdir()
    records=json.loads(mp.read_text())['cases']
    for c in records:
        shutil.copy(root/c['image'],root/'imagesTr'/c['image']);shutil.copy(root/c['label'],root/'labelsTr'/c['label'])
    with pytest.raises(ValueError):import_nnunet(root,None,tmp_path/'out')
    mapping=tmp_path/'mapping.json'
    mapping.write_text(json.dumps({'target_semantics':'stent_struts','cases':{c['case_id']:c for c in records}}))
    result=import_nnunet(root,mapping,tmp_path/'out')
    assert len(load_manifest(result,require_labels=True).cases)==4


def test_manifest_snapshot_roundtrips_without_json_coercion(tmp_path):
    from phase_3.fdk_unet.manifest import load_manifest,manifest_from_dict
    mp,_=synthetic_dataset(tmp_path)
    m=load_manifest(mp,require_labels=True)
    assert manifest_from_dict(m.to_dict(),tmp_path).to_dict()==m.to_dict()


def test_formal_comparison_rejects_mixed_projection_sources(tmp_path):
    from phase_3.fdk_unet.manifest import load_manifest,load_splits,validate_data
    mp,sp=synthetic_dataset(tmp_path);raw=json.loads(mp.read_text())
    raw['cases'][1]['projection_source']='measured';mp.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='projection'):validate_data(load_manifest(mp,require_labels=True),load_splits(sp),formal=True)
