import json
from pathlib import Path
import numpy as np
import pytest
from stent_repro.provenance import fingerprint_files, verify_fingerprints
from stent_repro.workflow import train_workflow, prepare_workflow
from stent_repro.io import save_projections,save_volume


def raw_cases(tmp_path):
    rows=[]
    for i,split in enumerate(('train','validation')):
        mask=np.zeros((16,16,16),np.uint8);mask[5:11,6:10,7:9]=1
        save_volume(tmp_path/f't{i}.npz',mask,{'spacing_mm':[1]*3,'origin_mm':[-7.5]*3,'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,16,16),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        rows.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}',
                     'projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    p=tmp_path/'raw.json';p.write_text(json.dumps({'cases':rows}),encoding='utf8');return p


def test_fingerprints_include_content_and_sidecar_creation(tmp_path):
    p=tmp_path/'x.npy';np.save(p,np.ones(2))
    old=fingerprint_files([p]);verify_fingerprints(old)
    Path(str(p)+'.json').write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='changed'):verify_fingerprints(old)


def test_resolved_workflow_trains_both_stages_and_records_sources(tmp_path,workflow_config):
    c=workflow_config();manifest=raw_cases(tmp_path)
    prep=prepare_workflow(c,manifest,tmp_path/'prep')
    for stage in ('dte','dtr'):
        path=prep['dte_pool_manifest'] if stage=='dte' else prep['dtr_manifest']
        out=tmp_path/stage
        result=train_workflow(c,path,out,stage,'cpu')
        assert result['history'][0]['train_samples']==4
        for name in ['resolved_config.yaml','provenance.json','split_manifest.json','architecture.json','best.pt']:
            assert (out/name).exists()
    saved=json.loads((tmp_path/'dte'/'provenance.json').read_text(encoding='utf8'))
    assert saved['source_fingerprints']
    np.save(tmp_path/'prep'/'unrelated.npy',np.ones(3))
    # Change a referenced pool source without changing its manifest.
    pools=json.loads(Path(prep['dte_pool_manifest']).read_text(encoding='utf8'))['pools']
    changed=Path(prep['dte_pool_manifest']).parent/pools[0]['path']
    changed.write_bytes(changed.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):
        train_workflow(c,prep['dte_pool_manifest'],tmp_path/'dte','dte','cpu',resume=tmp_path/'dte'/'last.pt')


def test_prepare_rejects_wrong_grid_before_writing(tmp_path,workflow_config):
    c=workflow_config();manifest=raw_cases(tmp_path);c['geometry']['grid']['origin_mm']=[0]*3
    with pytest.raises(ValueError,match='grid'):prepare_workflow(c,manifest,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_workflow_resume_can_extend_epoch_budget(tmp_path,workflow_config):
    c=workflow_config();prep=prepare_workflow(c,raw_cases(tmp_path),tmp_path/'prep')
    out=tmp_path/'run';train_workflow(c,prep['dte_pool_manifest'],out,'dte','cpu')
    c['training']['dte']['epochs']=2
    result=train_workflow(c,prep['dte_pool_manifest'],out,'dte','cpu',resume=out/'last.pt')
    assert len(result['history'])==2
    import yaml
    assert yaml.safe_load((out/'resolved_config.yaml').read_text(encoding='utf8'))['training']['dte']['epochs']==2
