import json
import numpy as np
import pytest
from stent_repro.simulation import generate_scene, write_simulated_dataset
from stent_repro.io import load_projections


@pytest.mark.parametrize('kind',['stent','wire','coil','mixed'])
def test_generated_geometry_is_reproducible_nonempty_and_in_mm(workflow_config,kind):
    c=workflow_config('B');a=generate_scene(c,1,kind);b=generate_scene(c,1,kind)
    np.testing.assert_array_equal(a['mask'],b['mask'])
    assert 0<a['mask'].sum()<a['mask'].size
    assert a['vertices'].shape[1]==3 and a['faces'].shape[1]==3
    assert a['parameters']['units']=='mm' and a['parameters']['author_original'] is False
    assert a['mask'].shape==tuple(c['geometry']['grid']['shape'])


def test_dataset_split_independent_and_inference_has_no_truth(tmp_path,workflow_config):
    p=write_simulated_dataset(workflow_config(),tmp_path/'dataset',[2,1,1])
    rows=json.loads(p.read_text(encoding='utf8'))['cases']
    assert len({r['model_ids'][0] for r in rows})==4
    inference=json.loads((p.parent/'inference.json').read_text(encoding='utf8'))
    assert len(inference['cases'])==1 and 'truth' not in inference['cases'][0]
    data,meta=load_projections(p.parent/inference['cases'][0]['projections'])
    assert data.shape==(3,16,16) and meta['units']=='line_integral'
    assert len(list(p.parent.glob('*.stl')))==4
    with pytest.raises(FileExistsError):write_simulated_dataset(workflow_config(),p.parent,[2,1,1])
