import copy
import pytest
from stent_repro.configuration import resolve_config
from stent_repro.model_registry import architecture_report, build_model


def config(method='A', compact=True):
    return resolve_config({
        'schema_version': 2, 'method': method,
        'profile': f'{method}-adapted-3view', 'seed': 7,
        'geometry': {'sdd_mm': 200, 'sid_mm': 100, 'pixel_size_mm': 2,
                     'detector_shape': [16, 16], 'angles_deg': [0, 30, 60],
                     'grid': {'shape': [16,16,16], 'spacing_mm':[1,1,1],
                              'origin_mm':[-7.5]*3, 'direction':[[1,0,0],[0,1,0],[0,0,1]]}},
        'network': {'architecture':'compact-g0' if compact else ('legacy' if method=='A' else 'paper')},
        'training': {stage: {'epochs':1, 'batch_size':2, 'train_pairs':4, 'validation_pairs':2,
                    'patch_shape': [16,16] if stage=='dte' or method=='A' else [16]*3}
                    for stage in ('dte','dtr')},
        'provenance': {'purpose':'g0', 'dataset_version':'test'},
    })


def test_config_rejects_unknown_nested_and_nonfinite():
    c=config(); c['training']['dte']['epohs']=3
    with pytest.raises(ValueError, match='epohs'): resolve_config(c)
    c=config(); c['geometry']['pixel_size_mm']=float('nan')
    with pytest.raises(ValueError): resolve_config(c)


def test_angles_channels_and_resource_labels_are_enforced():
    c=config('B'); c['network']['input_channels']=1
    with pytest.raises(ValueError, match='channel'): resolve_config(c)
    c['profile']='B-adapted-one-channel'
    assert resolve_config(c)['network']['input_channels']==1
    c=config(); c['provenance']['purpose']='adapted'
    with pytest.raises(ValueError, match='g0'): resolve_config(c)
    c=config(); c['geometry']['angles_deg']=[0,60,120]
    with pytest.raises(ValueError, match='angle'): resolve_config(c)


def test_architecture_report_is_explicit_and_meta_only():
    c=config(compact=False)
    a=architecture_report(c,'dtr')
    assert (a['encoder_stages'],a['decoder_stages'],a['downsampling_steps'])==(4,3,3)
    assert a['output_shape']==[1,1,16,16]
    assert a['layers'] and a['parameter_count']>0
    c['network']['architecture']='four-downsample'
    b=architecture_report(resolve_config(c),'dte')
    assert b['downsampling_steps']==4
    assert b['architecture_status']=='interpretation_not_author_verified'


def test_bn_semantics_reach_real_model():
    c=config(); c['network']['bn_eps']=0.002; c['network']['bn_momentum']=0.03
    model=build_model(resolve_config(c),'dte')
    assert model.encoder[0][1].eps==0.002
    assert model.encoder[0][1].momentum==0.03
    assert config('B')['training']['dtr']['selection_metric']=='dice'
