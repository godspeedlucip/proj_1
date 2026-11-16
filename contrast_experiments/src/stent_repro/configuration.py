"""Versioned, strict configuration shared by the complete workflow."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

ANGLE_PATTERNS={'A-original-4view':[0,45,90,135],'A-adapted-3view':[0,30,60],
    'A-reference-2view90':[0,90],'A-reference-3view60':[0,60,120],
    'A-reference-4view30':[0,30,60,90],'A-reference-4view22_5':[0,22.5,45,67.5],
    'B-original-3view':[0,60,120],'B-one-channel-ablation':[0,60,120],
    'B-adapted-3view':[0,30,60],'B-adapted-one-channel':[0,30,60]}


def validate_profile_angles(profile,angles):
    if profile not in ANGLE_PATTERNS:raise ValueError('unknown angle profile')
    expected=ANGLE_PATTERNS[profile]
    if not isinstance(angles,list) or len(angles)!=len(expected) or any(type(x) not in (int,float) or not math.isfinite(x) for x in angles):raise ValueError('invalid profile angles')
    actual=angles if 'adapted' in profile else [(v-angles[0])%360 for v in angles]
    if any(not math.isclose(x,y,abs_tol=1e-7) for x,y in zip(actual,expected)):raise ValueError('angles do not match profile')


def validate_case_angles(config,angles):
    validate_profile_angles(config['profile'],angles)
    if config['simulation']['angular_offset_mode']=='fixed' and angles!=config['geometry']['angles_deg']:
        raise ValueError('case angles differ from fixed acquisition')


def _merge(default, supplied, path='config'):
    if not isinstance(supplied, dict):
        raise ValueError(f'{path} must be a mapping')
    unknown=set(supplied)-set(default)
    if unknown: raise ValueError(f'{path}: unknown fields {sorted(unknown)}')
    result=deepcopy(default)
    for key,value in supplied.items():
        result[key]=_merge(default[key],value,f'{path}.{key}') if isinstance(default[key],dict) else deepcopy(value)
    return result


def _number(value, name, low=0, strict=True):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value<=low if strict else value<low):
        raise ValueError(f'{name} must be finite and {">" if strict else ">="} {low}')


def _integers(values, length, name):
    if not isinstance(values,list) or len(values)!=length or any(type(x) is not int or x<1 for x in values):
        raise ValueError(f'{name} must contain {length} positive integers')


def resolve_config(source):
    raw=yaml.safe_load(Path(source).read_text(encoding='utf-8')) if isinstance(source,(str,Path)) else deepcopy(source)
    if not isinstance(raw,dict) or raw.get('method') not in ('A','B'): raise ValueError('method must be A or B')
    method=raw['method']; a=method=='A'
    defaults={
        'schema_version':2,'method':method,'profile':f'{method}-adapted-3view','seed':42,
        'geometry':{'sdd_mm':None,'sid_mm':None,'pixel_size_mm':None,'detector_shape':None,'angles_deg':None,'calibration_id':None,
                    'grid':{'shape':None,'spacing_mm':None,'origin_mm':None,'direction':[[1,0,0],[0,1,0],[0,0,1]]}},
        'network':{'architecture':'legacy' if a else 'paper','slice_channels':5,'input_channels':3,
                   'bn_eps':1e-5,'bn_momentum':0.1,'conv_bias':True,'initialization':'he_normal',
                   'dte_activation':'relu','dtr_activation':'sigmoid','dice_smoothing':1.0,'compact_base':4},
        'training':{s:{'epochs':None if a else (200 if s=='dte' else 150),
                      'batch_size':(6 if a else 24) if s=='dte' else (16 if a else 8),
                      'learning_rate':1e-5 if not a and s=='dte' else 1e-4,
                      'train_pairs':None if a else 16000,'validation_pairs':None if a else 4000,
                      'patch_shape':[384,384] if s=='dte' else ([224,224] if a else [128,128,128]),
                      'normalization':'running' if a and s=='dte' else ('volume' if a else 'none'),
                      'precision':'float32','activation_checkpointing':False,
                      'selection_metric':'mae' if s=='dte' else 'dice'} for s in ('dte','dtr')},
        'preprocessing':{'attenuation_per_mm':0.1,'integration_step_mm':0.5,
                         'photons_range':[10000,100000],'blur_sigma_px':1.4,'scatter_range':[0.0,0.0],
                         'min_background':0.01,'false_negative_rate':0.05,'false_positive_rate':0.001,
                         'false_positive_scale':0.2,'dtr_crop_shape':None,'spectral':None,
                         'error_model':'independent_pixel','correlation_sigma_px':2.0},
        'simulation':{'angular_offset_mode':'fixed','start_angle_range_deg':[0,180],
                      'dte_views':'same','material_density_range_g_cm3':None},
        'inference':{'threshold':0.5,'slice_batch':1,'max_volume_voxels':16777216,'retain_intermediates':True,
                     'tiling':{'enabled':False,'dte_shape':None,'dtr_shape':None,'overlap_fraction':.25}},
        'evaluation':{'allow_empty_truth':False},
        'provenance':{'purpose':'adapted','dataset_version':'unspecified','geometry_calibrated':False,
                      'author_data':False,'notes':[]},
    }
    c=_merge(defaults,raw)
    if c['schema_version']!=2: raise ValueError('schema_version must be 2')
    if type(c['seed']) is not int or c['seed']<0: raise ValueError('seed must be a nonnegative integer')
    g=c['geometry']; n=c['network']; p=c['preprocessing']
    for key in ('sdd_mm','sid_mm','pixel_size_mm'): _number(g[key],key)
    if g['sdd_mm']<=g['sid_mm']: raise ValueError('sdd_mm must exceed sid_mm')
    if g['calibration_id'] is not None and (not isinstance(g['calibration_id'],str) or not g['calibration_id']):raise ValueError('calibration_id must be a nonempty string or null')
    _integers(g['detector_shape'],2,'detector_shape'); _integers(g['grid']['shape'],3,'grid.shape')
    for name in ('spacing_mm','origin_mm'):
        v=g['grid'][name]
        if not isinstance(v,list) or len(v)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v):
            raise ValueError(f'grid.{name} must be three finite numbers')
    if any(x<=0 for x in g['grid']['spacing_mm']): raise ValueError('grid spacing must be positive')
    if g['grid']['direction']!=[[1,0,0],[0,1,0],[0,0,1]]: raise ValueError('rotated grid direction unsupported; resample explicitly')
    if c['profile'] not in ANGLE_PATTERNS or not c['profile'].startswith(method+'-'):raise ValueError('profile does not match method')
    validate_profile_angles(c['profile'],g['angles_deg'])
    if n['slice_channels'] not in (1,3,5): raise ValueError('slice_channels must be 1,3,5')
    if n['input_channels']!=(1 if 'one-channel' in c['profile'] else 3): raise ValueError('input channel count conflicts with profile')
    if n['architecture'] not in (('legacy','four-downsample','compact-g0') if a else ('paper','compact-g0')): raise ValueError('unsupported architecture')
    if c['provenance']['purpose'] not in ('g0','adapted','reference'): raise ValueError('invalid purpose')
    if n['architecture']=='compact-g0' and c['provenance']['purpose']!='g0': raise ValueError('compact architecture requires g0 purpose')
    _number(n['bn_eps'],'bn_eps'); _number(n['bn_momentum'],'bn_momentum')
    if n['bn_momentum']>1: raise ValueError('bn_momentum must be at most 1')
    if type(n['conv_bias']) is not bool or n['initialization']!='he_normal': raise ValueError('unsupported convolution semantics')
    if n['dte_activation']!='relu' or n['dtr_activation']!='sigmoid': raise ValueError('unsupported output activation')
    if type(n['compact_base']) is not int or n['compact_base']<2: raise ValueError('invalid compact_base')
    _number(n['dice_smoothing'],'dice_smoothing')
    if a and n['dice_smoothing']!=1: raise ValueError('A Dice smoothing must be 1')
    for stage,t in c['training'].items():
        if t['precision'] not in ('float32','amp_float16','amp_bfloat16'):raise ValueError('invalid training precision')
        if type(t['activation_checkpointing']) is not bool:raise ValueError('activation_checkpointing must be boolean')
        for key in ('epochs','train_pairs','validation_pairs'):
            if t[key] is not None and (type(t[key]) is not int or t[key]<1): raise ValueError(f'{stage}.{key} must be a positive integer')
        if type(t['batch_size']) is not int or t['batch_size']<1: raise ValueError('invalid batch_size')
        _number(t['learning_rate'],'learning_rate')
        _integers(t['patch_shape'],3 if not a and stage=='dtr' else 2,'patch_shape')
        expected_norm='running' if a and stage=='dte' else ('volume' if a else 'none')
        if t['normalization']!=expected_norm: raise ValueError(f'{stage} normalization must be {expected_norm}')
        if t['selection_metric']!=('mae' if stage=='dte' else 'dice'): raise ValueError('unsupported selection_metric')
    for key in ('attenuation_per_mm','integration_step_mm'): _number(p[key],key)
    for key in ('blur_sigma_px','min_background','false_positive_scale'): _number(p[key],key,strict=False)
    for key in ('false_negative_rate','false_positive_rate'):
        _number(p[key],key,strict=False)
        if p[key]>1: raise ValueError(f'{key} must be <=1')
    for key in ('photons_range','scatter_range'):
        if not isinstance(p[key],list) or len(p[key])!=2: raise ValueError(f'{key} requires two bounds')
        for v in p[key]: _number(v,key,strict=key=='photons_range')
        if p[key][0]>p[key][1]: raise ValueError(f'invalid {key} bounds')
    if a and any(p['scatter_range']): raise ValueError('A does not simulate scatter')
    if p['error_model'] not in ('independent_pixel','correlated_spatial'):raise ValueError('unknown projection error_model')
    _number(p['correlation_sigma_px'],'correlation_sigma_px')
    if p['spectral'] is not None:
        from stent_repro.spectral import spectral_project
        if not isinstance(p['spectral'],dict) or set(p['spectral'].get('materials',{}))!={'tool'}:raise ValueError('volume simulator requires one spectral tool material')
        spectral_project({'tool':[0.]},**p['spectral'])
    sim=c['simulation']
    if sim['angular_offset_mode'] not in ('fixed','random_uniform'):raise ValueError('invalid angular offset mode')
    if sim['angular_offset_mode']=='random_uniform' and ('adapted' in c['profile'] or not a):raise ValueError('random starting angles currently apply to A reference profiles')
    bounds=sim['start_angle_range_deg']
    if not isinstance(bounds,list) or len(bounds)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in bounds) or not 0<=bounds[0]<bounds[1]<=360:raise ValueError('invalid start angle range')
    if sim['dte_views'] not in ('same','a_17x10') or (sim['dte_views']=='a_17x10' and not a):raise ValueError('invalid DTE projection protocol')
    density=sim['material_density_range_g_cm3']
    if density is not None:
        if p['spectral'] is None or not isinstance(density,list) or len(density)!=2:raise ValueError('density range requires a spectrum and two bounds')
        for value in density:_number(value,'density range')
        if density[0]>density[1]:raise ValueError('invalid density range')
    tiling=c['inference']['tiling']
    if type(tiling['enabled']) is not bool:raise ValueError('tiling enabled must be boolean')
    for stage in ('dte','dtr'):
        if tiling[stage+'_shape'] is not None:_integers(tiling[stage+'_shape'],3 if not a and stage=='dtr' else 2,'tile shape')
    _number(tiling['overlap_fraction'],'overlap_fraction',strict=False)
    if tiling['overlap_fraction']>=1:raise ValueError('overlap must be below one')
    if p['dtr_crop_shape'] is not None: _integers(p['dtr_crop_shape'],3,'dtr_crop_shape')
    _number(c['inference']['threshold'],'threshold',strict=False)
    if c['inference']['threshold']>1: raise ValueError('threshold must be <=1')
    for key in ('slice_batch','max_volume_voxels'):
        if type(c['inference'][key]) is not int or c['inference'][key]<1: raise ValueError(f'invalid {key}')
    for group,key in [('inference','retain_intermediates'),('evaluation','allow_empty_truth'),('provenance','author_data'),('provenance','geometry_calibrated')]:
        if type(c[group][key]) is not bool: raise ValueError(f'{group}.{key} must be boolean')
    if not isinstance(c['provenance']['notes'],list) or any(not isinstance(x,str) for x in c['provenance']['notes']): raise ValueError('notes must be strings')
    if not isinstance(c['provenance']['dataset_version'],str) or not c['provenance']['dataset_version']: raise ValueError('dataset_version required')
    return c


def validate_projection_metadata(metadata,config,*,angles=None,require_calibration=False):
    """Compare declared acquisition geometry, not the reconstruction grid."""
    import numpy as np
    g=config['geometry'];expected_angles=g['angles_deg'] if angles is None else angles
    if metadata.get('angles_deg')!=expected_angles:raise ValueError('projection angles/view order mismatch')
    geometry=metadata.get('geometry',{})
    if not isinstance(geometry,dict):raise ValueError('projection geometry metadata must be a mapping')
    for source in (metadata,geometry):
        for key in ('sdd_mm','sid_mm','pixel_size_mm','detector_shape'):
            if key in source:
                try:equal=np.shape(source[key])==np.shape(g[key]) and np.allclose(source[key],g[key],atol=1e-7,rtol=1e-7)
                except (TypeError,ValueError):equal=False
                if not equal:raise ValueError(f'projection geometry conflict: {key}')
        if 'angles_deg' in source and source['angles_deg']!=expected_angles:raise ValueError('projection geometry angles conflict')
        if 'calibration_id' in source and source['calibration_id'] is not None and source['calibration_id']!=g['calibration_id']:
            raise ValueError('projection calibration_id conflict')
    if 'pixel_spacing_mm' in metadata:
        if np.shape(metadata['pixel_spacing_mm'])!=(2,) or not np.allclose(metadata['pixel_spacing_mm'],[g['pixel_size_mm']]*2):raise ValueError('projection pixel spacing conflict')
    if require_calibration:
        identity=metadata.get('calibration_id',geometry.get('calibration_id'))
        if not g['calibration_id'] or identity!=g['calibration_id']:raise ValueError('formal projection requires matching calibration_id')
