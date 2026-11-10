"""Versioned, fully resolved experiment contracts. Unknown keys are errors."""
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

SCHEMA = 'fdk_unet_experiment_v3'
SPEC = 'fdk_unet_swin_fusion_spec_v3'
CORE = ('A', 'B', 'H00', 'H10', 'H01', 'C')


@dataclass(frozen=True)
class ExperimentConfig:
    resolved: dict

    @property
    def model_id(self): return self.resolved['model_id']
    @property
    def experiment_id(self): return self.resolved['experiment_id']
    @property
    def stage(self): return self.resolved['stage']
    @property
    def seed(self): return self.resolved['seed']
    @property
    def patch_size(self): return tuple(self.resolved['patch_size'])
    @property
    def batch_size(self): return self.resolved['batch_size']


def _defaults(name, seed=2026):
    fusion = name != 'A'
    bottle = name in ('B', 'H10', 'C')
    levels = [2, 3, 4] if bottle else [2, 3]
    return {
        'schema_version': SCHEMA, 'spec_version': SPEC, 'experiment_id': name,
        'core_experiment_id': name, 'run_kind': 'formal', 'seed': seed,
        'model_id': 'swin_prior_fdk_unet3d_v2' if fusion else 'plain_fdk_unet3d_v1',
        'stage': 'frozen_fusion' if fusion else 'baseline', 'input_adapter': False,
        'unet': {'channels': [16,32,64,128,256], 'norm': 'batch', 'out_channels': 1},
        'prior': {
            'init': 'random' if name == 'B' else 'ssl', 'implementation_contract': 'monai-1.3.0',
            'feature_size': 48, 'depths': [2,2,2,2], 'num_heads': [3,6,12,24],
            'window_size': [7,7,7], 'downsample': 'merging', 'use_v2': False,
            'patch_norm': False, 'normalize_features': True,
            'execute_through': 'layers3' if bottle else 'layers2', 'frozen': True,
            'artifact': None, 'use_checkpoint': False,
        } if fusion else None,
        'fusion': {
            'unet_indices': levels, 'swin_indices': [l-1 for l in levels],
            'swin_channels': [96,192,384][:len(levels)],
            'unet_channels': [64,128,256][:len(levels)],
            'gate_hidden_channels': [16,32,64][:len(levels)], 'skip_indices': [2,3],
            'bottleneck_enabled': bottle, 'decoder_conditioning': name in ('B','H01','C'),
            'skip_gate_input_multiplier': 3, 'bottleneck_gate_input_multiplier': 2,
            'feature_adapter': 'pointwise', 'gate_mode': 'learned', 'alpha_init': 0.0,
            'inject_into': 'bottleneck_and_decoder_skips' if bottle else 'decoder_skips_only',
        } if fusion else None,
        'swin_input': {'mode': 'train_percentile', 'percentiles': [0.5,99.5], 'samples_per_group': 65536},
        'patch_size': [128,128,128], 'batch_size': 2,
        'optimizer': {'name': 'Adam', 'lr': 1e-3, 'betas': [0.9,0.999], 'eps': 1e-8, 'weight_decay': 0.0},
        'loss': 'foreground_dice_bce', 'max_optimizer_steps': 10000, 'validate_every': 250,
        'foreground_probability': 0.33, 'augment': True, 'threshold': 0.5,
        'num_workers': 0, 'amp': 'cuda_float16', 'scheduler': None, 'tta': False,
        'postprocessing': None, 'checkpoint_selection': 'validation_group_dice',
        'inference_stride': [64,64,64],
    }


def _check_structure(actual, expected, path='config'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'{path}: missing or unknown fields')
        for key in expected:
            _check_structure(actual[key], expected[key], f'{path}.{key}')


def resolve_config(raw: dict) -> ExperimentConfig:
    r = copy.deepcopy(raw)
    if r.get('experiment_id') == 'H11':
        r['experiment_id'] = 'C'
    if r.get('core_experiment_id') == 'H11':
        r['core_experiment_id'] = 'C'
    name = r.get('core_experiment_id')
    if name not in CORE:
        raise ValueError('Unknown core_experiment_id')
    expected = _defaults(name)
    _check_structure(r, expected)
    for key in ('schema_version','spec_version','model_id','stage','unet','fusion','input_adapter',
                'loss','num_workers','amp','scheduler','tta','postprocessing','checkpoint_selection', 'swin_input'):
        if r[key] != expected[key]:
            raise ValueError(f'Unsupported or contradictory {key}')
    if r['prior'] is not None:
        for key, value in expected['prior'].items():
            if key not in ('artifact', 'use_checkpoint') and r['prior'][key] != value:
                raise ValueError(f'Contradictory prior.{key}')
        if r['prior']['artifact'] is not None and not isinstance(r['prior']['artifact'], str):
            raise ValueError('prior.artifact must be a path or null')
        if type(r['prior']['use_checkpoint']) is not bool:
            raise ValueError('use_checkpoint must be boolean')
    if r['run_kind'] not in ('formal','smoke'):
        raise ValueError('run_kind must be formal or smoke')
    if not isinstance(r['experiment_id'], str) or not r['experiment_id']:
        raise ValueError('experiment_id required')
    if r['experiment_id'] in CORE and r['experiment_id']!=name:
        raise ValueError('Reserved experiment_id must match core_experiment_id')
    if r['run_kind'] == 'smoke' and (r['experiment_id'] in CORE or r['max_optimizer_steps'] == 10000):
        raise ValueError('Smoke requires a distinct experiment_id and explicit budget')
    for key in ('seed','batch_size','max_optimizer_steps','validate_every'):
        if type(r[key]) is not int or r[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid {key}')
    patch, stride = r['patch_size'], r['inference_stride']
    if not isinstance(patch, list) or len(patch) != 3 or any(type(s) is not int or s < 32 for s in patch):
        raise ValueError('Invalid patch_size')
    if name != 'A' and any(s % 32 for s in patch):
        raise ValueError('Fusion patch edges must be multiples of 32')
    if not isinstance(stride, list) or len(stride) != 3 or any(type(s) is not int or s < 1 for s in stride):
        raise ValueError('Invalid inference_stride')
    # Stride is clamped to each patch by the inference caller for explicit small smoke patches.
    for key in ('foreground_probability','threshold'):
        if not isinstance(r[key], (int,float)) or not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
            raise ValueError(f'Invalid {key}')
    if type(r['augment']) is not bool or r['optimizer'] != expected['optimizer']:
        raise ValueError('Unsupported augmentation or optimizer')
    if r['run_kind'] == 'formal' and r['experiment_id'] in CORE:
        for key in ('patch_size','batch_size','max_optimizer_steps','validate_every','foreground_probability',
                    'augment','threshold','inference_stride'):
            if r[key] != expected[key]:
                raise ValueError(f'Core formal {key} is fixed; use a new experiment_id')
    return ExperimentConfig(r)


def get_experiment_config(experiment_id: str, *, seed: int=2026) -> ExperimentConfig:
    name = 'C' if experiment_id == 'H11' else experiment_id
    if name not in CORE:
        raise ValueError(f'Unknown experiment {name}')
    return resolve_config(_defaults(name, seed))


def load_config(path: Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    if raw.get('prior') and raw['prior'].get('artifact'):
        raw['prior']['artifact'] = str((path.parent / raw['prior']['artifact']).resolve())
    return resolve_config(raw)
