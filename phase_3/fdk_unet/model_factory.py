"""One reconstruction path; full state restoration never opens prior files."""
import json
from pathlib import Path
from .config import resolve_config
from .initialization import named_rng,validate_initialization
from .plain_unet import PlainUNet3D


def build_model(config,*,initialization=None,model_state=None):
    config=resolve_config(config.resolved)
    if initialization is not None and model_state is not None:raise ValueError('Use initialization or complete model_state, not both')
    if initialization is not None:validate_initialization(initialization,config.seed)
    with named_rng(config.seed,'unet'):unet=PlainUNet3D()
    if initialization is not None:unet.load_state_dict(initialization['unet'],strict=True)
    if config.resolved['prior'] is None:model=unet
    else:
        from .swin_prior import SwinPriorEncoder
        from .hybrid_model import SwinPriorUNet3D
        p=config.resolved['prior']
        with named_rng(config.seed,'prior'):
            prior=SwinPriorEncoder(execute_through=p['execute_through'],use_checkpoint=p['use_checkpoint'])
        model=SwinPriorUNet3D(unet,prior,fusion_config=config.resolved['fusion'],seed=config.seed)
        if initialization is not None:
            state=model.state_dict()
            for key in state:
                if key.startswith(('skips.','bottleneck.')):state[key]=initialization['fusion'][key]
            model.load_state_dict(state,strict=True)
        model.prior_load_report=None
        if model_state is None and p['init']=='ssl':
            from .prior_weights import load_ssl_weights
            if not p['artifact']:raise ValueError('New SSL run requires an explicit prior artifact')
            path=Path(p['artifact']);artifact=json.loads(path.read_text(encoding='utf8'))
            if artifact.get('schema_version')!='fdk_prior_artifact_v1' or artifact.get('init')!='ssl' or not artifact.get('sha256'):
                raise ValueError('Invalid SSL artifact')
            weights=path.parent/artifact['weights']
            model.prior_load_report=load_ssl_weights(prior,weights,expected_sha256=artifact['sha256']).to_dict()
    if model_state is not None:
        try:model.load_state_dict(model_state,strict=True)
        except RuntimeError as exc:raise ValueError('Checkpoint tensors do not match model contract') from exc
    return model
