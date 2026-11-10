"""Explicit weight acquisition and transactional, audited SSL loading."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import torch

OFFICIAL_URL = 'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class LoadReport:
    sha256: str
    key_mapping: list
    loaded: list
    missing: list
    unexpected: list
    discarded_heads: list
    buffer_whitelist: list
    loaded_parameter_fraction: float
    mapping_version: str = 'monai_ssl_swin_v1'

    def to_dict(self): return asdict(self)


def obtain_prior_weights(*, weights, download_official, output):
    if (weights is not None) == bool(download_official):
        raise ValueError('SSL requires exactly one of weights or explicit download_official')
    if weights is not None:
        path = Path(weights).resolve()
        if not path.is_file(): raise ValueError(f'Weights not found: {path}')
        return path
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output/'model_swinvit.pt'
    if path.exists(): raise FileExistsError(path)
    temporary = output/'model_swinvit.pt.download'
    try:
        urllib.request.urlretrieve(OFFICIAL_URL,temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    (output/'download.json').write_text(json.dumps({
        'url': OFFICIAL_URL, 'downloaded_at': datetime.now(timezone.utc).isoformat(),
        'bytes': path.stat().st_size, 'sha256': file_sha256(path)},indent=2),encoding='utf8')
    return path


def _map_key(key):
    if key.startswith('module.'): key = key[len('module.'):]
    for prefix in ('swin_vit.', 'swinViT.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    # The selected SSL artifact predates MONAI's MLP linear layer naming.
    key = re.sub(r'^(layers[1-4]\.0\.blocks\.\d+\.mlp)\.fc1\.', r'\1.linear1.', key)
    key = re.sub(r'^(layers[1-4]\.0\.blocks\.\d+\.mlp)\.fc2\.', r'\1.linear2.', key)
    return key


def load_ssl_weights(encoder, path, *, expected_sha256=None):
    path = Path(path)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('SSL SHA-256 mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    source = payload.get('state_dict', payload)
    if not isinstance(source, dict): raise ValueError('Expected a state_dict mapping')
    target = encoder.swinViT.state_dict()
    parameters = dict(encoder.swinViT.named_parameters())
    mapped, mapping, discarded, unexpected = {}, [], [], []
    heads = ('rotation_head.', 'contrastive_head.', 'conv.', 'rotation_pre.', 'contrastive_pre.')
    for original, value in source.items():
        key = _map_key(original)
        if key.startswith(heads) or key in ('norm.weight','norm.bias','convTrans3d.weight','convTrans3d.bias'):
            discarded.append(original)
            continue
        if key in mapped: raise ValueError(f'SSL mapping collision: {key}')
        if key not in target:
            unexpected.append(original)
            continue
        if not isinstance(value, torch.Tensor) or value.shape != target[key].shape:
            raise ValueError(f'SSL shape mismatch: {original}')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite SSL tensor: {original}')
        mapped[key] = value
        mapping.append({'source': original, 'target': key, 'shape': list(value.shape)})
    whitelist = [k for k in target if k.endswith('.relative_position_index') and k not in parameters]
    missing = sorted(set(target)-set(mapped)-set(whitelist))
    if missing or unexpected:
        raise ValueError(f'Incomplete SSL match: missing={missing}, unexpected={unexpected}')
    for key in whitelist:
        if key in mapped and not torch.equal(mapped[key], target[key].cpu()):
            raise ValueError(f'Deterministic buffer differs: {key}')
        mapped[key] = target[key].cpu().clone()
    # Nothing is copied until every tensor, key and buffer has passed validation.
    encoder.swinViT.load_state_dict(mapped, strict=True)
    return LoadReport(digest, mapping, sorted(mapped), [], [], sorted(discarded), whitelist, 1.0)


def prepare_prior(manifest,split,*,init='ssl',weights=None,download_official=False,output,seed=2026):
    from .prior_stats import fit_prior_stats
    if init not in ('ssl','random'):raise ValueError('Unknown prior initialization')
    if init=='random' and (weights is not None or download_official):raise ValueError('Random prior must not read SSL weights')
    if init=='ssl' and (weights is not None)==bool(download_official):
        raise ValueError('SSL requires explicit weights or download_official, exactly one')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    stats=fit_prior_stats(manifest,split,seed=seed)
    output.mkdir(parents=True,exist_ok=False)
    path=report=digest=None
    if init=='ssl':
        from .swin_prior import SwinPriorEncoder
        path=obtain_prior_weights(weights=weights,download_official=download_official,output=output)
        report=load_ssl_weights(SwinPriorEncoder(execute_through='layers3'),path)
        digest=report.sha256
        (output/'prior_load_report.json').write_text(json.dumps(report.to_dict(),indent=2),encoding='utf8')
    artifact={'schema_version':'fdk_prior_artifact_v1','init':init,'weights':str(path) if path else None,
              'sha256':digest,'load_report':report.to_dict() if report else None,'stats':stats.to_dict(),
              'train_group_digest':stats.train_group_digest,'source_url':OFFICIAL_URL if download_official else None,
              'prepared_at':datetime.now(timezone.utc).isoformat(),'weight_bytes':path.stat().st_size if path else None}
    result=output/'prior.json';result.write_text(json.dumps(artifact,indent=2),encoding='utf8');return result
