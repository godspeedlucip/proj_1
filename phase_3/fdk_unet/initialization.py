"""Independent, stable RNG streams; initialization never advances data RNGs."""
from contextlib import contextmanager
import hashlib
import random
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def seed_for(root_seed: int, scope: str) -> int:
    return int.from_bytes(hashlib.sha256(f"fdk-v3:{root_seed}:{scope}".encode()).digest()[:8], 'big') % (2**32)


def capture_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG state requires CUDA for exact training resume')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@contextmanager
def named_rng(root_seed: int, scope: str):
    state = capture_rng_state()
    seed = seed_for(root_seed, scope)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)


def tensor_digest(states):
    digest=hashlib.sha256()
    for namespace,state in sorted(states.items()):
        for key,value in sorted(state.items()):
            tensor=value.detach().cpu().contiguous()
            digest.update(f'{namespace}:{key}:{tensor.dtype}:{list(tensor.shape)}'.encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_initialization_bundle(seed):
    from .plain_unet import PlainUNet3D
    from .fusion import BottleneckFusion,DecoderGuidedFusion
    with named_rng(seed,'unet'):unet=PlainUNet3D().state_dict()
    fusion={}
    for level,cs,cu in ((2,96,64),(3,192,128),(4,384,256)):
        with named_rng(seed,f'fusion.scale{level}'):
            module=BottleneckFusion(cs,cu) if level==4 else DecoderGuidedFusion(cs,cu,decoder_conditioning=True)
        prefix='bottleneck' if level==4 else f'skips.{level}'
        fusion.update({f'{prefix}.{key}':value for key,value in module.state_dict().items()})
    return {'schema_version':'fdk_initialization_v1','seed':seed,'unet':unet,'fusion':fusion,
            'tensor_digest':tensor_digest({'unet':unet,'fusion':fusion})}


def validate_initialization(bundle,seed):
    if bundle.get('schema_version')!='fdk_initialization_v1' or bundle.get('seed')!=seed:
        raise ValueError('Initialization version or seed differs')
    if bundle.get('tensor_digest')!=tensor_digest({'unet':bundle['unet'],'fusion':bundle['fusion']}):
        raise ValueError('Initialization tensor digest differs')


def create_initialization_bundle(*,seed,output):
    output=Path(output)
    if output.exists():
        bundle=torch.load(output,map_location='cpu',weights_only=True);validate_initialization(bundle,seed);return bundle
    output.parent.mkdir(parents=True,exist_ok=True)
    bundle=make_initialization_bundle(seed)
    fd,name=tempfile.mkstemp(prefix='.initialization-',suffix='.tmp',dir=output.parent)
    try:
        with os.fdopen(fd,'wb') as stream:torch.save(bundle,stream);stream.flush();os.fsync(stream.fileno())
        try:os.link(name,output)  # atomic create, never overwrite an existing common initialization
        except FileExistsError:
            bundle=torch.load(output,map_location='cpu',weights_only=True);validate_initialization(bundle,seed)
    finally:Path(name).unlink(missing_ok=True)
    return bundle
