"""Content fingerprints include array metadata and directory membership."""
from pathlib import Path
import hashlib
import json
import platform
import numpy
import scipy
import torch


def sha256(path):
    path=Path(path)
    if path.is_dir():
        rows=[(str(p.relative_to(path)),sha256(p)) for p in sorted(path.rglob('*')) if p.is_file()]
        return hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest()
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths):
    result={}
    for p in paths:
        path=Path(p).resolve();result[str(path)]=sha256(path)
        if path.suffix.lower() in ('.npy','.tif','.tiff'):
            sidecar=Path(str(path)+'.json');result[str(sidecar)]=sha256(sidecar) if sidecar.exists() else None
    return dict(sorted(result.items()))


def verify_fingerprints(expected):
    for name,digest in expected.items():
        path=Path(name)
        actual=sha256(path) if path.exists() else None
        if actual!=digest:raise ValueError(f'referenced data or metadata changed: {name}')


def manifest_fingerprints(path):
    path=Path(path).resolve();doc=json.loads(path.read_text(encoding='utf-8'));paths=[path]
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('path','input','target','truth','background','head_background','tool','projections','ideal','dte_ideal'):
                    for p in item if isinstance(item,list) else [item]:
                        if isinstance(p,str):paths.append(path.parent/p)
                elif isinstance(item,(dict,list)):visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(doc);return fingerprint_files(paths)


def freeze_or_verify_manifest(path,expected=None,required=False):
    if expected is None and required:raise ValueError('formal inference requires a frozen input fingerprint contract')
    if isinstance(expected,(str,Path)):expected=json.loads(Path(expected).read_text(encoding='utf8'))
    if expected is not None:verify_fingerprints(expected)
    actual=manifest_fingerprints(path)
    if expected is not None and actual!=expected:raise ValueError('frozen input manifest membership or content changed')
    return actual


def environment_record():
    source=Path(__file__).parent
    return {'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,
            'torch':str(torch.__version__),'cuda_available':torch.cuda.is_available(),
            'code_sha256':{str(p.relative_to(source)):sha256(p) for p in sorted(source.rglob('*.py'))}}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
