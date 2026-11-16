"""Read/write numerical projections and reconstruction volumes without rescaling.

Projection arrays are [view, v, u], volumes are [z, y, x]. Spatial metadata
uses x,y,z coordinates; its absence never implies a default physical grid.
"""
from pathlib import Path
import json
from zipfile import BadZipFile
import numpy as np

_SCHEMA = 'stent-repro-array-v1'

def _validate(array, metadata, kind):
    array = np.asarray(array)
    if array.dtype.kind not in 'buif' or not np.isfinite(array).all():
        raise ValueError('array must contain finite real numerical values')
    if kind == 'projections':
        if array.ndim == 2: array = array[None, ...]
        if array.ndim != 3: raise ValueError('projections must have shape [view, v, u] or [v, u]')
    elif array.ndim != 3:
        raise ValueError('volume must have shape [z, y, x]')
    if any(n == 0 for n in array.shape): raise ValueError('empty dimensions are not supported')
    if not isinstance(metadata, dict): raise ValueError('metadata must be a dictionary')
    try:
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ValueError('metadata must be finite JSON-compatible data') from e
    if kind == 'projections' and 'angles_deg' in metadata:
        a = np.asarray(metadata['angles_deg'])
        if a.shape != (array.shape[0],) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
            raise ValueError('angles_deg must contain one finite angle per view')
    for key, length in (('pixel_spacing_mm',2), ('spacing_mm',3)):
        if key in metadata:
            s = np.asarray(metadata[key])
            if s.shape != (length,) or s.dtype.kind not in 'iuf' or not np.isfinite(s).all() or (s <= 0).any():
                raise ValueError(f'{key} must contain {length} positive finite values')
    if 'origin_mm' in metadata:
        origin = np.asarray(metadata['origin_mm'])
        if origin.shape != (3,) or origin.dtype.kind not in 'iuf' or not np.isfinite(origin).all(): raise ValueError('origin_mm must contain three finite values')
    if 'direction' in metadata:
        direction = np.asarray(metadata['direction'])
        if direction.shape != (3,3) or direction.dtype.kind not in 'iuf' or not np.isfinite(direction).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            raise ValueError('direction must be a finite orthonormal 3x3 matrix')
    return array, metadata

def _save(path, array, metadata, kind):
    path = Path(path)
    if path.suffix.lower() not in ('.npy','.npz','.tif','.tiff'): raise ValueError('supported formats: .npy, .npz, .tif, .tiff')
    array, metadata = _validate(array, metadata, kind)
    envelope = {'schema':_SCHEMA, 'kind':kind, 'metadata':metadata}
    encoded = json.dumps(envelope, ensure_ascii=False, allow_nan=False)
    if path.suffix.lower() == '.npz':
        with path.open('wb') as f: np.savez_compressed(f, data=array, metadata_json=np.array(encoded))
    else:
        if path.suffix.lower() == '.npy':
            with path.open('wb') as f: np.save(f, array, allow_pickle=False)
        else:
            import tifffile
            tifffile.imwrite(path, array, photometric='minisblack', metadata=None)
        Path(str(path)+'.json').write_text(encoded, encoding='utf-8')

def _load(path, kind):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'data','metadata_json'}: raise ValueError('unsupported NPZ archive: expected data and metadata_json')
            array = archive['data']
            envelope = json.loads(str(archive['metadata_json'].item()))
    elif suffix in ('.npy','.tif','.tiff'):
        if suffix == '.npy':
            with path.open('rb') as f: array = np.load(f, allow_pickle=False)
        else:
            array = _read_tiff(path)
        sidecar = Path(str(path)+'.json')
        envelope = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else None
    else:
        raise ValueError('supported formats: .npy, .npz, .tif, .tiff')
    if envelope is not None:
        if not isinstance(envelope, dict) or envelope.get('schema') != _SCHEMA or envelope.get('kind') != kind:
            raise ValueError('unsupported metadata schema or wrong data kind')
        metadata = envelope.get('metadata')
    else: metadata = {}
    if kind == 'volume' and suffix in ('.tif', '.tiff') and array.ndim == 2:
        array = array[None, ...]
    return _validate(array, metadata, kind)

def save_projections(path, projections, metadata=None):
    _save(path, projections, {} if metadata is None else metadata, 'projections')

def load_projections(path):
    try:
        if Path(path).is_dir(): return load_tiff_sequence(path)
        return _load(path, 'projections')
    except (EOFError,BadZipFile,UnicodeError) as exc:
        raise ValueError(f'corrupt projection file: {path}: {exc}') from exc

def save_volume(path, volume, metadata=None):
    _save(path, volume, {} if metadata is None else metadata, 'volume')

def load_volume(path):
    try:return _load(path, 'volume')
    except (EOFError,BadZipFile,UnicodeError) as exc:
        raise ValueError(f'corrupt volume file: {path}: {exc}') from exc

def _validate_cases(cases):
    if not isinstance(cases, list): raise ValueError('cases must be a list')
    ids = set()
    model_splits = {}
    for row in cases:
        if not isinstance(row, dict): raise ValueError('case must be an object')
        for field in ('id','projections','volume','split'):
            if not isinstance(row.get(field), str) or not row[field]: raise ValueError(f'case {field} must be a nonempty string')
        if row['id'] in ids: raise ValueError('duplicate case id')
        ids.add(row['id'])
        if row['split'] not in ('train','validation','test'): raise ValueError('invalid split')
        if 'model_ids' in row and (not isinstance(row['model_ids'],list) or not all(isinstance(x,str) and x for x in row['model_ids'])):
            raise ValueError('model_ids must be a list of nonempty strings')
        for model in row.get('model_ids', []):
            if model in model_splits and model_splits[model] != row['split']:
                raise ValueError(f'model {model!r} appears in multiple splits')
            model_splits[model] = row['split']
    return cases

def save_manifest(path, cases):
    """Write JSON cases; relative paths are relative to the manifest directory."""
    _validate_cases(cases)
    Path(path).write_text(json.dumps({'cases':cases}, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def load_manifest(path):
    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(doc, dict): raise ValueError('manifest must be a JSON object')
    cases = _validate_cases(doc.get('cases'))
    return [{**row, 'projections': (path.parent / row['projections']).resolve(), 'volume': (path.parent / row['volume']).resolve()} for row in cases]

def load_case(case):
    projections, projection_metadata = load_projections(case['projections'])
    volume, volume_metadata = load_volume(case['volume'])
    return {**case, 'projections':projections, 'volume':volume, 'projection_metadata':projection_metadata, 'volume_metadata':volume_metadata}

def _read_tiff(path):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        if any(page.samplesperpixel != 1 or page.photometric.name not in ('MINISBLACK','MINISWHITE') for page in tif.pages):
            raise ValueError('TIFF must contain grayscale images')
        images = [page.asarray() for page in tif.pages]
        if not images or any(image.ndim != 2 for image in images):
            raise ValueError('TIFF must contain 2D grayscale pages')
        if any(image.shape != images[0].shape or image.dtype != images[0].dtype for image in images):
            raise ValueError('TIFF pages must have identical shape and dtype')
        return images[0] if len(images) == 1 else np.stack(images)

def save_tiff_sequence(directory, projections, metadata=None):
    """Write one image per view and a sequence manifest; directory must be empty."""
    import tifffile
    array, metadata = _validate(projections, {} if metadata is None else metadata, 'projections')
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError('sequence directory must be empty')
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for index, image in enumerate(array):
        name = f'view_{index:05d}.tif'
        tifffile.imwrite(directory / name, image, photometric='minisblack', metadata=None)
        files.append(name)
    doc = {'schema':_SCHEMA, 'kind':'projection_sequence', 'files':files, 'metadata':metadata}
    (directory/'sequence.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load_tiff_sequence(paths, metadata=None):
    """Load explicitly ordered paths, or a directory containing sequence.json."""
    if isinstance(paths, (str,Path)):
        directory = Path(paths)
        doc = json.loads((directory/'sequence.json').read_text(encoding='utf-8'))
        if doc.get('schema') != _SCHEMA or doc.get('kind') != 'projection_sequence': raise ValueError('invalid sequence manifest')
        files = doc.get('files')
        if not isinstance(files,list) or not files or any(not isinstance(x,str) or Path(x).name != x or Path(x).suffix.lower() not in ('.tif','.tiff') for x in files):
            raise ValueError('invalid sequence filenames')
        if metadata is not None: raise ValueError('metadata already supplied by sequence manifest')
        metadata = doc.get('metadata')
        paths = [directory/name for name in files]
    paths = list(paths)
    if not paths: raise ValueError('TIFF sequence must not be empty')
    images = [_read_tiff(path) for path in paths]
    if any(x.ndim != 2 for x in images): raise ValueError('each TIFF sequence file must be one grayscale image')
    if any(x.shape != images[0].shape or x.dtype != images[0].dtype for x in images): raise ValueError('TIFF sequence images must have identical shape and dtype')
    return _validate(np.stack(images), {} if metadata is None else metadata, 'projections')
