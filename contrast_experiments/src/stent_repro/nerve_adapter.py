"""Import registered TopBrain/TIGRE synthetic overlays and original stent truth."""

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
import tifffile

from stent_repro.evaluation import validate_grid
from stent_repro.geometry import circular_view, project_points
from stent_repro.io import save_projections, save_volume
from stent_repro.provenance import sha256, write_json


@dataclass(frozen=True)
class NerveCase:
    head_dir: Path | str
    stent_projection_dir: Path | str
    stack_dir: Path | str
    model_dir: Path | str
    split: str
    legacy_source_split: str | None = None


_NOISELESS = 'image = head + Gaussian(combined implant projection)'
_NOISY = 'image = head + GaussianNoise(0, noise_std) + Gaussian(combined implant projection)'
_POLYFIT = _NOISY.replace('= head', '= Polyfit(head)')
_SOURCE_KIND = 'synthetic_topbrain_overlay'
_FRAME = '(x_repo,y_repo,z_repo)=(y_tigre,-x_tigre,z_tigre), relative to the source voxel-grid center'


def _json(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected a JSON object in {path}')
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('source sample and stent series must be filename-safe identifiers')
    return value


def _variant(value):
    if type(value) not in (int, str) or not str(value).isascii() or not str(value).isdigit():
        raise ValueError('stent variant must be a positive integer identifier')
    number = int(value)
    if number < 1:
        raise ValueError('stent variant must be a positive integer identifier')
    return f'{number:03}'


def _numeric(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'uif' or not np.isfinite(a).all():
        raise ValueError(f'{name} requires finite numerical values of shape {shape}')
    return a.astype(float)


def _geometry(doc):
    sid, sdd = [float(_numeric(doc.get(k), (), k)) for k in ('DSO_mm', 'DSD_mm')]
    pixels = _numeric(doc.get('dDetector_mm'), (2,), 'dDetector_mm')
    shape = doc.get('nDetector')
    if (not 0 < sid < sdd or np.any(pixels <= 0) or not np.isclose(*pixels)
            or not isinstance(shape, list) or len(shape) != 2
            or any(type(x) is not int or x <= 0 for x in shape)):
        raise ValueError('unsupported TIGRE cone-beam acquisition geometry')
    for key, size in [('offOrigin_mm', 3), ('offDetector_mm', 2)]:
        if np.any(_numeric(doc.get(key), (size,), key) != 0):
            raise ValueError('nonzero TIGRE offset is unsupported by circular geometry')
    for key in ('rotDetector', 'rotDetector_rad', 'rotDetector_deg', 'COR', 'COR_mm'):
        if key in doc and np.any(np.asarray(doc[key]) != 0):
            raise ValueError('nonzero TIGRE detector rotation or COR is unsupported')
    return {'sid_mm': sid, 'sdd_mm': sdd, 'pixel_size_mm': float(pixels[0]), 'detector_shape': shape}


def _view_table(docs, angles):
    fields = ('angles_deg', 'view_indices', 'output_mode')
    source_angles, view_indices, mode = (docs[0].get(key) for key in fields)
    if (not isinstance(source_angles, list) or not source_angles
            or not isinstance(view_indices, list) or len(source_angles) != len(view_indices)
            or any(type(v) is not int or v < 0 for v in view_indices)
            or len(set(view_indices)) != len(view_indices) or mode not in ('2D', '3D')):
        raise ValueError('invalid source view order or projection output mode')
    _numeric(source_angles, (len(source_angles),), 'source angles_deg')
    if len(set(source_angles)) != len(source_angles):
        raise ValueError('source angles must be unique')
    if any(tuple(d.get(k) for k in fields) != (source_angles, view_indices, mode) for d in docs[1:]):
        raise ValueError('head, implant and composite view tables differ')
    if any(a not in source_angles for a in angles):
        raise ValueError(f'requested angles must be among available angles {source_angles}')
    return [(a, source_angles.index(a), view_indices[source_angles.index(a)]) for a in angles], mode


def _inspect(case, angles, engineering_holdout):
    dirs = tuple(Path(p).resolve() for p in (case.head_dir, case.stent_projection_dir,
                                           case.stack_dir, case.model_dir))
    head_dir, stent_dir, stack_dir, model_dir = dirs
    paths = [head_dir / 'preparation.json', head_dir / 'projection.json',
             stent_dir / 'projection.json', stack_dir / 'stack.json', model_dir / 'stent.json',
             model_dir / 'marker.json']
    prep, head, stent, stack, model, markers = map(_json, paths)
    sample, family, variant = _name(head.get('sample')), _name(stent.get('stent_series')), _variant(stent.get('stent_variant'))
    if (sample != prep.get('sample') or sample != stack.get('sample') or family != stack.get('stent_series')
            or family != model.get('series') or variant != _variant(stack.get('stent_variant'))
            or variant != _variant(model.get('variant'))):
        raise ValueError('head, stent volume and stack identifiers disagree')

    split_sources = {d['split'] for d in (prep, head, stent, stack, model) if d.get('split') is not None}
    if any(not d.get('split') for d in (prep, head, stent, stack)) and case.legacy_source_split is None:
        raise ValueError('legacy source requires an explicit legacy_source_split declaration')
    if case.legacy_source_split is not None:
        split_sources.add(case.legacy_source_split)
    if isinstance(model.get('mesh'), str):
        path = PureWindowsPath(model['mesh'].replace('/', '\\'))
        if path.stem != family or path.parent.name not in ('train', 'test'):
            raise ValueError('model mesh must identify the matching stent family and source split')
        split_sources.add(path.parent.name)
    if len(split_sources) != 1 or not split_sources.issubset({'train', 'test'}):
        raise ValueError('head and stent source split declarations conflict')
    source_split = split_sources.pop()
    if case.split not in ('train', 'validation', 'test'):
        raise ValueError('invalid destination split')
    if case.split != source_split and not (engineering_holdout and source_split == 'train'):
        raise ValueError('source split reassignment requires an explicit engineering_holdout')

    geometries = [_geometry(d) for d in (head['geometry'], stent['geometry'],
                                         stack['head_geometry'], stack['stent_geometry'])]
    g = geometries[0]
    if any(other['detector_shape'] != g['detector_shape'] or any(
        not np.isclose(g[k], other[k], rtol=1e-6, atol=1e-7)
        for k in ('sid_mm', 'sdd_mm', 'pixel_size_mm')) for other in geometries[1:]):
        raise ValueError('head, implant and composite detector geometries differ')
    digest = hashlib.sha256(json.dumps({'geometry': g, 'frame': _FRAME}, sort_keys=True).encode()).hexdigest()
    g['calibration_id'] = f'nerve-tigre:{digest}'
    views, mode = _view_table((head, stent, stack), angles)
    for k in ('stent_mu', 'marker_mu'):
        value = _numeric(stent.get(k), (), k)
        if value <= 0 or value != _numeric(stack.get(k), (), k):
            raise ValueError(f'invalid or inconsistent {k}')
    if stack.get('formula') not in (_NOISELESS, _NOISY, _POLYFIT):
        raise ValueError('unrecognized synthetic overlay formula')
    return {'id': f'{sample}__{family}__{variant}', 'head': sample, 'family': family, 'variant': variant,
            'source_split': source_split, 'legacy_source_split_declared': case.legacy_source_split,
            'split': case.split, 'dirs': dirs, 'paths': paths,
            'geometry': g, 'views': views, 'mode': mode, 'stent_doc': stent, 'stack_doc': stack,
            'head_doc': head, 'model_doc': model, 'marker_doc': markers}


def _read_truth(record, target_grid):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError('nerve_adapter requires nibabel for original stent NIfTI truth') from exc
    path = record['dirs'][3] / 'stent_real_grid.nii.gz'
    img = nib.load(path)
    model, geo = record['model_doc'], record['stent_doc']['geometry']
    spacing = _numeric(model.get('voxel_spacing_mm'), (3,), 'stent voxel_spacing_mm')
    if (img.ndim != 3 or img.header.get_xyzt_units()[0] not in ('mm', 'unknown')
            or np.any(spacing <= 0) or not np.allclose(img.affine[:3, :3], np.diag(spacing))
            or img.shape != tuple(model.get('volume_shape', []))
            or img.shape[::-1] != tuple(geo.get('nVoxel', []))
            or not np.allclose(spacing[::-1], _numeric(geo.get('dVoxel_mm'), (3,), 'dVoxel_mm'))
            or record['stent_doc'].get('voxel_spacing_multiplier', 1) != 1):
        raise ValueError('source stent NIfTI grid disagrees with projection voxel grid')
    raw = np.asanyarray(img.dataobj)
    count = np.count_nonzero(raw)
    if (raw.dtype.kind not in 'bui' or raw.min() < 0 or raw.max() != 1
            or count != model.get('combined_voxels', count)):
        raise ValueError('source stent_real_grid.nii.gz must be an original nonempty binary integer volume')

    centers = []
    for marker in record['marker_doc'].get('markers', []):
        voxels = np.asarray(marker.get('voxels_ijk'))
        if (voxels.ndim != 2 or voxels.shape[1] != 3 or not len(voxels) or voxels.dtype.kind not in 'ui'
                or np.any(voxels < 0) or np.any(voxels >= img.shape)
                or not np.all(raw[tuple(voxels.T)] == 1)):
            raise ValueError('source marker voxels disagree with original 3D stent truth')
        center = _numeric(marker.get('center_voxel_ijk'), (3,), 'center_voxel_ijk')
        if not np.allclose(center, voxels.mean(axis=0)):
            raise ValueError('marker voxel indices and center disagree')
        centers.append(center)
    if len(centers) != 2:
        raise ValueError('the two original markers are required to check TIGRE registration')
    xyz = (np.asarray(centers) - (np.asarray(img.shape) - 1) / 2) * spacing
    markers = xyz[:, [1, 0, 2]] * [1, -1, 1]

    native = raw.transpose(2, 0, 1)[:, ::-1, :]
    native_spacing = spacing[[1, 0, 2]]
    native_origin = -(np.asarray(native.shape[::-1]) - 1) * native_spacing / 2
    native_grid = {'shape': list(native.shape), 'spacing_mm': native_spacing.tolist(),
                   'origin_mm': native_origin.tolist(), 'direction': np.eye(3).tolist()}
    meta = {'units': 'binary', 'source_kind': 'original_tigre_stent_and_marker_truth',
            'frame': _FRAME, 'source_affine': img.affine.tolist(), 'native_grid': native_grid,
            'resampling': 'none', 'source_foreground_voxels': int(count),
            'output_foreground_voxels': int(count)}
    if target_grid is None:
        return native, {**meta, **native_grid}, markers, path
    spacing_out = np.asarray(target_grid['spacing_mm'])
    origin_out = np.asarray(target_grid['origin_mm'])
    occupied_by_axis = [np.flatnonzero(native.any(axis=other_axes))
                        for other_axes in ((0, 1), (0, 2), (1, 2))]
    occupied_min = np.array([occupied[0] for occupied in occupied_by_axis])
    occupied_max = np.array([occupied[-1] for occupied in occupied_by_axis])
    target_min = origin_out - spacing_out / 2
    target_max = origin_out + (np.asarray(target_grid['shape'][::-1]) - 0.5) * spacing_out
    if (np.any(native_origin + (occupied_min - 0.5) * native_spacing < target_min - 1e-6)
            or np.any(native_origin + (occupied_max + 0.5) * native_spacing > target_max + 1e-6)):
        raise ValueError('fixed target grid would crop original stent truth')
    scale = (spacing_out / native_spacing)[::-1]
    offset = ((origin_out - native_origin) / native_spacing)[::-1]
    result = affine_transform(native, np.diag(scale), offset, output_shape=target_grid['shape'],
                              order=0, mode='grid-constant', cval=0, prefilter=False)
    output_foreground = int(np.count_nonzero(result))
    if not output_foreground:
        raise ValueError('original stent truth vanishes on the fixed target grid')
    return result, {**meta, **target_grid, 'resampling': 'nearest_fixed_grid',
                    'output_foreground_voxels': output_foreground,
                    'target_to_native_index_scale_zyx': scale.tolist(),
                    'target_to_native_index_offset_zyx': offset.tolist()}, markers, path


def _read_frame(directory, kind, view, mode, geometry):
    angle, page, _ = view
    base = directory / 'projections'
    if mode == '2D':
        whole, dot, fraction = f'{angle:.12f}'.rstrip('0').rstrip('.').partition('.')
        name = f'{int(whole):03d}' + (f'.{fraction}' if dot else '')
        path = base / kind / f'{name}.tif'
        image = tifffile.imread(path)
    else:
        path = base / f'{kind}.tif'
        image = tifffile.imread(path, key=page)
    if (image.shape != tuple(geometry['detector_shape']) or image.dtype.kind != 'f'
            or image.dtype.itemsize != 4 or not np.isfinite(image).all()):
        raise ValueError(f'invalid float32 detector frame in {path}')
    return image.astype(np.float32, copy=False), path


def _validate_overlay(head, implant, image, record, view_index):
    doc = record['stack_doc']
    sigma = float(_numeric(doc.get('implant_sigma_px'), (), 'implant_sigma_px'))
    noise = float(_numeric(doc.get('noise_std', 0 if doc['formula'] == _NOISELESS else None), (), 'noise_std'))
    if sigma < 0 or noise < 0 or (doc['formula'] == _NOISELESS and noise != 0):
        raise ValueError('invalid overlay noise or blur parameter')
    if doc['formula'] == _POLYFIT:
        coeff = np.asarray(doc.get('head_polyfit_coefficients_ascending'))
        if coeff.ndim != 1 or not 2 <= len(coeff) <= 4:
            raise ValueError('legacy head remapping requires explicit polynomial coefficients')
        head = np.polynomial.polynomial.polyval(head, _numeric(coeff, coeff.shape, 'polyfit')).astype(np.float32)
    if noise:
        seed = doc.get('noise_seed')
        if type(seed) is not int or seed < 0:
            raise ValueError('noisy composite requires a valid source noise_seed')
        sid = int.from_bytes(hashlib.sha256(record['head'].encode()).digest()[:8], 'little')
        rng = np.random.default_rng(np.random.SeedSequence([seed, sid & 0xffffffff, sid >> 32, view_index]))
        head = head + rng.normal(0, noise, head.shape).astype(np.float32)
    expected = head + (gaussian_filter(implant, sigma) if sigma else implant)
    if not np.allclose(image, expected, rtol=2e-7, atol=2e-6):
        raise ValueError('composite values disagree with the declared head/implant overlay')


def _load_views(record, markers):
    head_dir, stent_dir, stack_dir, _ = record['dirs']
    g = record['geometry']
    files = []
    head_frames, implant_frames, composite_frames = np.empty(
        (3, len(record['views']), *g['detector_shape']), dtype=np.float32)
    marker_errors = []
    for pos, view in enumerate(record['views']):
        angle, _, idx = view
        marker_path = stent_dir / 'marker_metadata' / f'{idx:03}.json'
        doc = _json(marker_path)
        if (doc.get('angle_deg') != angle or doc.get('view_index') != idx
                or doc.get('coordinate_order') != 'row_column' or doc.get('index_base') != 0):
            raise ValueError('TIGRE marker sidecar view index, angle or coordinate convention differs')
        observed = _numeric(doc.get('marker_centers_rc'), (2, 2), 'marker_centers_rc')
        camera = circular_view(angle, g['sdd_mm'], g['sid_mm'], g['detector_shape'], g['pixel_size_mm'])
        predicted_rc = project_points(markers, camera)[:, ::-1]
        distances = np.linalg.norm(predicted_rc[np.argsort(predicted_rc[:, 0])] - observed[np.argsort(observed[:, 0])], axis=1)
        if not np.all(np.isfinite(distances) & (distances <= 2)):
            raise ValueError('original 3D marker projections disagree with TIGRE detector registration')
        marker_errors.extend(distances.tolist())
        h, hp = _read_frame(head_dir, 'head', view, record['mode'], g)
        p, pp = _read_frame(stent_dir, 'implant', view, record['mode'], g)
        x, xp = _read_frame(stack_dir, 'image', view, record['mode'], g)
        if min(h.min(), p.min(), x.min()) < 0 or not p.any():
            raise ValueError('source alignment training requires nonnegative line integrals and a nonempty implant')
        _validate_overlay(h, p, x, record, idx)
        head_frames[pos] = h
        implant_frames[pos] = p
        composite_frames[pos] = x
        files.extend((marker_path, hp, pp, xp))
    return head_frames, implant_frames, composite_frames, max(marker_errors), files


def import_nerve_dataset(cases: Iterable[NerveCase], output: Path | str, *, angles_deg,
                         target_grid: dict | None = None, engineering_holdout: bool = False) -> dict[str, Path]:
    """Import fixed views from read-only sources into raw, inference and truth manifests.

    A source train case can enter a derived holdout only with an explicit
    engineering_holdout request. Head IDs and whole stent families are disjoint
    across destination splits. The output grid, if supplied, applies to all cases.
    """
    angles = list(angles_deg)
    if not angles:
        raise ValueError('at least one ordered source angle is required')
    angles = _numeric(angles, (len(angles),), 'angles_deg').tolist()
    if len({a % 360 for a in angles}) != len(angles):
        raise ValueError('duplicate angles')
    if type(engineering_holdout) is not bool:
        raise ValueError('engineering_holdout must be boolean')
    grid = validate_grid(target_grid) if target_grid is not None else None
    records = [_inspect(case, angles, engineering_holdout) for case in cases]
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('at least one case and unique identities required')
    for key in ('head', 'family'):
        assignments = {}
        for row in records:
            ident, split = row[key], row['split']
            if assignments.setdefault(ident, split) != split:
                raise ValueError(f'{key} {ident!r} appears in multiple splits')
    if any(row['geometry'] != records[0]['geometry'] for row in records):
        raise ValueError('all imported cases must share the same acquisition geometry')

    destination = Path(output).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('nerve adapter output directory must be empty')
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw, inference, truths, evidence = [], [], [], []
    with TemporaryDirectory(prefix='.nerve-import-', dir=destination.parent) as temp:
        stage = Path(temp) / 'dataset'
        stage.mkdir()
        for row in records:
            truth, truth_meta, markers, truth_source = _read_truth(row, grid)
            head, ideal, image, marker_error, source_paths = _load_views(row, markers)
            ident = row['id']
            names = {k: f'{ident}_{k}.npz' for k in ('truth', 'head', 'ideal', 'image')}
            save_volume(stage / names['truth'], truth, truth_meta)
            meta = {'units': 'line_integral', 'source_kind': _SOURCE_KIND, 'angles_deg': angles,
                    'geometry': row['geometry'], 'calibration_id': row['geometry']['calibration_id'],
                    'detector_index_mapping': 'identity', 'source_angle_sign': 1}
            for kind, data in [('head', head), ('ideal', ideal), ('image', image)]:
                save_projections(stage / names[kind], data, {**meta, 'projection_kind': kind})
            raw.append({'id': ident, 'split': row['split'], 'patient_id': row['head'], 'model_ids': [row['family']],
                        'projection_role': 'measured_with_tool', 'registration_confirmed': True,
                        'source_kind': _SOURCE_KIND, 'truth': names['truth'], 'ideal': names['ideal'],
                        'background': names['image'], 'head_background': names['head'], 'angles_deg': angles})
            if row['split'] == 'test':
                inference.append({'id': ident, 'entity_id': row['family'], 'projections': names['image']})
                truths.append({'id': ident, 'entity_id': row['family'], 'truth': names['truth'], 'ideal': names['ideal']})
            source_files = dict.fromkeys([*row['paths'], truth_source, *source_paths])
            evidence.append({'id': ident, 'head': row['head'], 'family': row['family'], 'variant': row['variant'],
                             'split': row['split'], 'source_split': row['source_split'],
                             'legacy_source_split_declared': row['legacy_source_split_declared'],
                             'stack': row['stack_doc'], 'head_polyfit': row['head_doc'].get('head_polyfit'),
                             'truth_grid': truth_meta, 'marker_centroid_max_error_px': marker_error,
                             'selected_source_views': row['views'], 'files_sha256': {str(p): sha256(p) for p in source_files}})
            del truth, head, image, ideal
        for name, entries in [('raw', raw), ('inference', inference), ('test_truth', truths)]:
            write_json(stage / f'{name}.json', {'cases': entries})
        write_json(stage / 'provenance.json', {'source_kind': _SOURCE_KIND, 'paper_reproduction': False,
                  'acquisition_is_synthetic': True, 'engineering_holdout': engineering_holdout,
                  'angles_deg': angles, 'geometry': records[0]['geometry'], 'fixed_target_grid': grid,
                  'frame': _FRAME, 'cases': evidence})
        if destination.exists():
            destination.rmdir()
        stage.rename(destination)
    return {key: destination / f'{key}.json' for key in ('raw', 'inference', 'test_truth', 'provenance')}
