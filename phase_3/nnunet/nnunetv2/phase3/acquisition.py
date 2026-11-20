"""Exact, center-matched TIGRE acquisition datasets with full Phase 2 coverage.

Generation runs in the nnunet environment (which provides TIGRE). Training uses
the separate phase3 environment. No trained Phase 2 predictions are fabricated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np


ARMS = {'a60_n3': (60, 3), 'a45_n3': (45, 3), 'a75_n3': (75, 3),
        'a60_n2': (60, 2), 'a60_n5': (60, 5)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def acquisition_angles(center, span, views):
    if type(views) is not int or views < 2:
        raise ValueError('At least two projections are required')
    if not np.isfinite([center, span]).all() or not 0 < span < 360:
        raise ValueError('Acquisition needs a finite center and span in (0, 360)')
    return (float(center) + np.linspace(-span / 2, span / 2, views)).tolist()


def collect_acquisitions(phase1_manifest, phase2_mapping, *, span, views):
    source = read_json(phase1_manifest)
    mapping = read_json(phase2_mapping)
    fold = source['folds']['fold_0']
    groups = {}
    for name in ('train', 'val', 'test'):
        if not fold[name]:
            raise ValueError('Empty Phase 2 split')
        for group in fold[name]:
            if group in groups:
                raise ValueError('A group crosses splits')
            groups[group] = name
    mapped = {}
    for row in mapping:
        key = (row['case_id'], row['view_id'])
        if key in mapped:
            raise ValueError('Duplicate Phase 2 view')
        mapped[key] = row
    records, expected, ids = [], set(), set()
    for case in sorted(source['cases'], key=lambda c: c['case_id']):
        cid, group = case['case_id'], case['group_id']
        if cid in ids or group not in groups:
            raise ValueError('Duplicate case or uncovered source group')
        ids.add(cid)
        original = sorted(case['views'], key=lambda v: v['angle_deg'])
        angles = [float(v['angle_deg']) for v in original]
        if len(angles) != 3 or not np.isfinite(angles).all() or not np.allclose(np.diff(angles), 30):
            raise ValueError('Expected the original 60-degree Phase 2 triplets')
        for view in original:
            key = (cid, view['view_id'])
            if key in expected:
                raise ValueError('Duplicate source view')
            expected.add(key)
            row = mapped.get(key)
            if (row is None or row['group_id'] != group or row['split'] != groups[group]
                    or not np.isclose(row['angle_deg'], view['angle_deg'], atol=1e-6, rtol=0)):
                raise ValueError(f'Phase 2 inventory/split/angle mismatch: {key}')
        center = (angles[0] + angles[-1]) / 2
        records.append(dict(case_id=cid, group_id=group,
                            split='validation' if groups[group] == 'val' else groups[group],
                            sample=case['metadata']['sample'], variant=str(case['metadata']['stent_variant']),
                            source_split=case['metadata']['source_split'],
                            source_angles_deg=angles, center_deg=center,
                            angles_deg=acquisition_angles(center, span, views)))
    if expected != set(mapped):
        raise ValueError('Phase 1 and Phase 2 must cover exactly the same views')
    splits = dict(schema_version='fdk_stent_splits_v1', folds={'0': {
        'train': sorted(fold['train']), 'validation': sorted(fold['val']), 'test': sorted(fold['test'])}})
    return records, splits


def _geometry(info):
    import tigre
    geo = tigre.geometry()
    for name, key, dtype in [('nVoxel','nVoxel', np.int32), ('dVoxel','dVoxel_mm', np.float32),
                             ('nDetector','nDetector', np.int32), ('dDetector','dDetector_mm', np.float32)]:
        setattr(geo, name, np.asarray(info[key], dtype=dtype))
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.DSO, geo.DSD = float(info['DSO_mm']), float(info['DSD_mm'])
    geo.offOrigin = np.asarray(info.get('offOrigin_mm', [0, 0, 0]), np.float32)
    geo.offDetector = np.asarray(info.get('offDetector_mm', [0, 0]), np.float32)
    geo.rotDetector = np.zeros(3, np.float32)
    geo.COR, geo.accuracy, geo.mode = 0., 0.5, 'cone'
    return geo


def _save_volume(path, data, affine):
    import nibabel as nib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units('mm')
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    temporary = path.with_name(path.name + '.partial.nii.gz')
    nib.save(image, temporary)
    temporary.replace(path)


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(destination)
    os.link(source, destination)


def build_dataset(*, phase1_root, phase1_manifest, phase2_mapping, output, span, views, gpu, resume=False):
    """Build complete full-grid data; resume only verified per-volume artifacts."""
    import nibabel as nib
    import tifffile
    import tigre
    from tigre.algorithms import fdk
    from tigre.utilities.gpu import GpuIds

    root, output = Path(phase1_root).resolve(), Path(output).resolve()
    records, splits = collect_acquisitions(phase1_manifest, phase2_mapping, span=span, views=views)
    groups = sorted({(r['group_id'], r['variant']) for r in records})
    inputs = [Path(phase1_manifest), Path(phase2_mapping)]
    for group, variant in groups:
        inputs += [root/'stent_models'/group/variant/name for name in
                   ('stent_real_grid.nii.gz', 'stent.json', 'marker.json')]
        inputs.append(root/'stent_projections'/group/variant/'projection.json')
    # Fail before output if any original Phase 2-linked image/mask is missing.
    source = read_json(phase1_manifest)
    for case in source['cases']:
        for view in case['views']:
            for key in ('projection_path', 'mask_path'):
                path = (Path(phase1_manifest).parent / view[key]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
    identity = dict(schema_version='phase3_acquisition_build_v1', span_deg=span, views=views,
                    source_sha256={str(p.resolve()): sha256(p) for p in inputs},
                    projection_source='simulated_stent_projection', field_of_view='full_source',
                    tigre_version=getattr(tigre, '__version__', 'unknown'))
    if output.exists():
        if not resume or not (output/'build_identity.json').is_file() or read_json(output/'build_identity.json') != identity:
            raise FileExistsError(f'Existing dataset is not an identical resumable build: {output}')
    else:
        output.mkdir(parents=True)
        write_json(output/'build_identity.json', identity)
    gpuids = GpuIds()
    gpuids.devices = [int(gpu)]
    unique, manifest, provenance, model_cache = {}, [], [], {}
    baseline_checks = []
    started = time.monotonic()
    for index, record in enumerate(records):
        group, variant = record['group_id'], record['variant']
        model_dir = root/'stent_models'/group/variant
        projection_dir = root/'stent_projections'/group/variant
        info = read_json(projection_dir/'projection.json')
        head_info = read_json(root/'stacked'/record['source_split']/record['sample']/group/variant/'projection.json')
        geo_info = info['geometry']
        for name in ('DSO_mm','DSD_mm','nVoxel','dVoxel_mm','nDetector','dDetector_mm','offOrigin_mm','offDetector_mm'):
            for other in ('head_geometry', 'stent_geometry'):
                if not np.allclose(geo_info[name], head_info[other][name], atol=1e-5, rtol=0):
                    raise ValueError(f'Geometry mismatch for {record["case_id"]}: {name}')
        key = hashlib.sha256(json.dumps([group, variant, record['angles_deg'], geo_info], sort_keys=True).encode()).hexdigest()[:20]
        if key not in unique:
            unit_file = output/'units'/f'{key}.json'
            image_path = output/'unique'/f'{key}.nii.gz'
            label_path = output/'targets'/f'{group}_{variant}.nii.gz'
            if unit_file.exists():
                unit = read_json(unit_file)
                if (not image_path.is_file() or not label_path.is_file()
                        or sha256(image_path) != unit['image_sha256'] or sha256(label_path) != unit['label_sha256']):
                    raise ValueError(f'Damaged resumable volume: {key}')
            else:
                if shutil.disk_usage(output).free < 5 * 2**30:
                    raise OSError('Less than 5 GiB free; stopping before reconstruction')
                if (group, variant) not in model_cache:
                    model_cache.clear()  # one full-grid model at a time
                    nii = nib.load(model_dir/'stent_real_grid.nii.gz')
                    label = np.asarray(nii.dataobj, dtype=np.uint8)
                    if not np.isin(label, [0, 1]).all() or not label.any():
                        raise ValueError('Expected a nonempty binary stent and marker target')
                    meta = read_json(model_dir/'stent.json')
                    if (tuple(geo_info['nVoxel'][::-1]) != label.shape
                            or not np.allclose(np.diag(nii.affine)[:3], geo_info['dVoxel_mm'][::-1], atol=1e-6)):
                        raise ValueError('Source label and acquisition geometry differ')
                    attenuation = label.astype(np.float32) * float(info['stent_mu'])
                    for marker in read_json(model_dir/'marker.json')['markers']:
                        coords = np.asarray(marker['voxels_ijk'], int)
                        if not np.all(label[tuple(coords.T)] == 1):
                            raise ValueError('Marker target missing')
                        attenuation[tuple(coords.T)] = float(info['marker_mu'])
                    model_cache[(group, variant)] = (np.ascontiguousarray(attenuation.transpose(2, 1, 0)), nii.affine)
                    if not label_path.exists():
                        _save_volume(label_path, label, nii.affine)
                    del label, attenuation
                volume, affine = model_cache[(group, variant)]
                angles = np.deg2rad(np.asarray(record['angles_deg'], np.float32))
                geometry = _geometry(geo_info)
                projections = tigre.Ax(volume, deepcopy(geometry), angles, gpuids=gpuids)
                if not np.isfinite(projections).all() or np.any(projections < 0):
                    raise ValueError('Invalid generated projection')
                checks = []
                for i, angle in enumerate(record['angles_deg']):
                    existing = projection_dir/'projections/implant'/f'{int(round(angle)):03d}.tif'
                    if abs(angle-round(angle)) < 1e-8 and existing.is_file():
                        old = tifffile.imread(existing)
                        error = float(np.max(np.abs(old-projections[i])))
                        if not np.allclose(old, projections[i], atol=2e-5, rtol=2e-5):
                            raise ValueError(f'Existing projection mismatch: {group}, {angle}, max_abs={error}')
                        checks.append(dict(angle_deg=angle, max_absolute_error=error, source_sha256=sha256(existing)))
                reconstruction = np.asarray(fdk(projections, geometry, angles, gpuids=gpuids), np.float32)
                if reconstruction.shape != volume.shape or not np.isfinite(reconstruction).all():
                    raise ValueError('Invalid FDK volume')
                _save_volume(image_path, reconstruction.transpose(2, 1, 0), affine)
                unit = dict(key=key, group_id=group, angles_deg=record['angles_deg'],
                            image_sha256=sha256(image_path), label_sha256=sha256(label_path),
                            projection_sha256=hashlib.sha256(projections.tobytes()).hexdigest(),
                            existing_projection_checks=checks,
                            shape_xyz=list(reconstruction.shape[::-1]),
                            spacing_xyz_mm=geo_info['dVoxel_mm'][::-1],
                            stent_mu=info['stent_mu'], marker_mu=info['marker_mu'])
                write_json(unit_file, unit)
                del projections, reconstruction
            unique[key] = (image_path, label_path, unit)
            baseline_checks.extend(unit['existing_projection_checks'])
            print(json.dumps(dict(event='volume_complete', unique_volumes=len(unique),
                                  case_id=record['case_id'], seconds=round(time.monotonic()-started, 1))), flush=True)
        image_path, label_path, unit = unique[key]
        image = output/'images'/f'{record["case_id"]}_fdk.nii.gz'
        label = output/'labels'/f'{record["case_id"]}_stent.nii.gz'
        _link(image_path, image)
        _link(label_path, label)
        manifest.append(dict(case_id=record['case_id'], group_id=group,
                             source_model_ids=[f'stent_model:{group}/{variant}'], patient_id=None,
                             image=str(image), label=str(label), angles_deg=record['angles_deg'],
                             input_kind='fdk_sparse', projection_source='simulated_stent_projection', upstream_run_id=None))
        provenance.append(dict(record, unique_volume_key=key, source_model=str(model_dir)))
        if (index+1) % 100 == 0:
            write_json(output/'build_status.json', dict(state='building', completed_cases=index+1, total_cases=len(records),
                       unique_volumes=len(unique), pid=os.getpid()))
    counts = dict(Counter(r['split'] for r in records))
    unique_counts = {split: len({p['unique_volume_key'] for p in provenance if p['split'] == split})
                     for split in ('train','validation','test')}
    audit = dict(state='complete', case_count=len(records), split_counts=counts,
                 unique_volume_count=len(unique), unique_counts=unique_counts,
                 physical_counts={split: len({(r['sample'],r['group_id'],r['variant']) for r in records if r['split']==split})
                                  for split in counts},
                 existing_projection_checks=len(baseline_checks),
                 max_existing_projection_error=max((c['max_absolute_error'] for c in baseline_checks), default=None),
                 source_view_count=len(read_json(phase2_mapping)), pid=os.getpid())
    write_json(output/'splits.json', splits)
    write_json(output/'source_mapping.json', provenance)
    write_json(output/'manifest.json', dict(schema_version='fdk_stent_manifest_v1', cases=manifest,
               target_semantics='stent_and_markers', upstream_registry={},
               group_provenance={g:f'Original stent model {root}/stent_models/{g}/{v}; head replicas share ideal implant projections'
                                 for g,v in groups}))
    write_json(output/'audit.json', audit)
    write_json(output/'build_status.json', audit)
    print(json.dumps(audit), flush=True)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-root', type=Path, required=True)
    parser.add_argument('--phase1-manifest', type=Path, required=True)
    parser.add_argument('--phase2-mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--span', type=float, required=True)
    parser.add_argument('--views', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0, help='process-visible GPU index')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    build_dataset(**vars(args))


if __name__ == '__main__':
    main()
