"""Explicit provenance, group splits and content/physical-grid leakage checks."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .nifti_io import load_volume
from .provenance import file_sha256


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def manifest_digest_matches(path, manifest, expected_digest):
    """Accept a moved Dataset603 manifest only when its old form hashes exactly."""
    current = manifest.to_dict()
    if json_digest(current) == expected_digest:
        return True
    relocation_path = Path(path).parent / 'relocation.json'
    if not relocation_path.is_file():
        return False
    relocation = json.loads(relocation_path.read_text(encoding='utf-8'))
    if (relocation.get('schema_version') != 'phase3_dataset603_relocation_v1'
            or relocation.get('original_manifest_digest') != expected_digest
            or relocation.get('relocated_manifest_digest') != json_digest(current)):
        return False
    old_raw = relocation.get('old_raw_prefix')
    new_raw = relocation.get('new_raw_prefix')
    old_test = relocation.get('old_test_label_prefix')
    new_test = relocation.get('new_test_label_prefix')
    if not all(isinstance(x, str) and x.startswith('/') for x in
               (old_raw, new_raw, old_test, new_test)):
        return False
    original = json.loads(json.dumps(current))
    for case in original['cases']:
        image, label = case['image'], case['label']
        if not image.startswith(new_raw + '/imagesTr/') and not image.startswith(new_raw + '/imagesTs/'):
            return False
        case['image'] = old_raw + image[len(new_raw):]
        if label == new_test + '/' + case['case_id'] + '.nii.gz':
            case['label'] = old_test + '/' + case['case_id'] + '_stent.nii.gz'
        elif label.startswith(new_raw + '/labelsTr/'):
            case['label'] = old_raw + label[len(new_raw):]
        else:
            return False
    return json_digest(original) == expected_digest


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    group_id: str
    source_model_ids: tuple
    patient_id: str | None
    image: Path
    label: Path | None
    angles_deg: tuple
    projection_source: str
    upstream_run_id: str | None
    input_kind: str = 'fdk_3view'


@dataclass
class Manifest:
    cases: tuple
    upstream_registry: dict
    target_semantics: str
    source_directory: str
    group_provenance: dict = field(default_factory=dict)

    def to_dict(self):
        cases=[]
        for case in self.cases:
            row=asdict(case);row['image']=str(case.image);row['label']=str(case.label) if case.label else None
            row['source_model_ids']=list(case.source_model_ids);row['angles_deg']=list(case.angles_deg);cases.append(row)
        return dict(schema_version='fdk_stent_manifest_v1',cases=cases,upstream_registry=self.upstream_registry,
                    target_semantics=self.target_semantics,source_directory=self.source_directory,group_provenance=self.group_provenance)


@dataclass(frozen=True)
class FoldSplit:
    train: tuple
    validation: tuple
    test: tuple

    def to_dict(self): return {k:list(v) for k,v in asdict(self).items()}


def manifest_from_dict(raw, base):
    if raw.get('schema_version') != 'fdk_stent_manifest_v1': raise ValueError('Unknown manifest schema')
    cases=[];seen=set()
    for c in raw.get('cases',[]):
        for key in ('case_id','group_id'):
            if not isinstance(c.get(key),str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',c[key]):
                raise ValueError(f'Invalid {key}')
        if c['case_id'] in seen: raise ValueError('Duplicate case_id')
        seen.add(c['case_id'])
        ids=c.get('source_model_ids')
        if not isinstance(ids,list) or not ids or any(not isinstance(i,str) or not i for i in ids):
            raise ValueError('source_model_ids required')
        kind = c.get('input_kind')
        if kind not in ('fdk_3view', 'fdk_sparse') or c.get('projection_source') not in ('measured','predicted_mask','ideal_mask','simulated_stent_projection'):
            raise ValueError('Unknown input/projection source')
        angles=c.get('angles_deg',[])
        if not isinstance(angles,list) or any(type(a) not in (int, float) for a in angles):
            raise ValueError('Expected a list of numeric angles')
        if kind == 'fdk_3view':
            if angles and (len(angles)!=3 or not np.isfinite(angles).all()):
                raise ValueError('Expected three finite angles')
        elif len(angles) < 2 or not np.isfinite(angles).all() or np.any(np.diff(angles) <= 0) or angles[-1] - angles[0] >= 360:
            raise ValueError('Sparse FDK requires distinct increasing finite angles within one rotation')
        image=(Path(base)/c['image']).resolve()
        label=(Path(base)/c['label']).resolve() if c.get('label') else None
        cases.append(CaseRecord(c['case_id'],c['group_id'],tuple(ids),c.get('patient_id'),image,label,
                                tuple(angles),c['projection_source'],c.get('upstream_run_id'),kind))
    if not cases: raise ValueError('Manifest is empty')
    return Manifest(tuple(cases),raw.get('upstream_registry',{}),raw.get('target_semantics','unknown'),
                    str(raw.get('source_directory',Path(base).resolve())),raw.get('group_provenance',{}))


def load_manifest(path, *, require_labels):
    path=Path(path).resolve()
    manifest=manifest_from_dict(json.loads(path.read_text(encoding='utf8')),path.parent)
    if require_labels and any(c.label is None for c in manifest.cases): raise ValueError('Labels required')
    return manifest


def load_splits(path):
    raw=json.loads(Path(path).read_text(encoding='utf8'))
    if raw.get('schema_version')!='fdk_stent_splits_v1' or not isinstance(raw.get('folds'),dict):
        raise ValueError('Unknown splits schema')
    result={}
    for key,value in raw['folds'].items():
        if not key.isdigit() or str(int(key))!=key or set(value)!={'train','validation','test'}:
            raise ValueError('Invalid fold definition')
        if any(not isinstance(v,list) or not v or any(not isinstance(x,str) for x in v) or len(set(v))!=len(v) for v in value.values()):
            raise ValueError('Splits require nonempty distinct group arrays')
        result[int(key)]=FoldSplit(**{k:tuple(v) for k,v in value.items()})
    if not result: raise ValueError('No folds')
    return result


def select_cases(manifest,split,split_name):
    if split_name not in ('train','validation','test'): raise ValueError('Invalid split_name')
    groups=getattr(split,split_name)
    return tuple(c for c in manifest.cases if c.group_id in groups)


def input_digest(manifest, groups, *, include_labels=False):
    rows=[];hashes={}
    def hash_once(path):
        stat=path.stat();key=(stat.st_dev,stat.st_ino)
        if key not in hashes:hashes[key]=file_sha256(path)
        return hashes[key]
    for c in sorted(manifest.cases,key=lambda c:c.case_id):
        if c.group_id in groups:
            rows.append({'case_id':c.case_id,'group_id':c.group_id,'image_sha256':hash_once(c.image),
                         'label_sha256':hash_once(c.label) if include_labels and c.label else None})
    return json_digest(rows)


def validate_data(manifest,splits,*,formal):
    if manifest.target_semantics not in ('stent_struts', 'stent_and_markers'):
        raise ValueError('Target must be verified stent struts or stent plus markers')
    if formal and len({c.projection_source for c in manifest.cases})!=1:
        raise ValueError('Formal comparisons require one fixed projection_source; use separate condition-specific runs')
    groups={c.group_id for c in manifest.cases}
    if formal and any(not manifest.group_provenance.get(g) for g in groups):
        raise ValueError('Formal data require group_provenance evidence for every group')
    source_owner={};patient_owner={};path_owner={};content_owner={};rows=[]
    decoded={};hashes={}
    def hash_once(path):
        stat=path.stat();key=(stat.st_dev,stat.st_ino)
        if key not in hashes:hashes[key]=file_sha256(path)
        return hashes[key]
    for c in manifest.cases:
        if formal and ((c.input_kind == 'fdk_3view' and len(c.angles_deg) != 3)
                       or (c.input_kind == 'fdk_sparse' and len(c.angles_deg) < 2)):
            raise ValueError('Formal acquisition angles missing')
        for owner,keys in ((source_owner,c.source_model_ids),(patient_owner,(c.patient_id,) if c.patient_id else ())):
            for key in keys:
                if key in owner and owner[key]!=c.group_id: raise ValueError('Shared/transitive source crosses groups')
                owner[key]=c.group_id
        image_stat=c.image.stat()
        label_stat=c.label.stat() if c.label else None
        identity=((image_stat.st_dev,image_stat.st_ino),
                  (label_stat.st_dev,label_stat.st_ino) if label_stat else None)
        if identity not in decoded:
            volume=load_volume(c,require_label=c.label is not None)
            digest=hashlib.sha256(volume.image_raw.astype('<f4').tobytes()).hexdigest()
            grid_digest=json_digest(np.round(volume.meta.affine_mm,6).tolist())
            decoded[identity]=digest,grid_digest
        digest,grid_digest=decoded[identity]
        for owner,key in ((path_owner,str(c.image)),(content_owner,digest)):
            if key in owner and owner[key]!=c.group_id: raise ValueError('Duplicate image across groups')
            owner[key]=c.group_id
        rows.append(dict(case_id=c.case_id,image_sha256=hash_once(c.image),
                         label_sha256=hash_once(c.label) if c.label else None,
                         decoded_sha256=digest,grid_digest=grid_digest))
    for split in splits.values():
        parts=[set(getattr(split,k)) for k in ('train','validation','test')]
        if any(not p for p in parts) or set.union(*parts)!=groups or sum(map(len,parts))!=len(groups):
            raise ValueError('Fold must cover all groups exactly once with disjoint train/validation/test')
        held=parts[1]|parts[2]
        for c in manifest.cases:
            if c.projection_source!='predicted_mask':continue
            registry=manifest.upstream_registry.get(c.upstream_run_id)
            if registry is None:
                if formal:raise ValueError('Missing upstream provenance')
                continue
            keys=('fit_group_ids','selection_group_ids','calibration_group_ids')
            if any(not isinstance(registry.get(k),list) for k in keys):raise ValueError('Incomplete upstream registry')
            used=set().union(*(set(registry[k]) for k in keys))
            if held & used: raise ValueError('Upstream model used held-out entities')
    return {'case_count':len(rows),'group_count':len(groups),'formal':formal,'files':rows,
            'manifest_digest':json_digest(manifest.to_dict()),'file_digest':json_digest(rows)}


def import_nnunet(source,source_groups,output):
    if source_groups is None:raise ValueError('Explicit source_groups mapping is required')
    source=Path(source).resolve();mapping=json.loads(Path(source_groups).read_text(encoding='utf8'))
    if mapping.get('target_semantics')!='stent_struts':raise ValueError('Confirm stent strut semantics; vessel is not stent')
    records=[];seen=set()
    images=sorted((source/'imagesTr').glob('*.nii*'))
    for image in images:
        suffix='.nii.gz' if image.name.endswith('.nii.gz') else '.nii'
        stem=image.name[:-len(suffix)]
        if not stem.endswith('_0000'):raise ValueError('Only single-channel _0000 images accepted')
        case_id=stem[:-5]
        if case_id in seen:raise ValueError('Duplicate image case')
        seen.add(case_id)
        if case_id not in mapping.get('cases',{}):raise ValueError(f'Missing source mapping: {case_id}')
        labels=[p for p in (source/'labelsTr'/f'{case_id}.nii',source/'labelsTr'/f'{case_id}.nii.gz') if p.is_file()]
        if len(labels)!=1:raise ValueError('Missing or ambiguous label')
        row=dict(mapping['cases'][case_id]);row.update(case_id=case_id,image=str(image),label=str(labels[0]))
        records.append(row)
    raw=dict(schema_version='fdk_stent_manifest_v1',target_semantics='stent_struts',cases=records,
             upstream_registry=mapping.get('upstream_registry',{}),group_provenance=mapping.get('group_provenance',{}))
    manifest_from_dict(raw,source)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    path=output/'manifest.json';path.write_text(json.dumps(raw,indent=2),encoding='utf8');return path
