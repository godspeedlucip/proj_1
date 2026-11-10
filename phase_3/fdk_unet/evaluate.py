"""Original-grid segmentation metrics with equal source-group aggregation."""
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from .manifest import select_cases,json_digest
from .prior_weights import file_sha256
from .nifti_io import load_volume,read_nifti,same_grid
from .predict import Prediction


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    group_id: str
    dice: float
    precision: float
    recall: float
    false_positive_mm3: float
    pred_components: int
    target_components: int
    skeleton_distance_mm: float | None
    empty_status: str


def compute_case_metrics(prediction,target,*,case_id,group_id):
    p,y=prediction.mask,target
    if p.shape!=y.shape or not np.isin(p,[0,1]).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Metrics require matching binary original-grid masks')
    p,y=p.astype(bool),y.astype(bool);npred,ny=int(p.sum()),int(y.sum());tp=int((p&y).sum())
    status='none';distance=None
    if not npred and not ny:dice=precision=recall=1.;distance=0.;status='both_empty'
    elif not npred:dice=precision=recall=0.;status='empty_prediction'
    elif not ny:dice=precision=0.;recall=1.;status='empty_target'
    else:
        dice=2*tp/(npred+ny);precision=tp/npred;recall=tp/ny
        points=[]
        for mask in (p,y):
            ijk=np.argwhere(skeletonize(mask,method='lee'))[:,::-1]
            points.append(ijk@prediction.meta.affine_mm[:3,:3].T+prediction.meta.affine_mm[:3,3])
        if not all(len(a) for a in points):status='empty_skeleton'
        else:
            distance=float((cKDTree(points[0]).query(points[1])[0].mean()+cKDTree(points[1]).query(points[0])[0].mean())/2)
    structure=np.ones((3,3,3),bool)
    return CaseMetrics(case_id,group_id,float(dice),float(precision),float(recall),
                       float((npred-tp)*abs(np.linalg.det(prediction.meta.affine_mm[:3,:3]))),
                       int(label(p,structure)[1]),int(label(y,structure)[1]),distance,status)


def aggregate_group_metrics(rows):
    rows=list(rows)
    if not rows or len({r.case_id for r in rows})!=len(rows):raise ValueError('Empty or duplicate case metrics')
    keys=('dice','precision','recall','false_positive_mm3','pred_components','target_components','skeleton_distance_mm')
    groups={}
    for group_id in sorted({r.group_id for r in rows}):
        members=[r for r in rows if r.group_id==group_id];values={}
        for key in keys:
            valid=[getattr(r,key) for r in members if getattr(r,key) is not None]
            values[key]=float(np.mean(valid)) if valid else None
        values.update(case_count=len(members),distance_valid_cases=sum(r.skeleton_distance_mm is not None for r in members),
                      distance_failures=sum(r.skeleton_distance_mm is None for r in members))
        groups[group_id]=values
    mean,std,denominator={},{},{}
    for key in keys:
        valid=[g[key] for g in groups.values() if g[key] is not None]
        mean[key]=float(np.mean(valid)) if valid else None;std[key]=float(np.std(valid,ddof=0)) if valid else None
        denominator[key]=len(valid)
    return {'groups':groups,'mean':mean,'std':std,'valid_group_denominators':denominator,
            'case_count':len(rows),'group_count':len(groups),'std_ddof':0,
            'distance_valid_cases':sum(r.skeleton_distance_mm is not None for r in rows),
            'distance_failures':sum(r.skeleton_distance_mm is None for r in rows),
            'distance_failure_cases':[r.case_id for r in rows if r.skeleton_distance_mm is None]}


def evaluate_predictions(manifest,split,predictions,*,split_name,output,expected_fold=None):
    cases=select_cases(manifest,split,split_name);predictions=Path(predictions);rows=[]
    provenance_path=predictions/'prediction_run.json'
    if not provenance_path.is_file():raise ValueError('Evaluation requires experimental prediction provenance')
    provenance=json.loads(provenance_path.read_text(encoding='utf8'))
    if (provenance.get('schema_version')!='fdk_predictions_v1' or provenance.get('mode')!='experimental'
            or provenance.get('split_digest')!=json_digest(split.to_dict()) or provenance.get('split_name')!=split_name
            or (expected_fold is not None and provenance.get('checkpoint_fold')!=expected_fold)):
        raise ValueError('Prediction provenance does not match requested evaluation fold/split')
    if (provenance.get('case_groups')!={c.case_id:c.group_id for c in cases}
            or provenance.get('projection_sources')!={c.case_id:c.projection_source for c in cases}):
        raise ValueError('Evaluation cases/groups/conditions differ from prediction provenance')
    ids={c.case_id for c in cases}
    for suffix in ('mask','prob'):
        actual={p.name[:-len(f'_{suffix}.nii.gz')] for p in predictions.glob(f'*_{suffix}.nii.gz')}
        if actual!=ids:raise ValueError(f'Prediction IDs differ: {suffix}')
    for case in cases:
        audit=provenance['outputs'][case.case_id]
        if (file_sha256(predictions/f'{case.case_id}_mask.nii.gz')!=audit['mask_sha256']
                or file_sha256(predictions/f'{case.case_id}_prob.nii.gz')!=audit['prob_sha256']
                or case.label is None or file_sha256(case.label)!=provenance['label_sha256'][case.case_id]):
            raise ValueError('Prediction or evaluation label differs from bound artifact')
        volume=load_volume(case,require_label=True)
        mask,meta=read_nifti(predictions/f'{case.case_id}_mask.nii.gz')
        prob,prob_meta=read_nifti(predictions/f'{case.case_id}_prob.nii.gz')
        if not same_grid(meta,volume.meta) or not same_grid(prob_meta,volume.meta):raise ValueError('Prediction physical grid differs')
        if np.any((prob<0)|(prob>1)):raise ValueError('Invalid probability range')
        rows.append(compute_case_metrics(Prediction(prob,mask,meta,{}),volume.target,case_id=case.case_id,group_id=case.group_id))
    summary=aggregate_group_metrics(rows)
    summary['provenance']={key:provenance[key] for key in ('checkpoint_sha256','checkpoint_fold','split_digest','split_name')}
    summary['projection_sources']=sorted({c.projection_source for c in cases})
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'metrics_per_case.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(asdict(rows[0])));writer.writeheader();writer.writerows(asdict(r) for r in rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    (output/'metrics_per_group.json').write_text(json.dumps(summary['groups'],indent=2,allow_nan=False),encoding='utf8')
    return summary
