"""Case-level evaluation in physical coordinates, with explicit failure accounting."""
from pathlib import Path
import csv
import json
import numpy as np
import skimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from stent_repro.io import load_volume,load_projections

METRICS=('dice','precision','recall','pred_to_truth_mm','truth_to_pred_mm')


def validate_grid(grid):
    if not isinstance(grid,dict): raise ValueError('grid must be a mapping')
    for key in ('shape','spacing_mm','origin_mm','direction'):
        if key not in grid: raise ValueError(f'grid missing {key}')
    shape=grid['shape']; spacing=np.asarray(grid['spacing_mm'],float); origin=np.asarray(grid['origin_mm'],float)
    if len(shape)!=3 or any(type(x) is not int or x<1 for x in shape): raise ValueError('invalid grid shape')
    if spacing.shape!=(3,) or origin.shape!=(3,) or not np.isfinite(spacing).all() or not np.isfinite(origin).all() or (spacing<=0).any():
        raise ValueError('invalid grid spacing/origin')
    if np.shape(grid['direction'])!=(3,3) or not np.allclose(grid['direction'],np.eye(3),atol=1e-8): raise ValueError('grid direction must be identity; rotated grids unsupported')
    return {'shape':list(shape),'spacing_mm':spacing.tolist(),'origin_mm':origin.tolist(),'direction':np.eye(3).tolist()}


def resample_volume(array,source_grid,target_grid,order=1):
    source=validate_grid(source_grid); target=validate_grid(target_grid); array=np.asarray(array)
    if array.shape!=tuple(source['shape']) or not np.isfinite(array).all(): raise ValueError('source grid/array mismatch or nonfinite data')
    if order not in (0,1): raise ValueError('only nearest and linear resampling supported')
    so=np.array(source['origin_mm']); ss=np.array(source['spacing_mm'])
    to=np.array(target['origin_mm']); ts=np.array(target['spacing_mm'])
    slo=so-ss/2; shi=so+(np.array(source['shape'][::-1])-.5)*ss
    tlo=to-ts/2; thi=to+(np.array(target['shape'][::-1])-.5)*ts
    if np.any(tlo<slo-1e-6) or np.any(thi>shi+1e-6): raise ValueError('prediction grid does not cover evaluation field of view')
    result=np.empty(target['shape'],dtype=np.float32)
    for start in range(0,target['shape'][0],16):
        z,y,x=np.meshgrid(np.arange(start,min(start+16,target['shape'][0])),np.arange(target['shape'][1]),np.arange(target['shape'][2]),indexing='ij')
        coords=[(to[i]+v*ts[i]-so[i])/ss[i] for i,v in zip((2,1,0),(z,y,x))]
        result[start:start+len(z)]=map_coordinates(array.astype(np.float32,copy=False),coords,order=order,mode='nearest')
    return result


def segmentation_metrics(pred,truth,grid):
    g=validate_grid(grid); p=np.asarray(pred); t=np.asarray(truth)
    if p.shape!=t.shape or p.shape!=tuple(g['shape']): raise ValueError('mask shapes must match grid')
    if not np.isin(p,[0,1]).all() or not np.isin(t,[0,1]).all(): raise ValueError('metrics require binary masks')
    p=p.astype(bool);t=t.astype(bool); pn=int(p.sum());tn=int(t.sum());tp=int((p&t).sum())
    result={'status':'ok','pred_voxels':pn,'truth_voxels':tn,
            'dice':2*tp/(pn+tn) if pn+tn else None,'precision':tp/pn if pn else None,
            'recall':tp/tn if tn else None,'pred_to_truth_mm':None,'truth_to_pred_mm':None}
    if not tn: result['status']='empty_truth';return result
    if not pn: result['status']='empty_prediction';return result
    points=[]
    for mask in (p,t):
        indices=np.argwhere(skeletonize(mask,method='lee'))
        points.append(indices[:,::-1]*np.array(g['spacing_mm'])+np.array(g['origin_mm']))
    if any(len(x)==0 for x in points): result['status']='empty_skeleton';return result
    result['pred_to_truth_mm']=float(cKDTree(points[1]).query(points[0])[0].mean())
    result['truth_to_pred_mm']=float(cKDTree(points[0]).query(points[1])[0].mean())
    return result


def projection_metrics(pred,truth):
    p=np.asarray(pred,dtype=float);t=np.asarray(truth,dtype=float)
    if p.shape!=t.shape or not p.size or not np.isfinite(p).all() or not np.isfinite(t).all() or (t<0).any(): raise ValueError('invalid projection pair')
    error=np.abs(p-t); foreground=t>0
    return {'mae':float(error.mean()),'foreground_mae':float(error[foreground].mean()) if foreground.any() else None,
            'background_mae':float(error[~foreground].mean()) if (~foreground).any() else None,
            'units':'line_integral','mape':None,'mape_reason':'paper-specific measured support and filtering not supplied'}


def summarize_cases(rows):
    if not rows: raise ValueError('cannot summarize an empty case list')
    ids=[r['case_id'] for r in rows]
    if len(set(ids))!=len(ids): raise ValueError('duplicate case IDs')
    failures=sum(r.get('status')!='ok' for r in rows)
    metrics={}
    for key in METRICS:
        values=[r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        metrics[key]={'n':len(values),'mean':float(np.mean(values)) if values else None,
                      'std':float(np.std(values,ddof=1)) if len(values)>1 else (0.0 if values else None)}
    return {'case_count':len(rows),'failure_count':failures,'failure_rate':failures/len(rows),
            'metrics':metrics,'aggregation':'macro per case; each metric reports valid count; structural distances conditional on valid nonempty skeletons',
            'skeletonization':{'algorithm':'Lee','library':'scikit-image','version':skimage.__version__},
            'cases':rows}


def evaluate_manifest(path,output):
    path=Path(path).resolve();doc=json.loads(path.read_text(encoding='utf-8')); rows=doc.get('cases',[])
    if not rows or len({r.get('id') for r in rows})!=len(rows): raise ValueError('nonempty unique case IDs required')
    threshold=doc.get('threshold',.5)
    if not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('invalid threshold')
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('evaluation output must be empty')
    result=[]
    for row in rows:
        entry={'case_id':row['id'],'entity_id':row.get('entity_id',row['id'])}
        try:
            if row.get('status')=='pipeline_failure': raise ValueError(row.get('error','inference failed'))
            pred,pm=load_volume(path.parent/row['probability']);truth,tm=load_volume(path.parent/row['truth'])
            pg={**pm,'shape':list(pred.shape)};tg={**tm,'shape':list(truth.shape)}
            if (pred<0).any() or (pred>1).any(): raise ValueError('probabilities must be in [0,1]')
            aligned=resample_volume(pred,pg,tg)
            entry.update(segmentation_metrics(aligned>=threshold,truth,tg))
            if doc.get('tolerance_radius_mm') is not None:
                from stent_repro.paper_metrics import project_tolerance_metrics
                tolerance=project_tolerance_metrics(aligned>=threshold,truth,tg,doc['tolerance_radius_mm'])
                entry.update({key:tolerance[key] for key in ('project_dilated_truth_precision','project_eroded_truth_recall')})
            if not np.any(truth) and doc.get('allow_empty_truth',False):
                entry['negative_control']=True;entry['status']='ok' if not np.any(aligned>=threshold) else 'false_positive_negative_control'
        except (ValueError,KeyError,OSError,TypeError) as exc:
            entry.update({'status':'pipeline_failure','error':str(exc)})
        result.append(entry)
    report=summarize_cases(result);report['threshold']=threshold
    from stent_repro.statistics import summarize_entities
    report['entity_summary']=summarize_entities(result,expected_case_entities={r['id']:r.get('entity_id',r['id']) for r in rows})
    if doc.get('tolerance_radius_mm') is not None:
        report['project_tolerance']={'radius_mm':doc['tolerance_radius_mm'],'paper_table4_equivalence_verified':False,
            'definition':'Euclidean voxel-centre ball; dilated-truth precision and eroded-truth recall; exterior background'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fields=['case_id','entity_id','status',*METRICS,'pred_voxels','truth_voxels','negative_control','error']
    if doc.get('tolerance_radius_mm') is not None:fields+=['project_dilated_truth_precision','project_eroded_truth_recall']
    with (output/'metrics_per_case.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(result)
    return report


def evaluate_dte_manifest(path,output):
    """Measured-support protocol is opt-in; simulation is not labelled Table 3."""
    from stent_repro.paper_metrics import dte_mape_measured_support
    from stent_repro.provenance import fingerprint_files,verify_fingerprints,write_json
    path=Path(path).resolve();doc=json.loads(path.read_text(encoding='utf8'));rows=doc.get('cases',[])
    if doc.get('support_origin')!='measured_prior_subtraction':raise ValueError('measured prior-subtraction support must be explicitly confirmed')
    if not rows or len({r.get('id') for r in rows})!=len(rows):raise ValueError('unique nonempty DTE cases required')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('DTE evaluation output must be empty')
    sources=[path]+[path.parent/r[key] for r in rows for key in ('prediction','measured','support')]
    frozen=fingerprint_files(sources);results=[]
    for row in rows:
        entry={'case_id':row['id'],'entity_id':row.get('entity_id',row['id'])}
        try:
            pred,pm=load_projections(path.parent/row['prediction']);truth,tm=load_projections(path.parent/row['measured']);support,_=load_projections(path.parent/row['support'])
            if pm.get('units')!='line_integral' or tm.get('units')!='line_integral':raise ValueError('DTE values require declared line_integral units')
            entry.update(dte_mape_measured_support(pred,truth,support));entry['status']='ok' if entry['mape_percent'] is not None else 'no_valid_support'
        except (ValueError,OSError,TypeError) as exc:entry.update(status='pipeline_failure',error=str(exc))
        results.append(entry)
    verify_fingerprints(frozen)
    from stent_repro.statistics import summarize_entities
    report={'cases':results,'entity_summary':summarize_entities(results,metrics=['mape_percent']),
            'source_fingerprints':frozen,'paper_table3_equivalence_verified':False,
            'protocol':'measured explicit support; per-view median3 reflect; zero filtered denominators excluded and counted'}
    output.mkdir(parents=True,exist_ok=True);write_json(output/'summary.json',report);return report
