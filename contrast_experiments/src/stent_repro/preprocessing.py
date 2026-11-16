"""CPU reference preparation for calibrated, pre-aligned synthetic training data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from stent_repro.geometry import Grid,circular_view,forward_project,backproject_each,fdk
from stent_repro.io import load_projections,load_volume,save_projections,save_volume
from stent_repro.training.data import read_training_manifest
from stent_repro.configuration import ANGLE_PATTERNS,validate_profile_angles,validate_projection_metadata


def project_tool(truth,grid,views,params,density=None):
    spectrum=params.get('spectral')
    if spectrum is None:return forward_project(truth.astype(np.float32)*params['attenuation_per_mm'],grid,views,step_mm=params['integration_step_mm'])
    from copy import deepcopy
    from stent_repro.spectral import spectral_project
    spectrum=deepcopy(spectrum)
    if density is not None:
        spectrum['materials']['tool']['density_g_cm3']=float(density)
    length=forward_project(truth.astype(np.float32),grid,views,step_mm=params['integration_step_mm'])
    return spectral_project({'tool':length},**spectrum).line_integral.astype(np.float32)


def _load_native_ideal(row,key,root,angles,params):
    if row.get('registration_confirmed') is not True:raise ValueError('native tool target requires confirmed registration with truth')
    value,meta=load_projections(root/row[key])
    if value.shape!=(len(angles),*params['detector_shape']) or (value<0).any() or meta.get('units')!='line_integral':raise ValueError('invalid native ideal tool target')
    validate_projection_metadata(meta,{'geometry':{**params,'calibration_id':params.get('calibration_id')}},angles=angles)
    return value.astype(np.float32)


def intensity_to_line_integrals(intensity, *, flat_field, dark_field, transmission_floor):
    values=np.asarray(intensity,dtype=np.float64)
    if not all(isinstance(v,(int,float)) and np.isfinite(v) for v in (flat_field,dark_field,transmission_floor)) or flat_field<=dark_field or not 0<transmission_floor<1:
        raise ValueError('flat, dark and transmission floor are invalid')
    if not np.isfinite(values).all(): raise ValueError('intensity must be finite')
    transmission=(values-dark_field)/(flat_field-dark_field)
    return (-np.log(np.clip(transmission,transmission_floor,1))).astype(np.float32)

def _case_arrays(row,truth_path,bg_path,params):
    truth,meta=load_volume(truth_path)
    extra=row.get('truth_grid',{})
    if not isinstance(extra,dict): raise ValueError('truth_grid must be an object')
    for key in ('spacing_mm','origin_mm','direction'):
        if key in extra:
            if key in meta and not np.allclose(meta[key],extra[key]): raise ValueError(f'truth grid metadata conflict: {key}')
            meta[key]=extra[key]
    grid=_grid(truth,meta)
    background,bg_meta=load_projections(bg_path)
    for case_key,meta_key in (('background_units','units'),('background_angles_deg','angles_deg')):
        if case_key in row:
            if meta_key in bg_meta:
                old,new=bg_meta[meta_key],row[case_key]
                same=old==new if meta_key=='units' else len(old)==len(new) and np.allclose(old,new)
                if not same: raise ValueError(f'background metadata conflict: {meta_key}')
            bg_meta[meta_key]=row[case_key]
    if bg_meta.get('units')=='detector_intensity':
        if any(k not in params for k in ('flat_field','dark_field','transmission_floor')): raise ValueError('detector intensity requires flat_field, dark_field and transmission_floor')
        background=intensity_to_line_integrals(background,flat_field=params['flat_field'],dark_field=params['dark_field'],transmission_floor=params['transmission_floor'])
        bg_meta['units']='line_integral'
    return truth,meta,grid,background,bg_meta

def synthesize_dte(background, tool, rng, *, photons, blur_sigma, scatter_fraction, flip):
    background=np.asarray(background,dtype=np.float32); tool=np.asarray(tool,dtype=np.float32)
    if background.shape!=tool.shape or background.ndim!=2: raise ValueError('background and tool must be equal-sized 2D arrays')
    if not np.isfinite(background).all() or not np.isfinite(tool).all() or (background<0).any() or (tool<0).any(): raise ValueError('nonnegative finite line integrals required')
    if photons<=0 or blur_sigma<0 or scatter_fraction<0: raise ValueError('invalid noise parameters')
    if flip:
        if rng.integers(2): tool=np.flip(tool,axis=0)
        if rng.integers(2): tool=np.flip(tool,axis=1)
    expected=photons*(np.exp(-(background+tool))+scatter_fraction)
    observed=rng.poisson(expected)
    # Continuity correction avoids log(0); no claim of detector calibration.
    image=-np.log(np.maximum(observed,1)/photons)
    if blur_sigma: image=gaussian_filter(image,blur_sigma,mode='nearest')
    return image.astype(np.float32),np.array(tool,dtype=np.float32,copy=True)


def perturb_projection(projection,rng,*,false_negative_rate,false_positive_rate,false_positive_scale):
    p=np.asarray(projection,dtype=np.float32)
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any(): raise ValueError('nonnegative 2D projection required')
    if not 0<=false_negative_rate<=1 or not 0<=false_positive_rate<=1 or false_positive_scale<0: raise ValueError('invalid error rates')
    result=p.copy()
    result[(p>0)&(rng.random(p.shape)<false_negative_rate)]=0
    false_positives=(p==0)&(rng.random(p.shape)<false_positive_rate)
    result[false_positives] = rng.uniform(0,false_positive_scale*float(p.max()),false_positives.sum())
    return result


def _grid(data,meta):
    if 'spacing_mm' not in meta or 'origin_mm' not in meta: raise ValueError('truth grid requires spacing_mm and origin_mm')
    if 'direction' not in meta: raise ValueError('truth grid requires direction')
    spacing=np.asarray(meta['spacing_mm']); origin=np.asarray(meta['origin_mm']); direction=np.asarray(meta['direction'])
    if spacing.shape!=(3,) or origin.shape!=(3,) or spacing.dtype.kind not in 'iuf' or origin.dtype.kind not in 'iuf' or not np.isfinite(spacing).all() or not np.isfinite(origin).all() or (spacing<=0).any(): raise ValueError('invalid truth grid spacing or origin')
    if direction.shape!=(3,3) or direction.dtype.kind not in 'iuf' or not np.isfinite(direction).all() or not np.allclose(direction,np.eye(3)): raise ValueError('rotated grids require physical resampling; unsupported')
    if not np.all((data==0)|(data==1)): raise ValueError('truth volume must be a binary combined device mask')
    return Grid(tuple(data.shape),tuple(meta['spacing_mm']),tuple(meta['origin_mm']))


def crop_volume_grid(volume, grid, shape):
    if len(shape)!=3 or any(not isinstance(n,int) or n<1 for n in shape) or any(n>m for n,m in zip(shape,grid.shape)):
        raise ValueError('invalid DTR crop shape')
    start=[(m-n)//2 for m,n in zip(grid.shape,shape)]
    cropped=volume[tuple(slice(a,a+n) for a,n in zip(start,shape))]
    origin=tuple(o+i*s for o,i,s in zip(grid.origin_mm,reversed(start),grid.spacing_mm))
    return cropped, Grid(tuple(shape),grid.spacing_mm,origin)

def _crop(background,tool,size,min_background,rng):
    h,w=background.shape
    if size<1 or h<size or w<size: raise ValueError('patch_size exceeds detector')
    for _ in range(100):
        y=int(rng.integers(h-size+1)); x=int(rng.integers(w-size+1))
        b=background[y:y+size,x:x+size]; t=tool[y:y+size,x:x+size]
        if float(b.mean())>=min_background: return b,t
    raise ValueError('could not find a background patch above min_background')


def prepare_samples(manifest_path, output_dir, *, method, profile, params, export_pools=False):
    if method not in ('A','B'): raise ValueError('method must be A or B')
    supported={name for name in ANGLE_PATTERNS if name.startswith(method+'-')}
    if profile not in supported: raise ValueError('profile does not match method')
    required=('sdd_mm','sid_mm','pixel_size_mm','detector_shape','attenuation_per_mm','photons','blur_sigma_px','scatter_fraction','false_negative_rate','false_positive_rate','false_positive_scale','integration_step_mm','patch_size','min_background','seed')
    if not isinstance(params,dict) or any(k not in params for k in required): raise ValueError('all physical and augmentation parameters must be explicit')
    numeric=[k for k in required if k not in ('detector_shape','patch_size','seed')]
    if any(not isinstance(params[k],(int,float)) or isinstance(params[k],bool) or not np.isfinite(params[k]) for k in numeric): raise ValueError('physical and augmentation parameters must be finite numbers')
    if not isinstance(params['patch_size'],int) or isinstance(params['patch_size'],bool) or params['patch_size']<1 or not isinstance(params['seed'],int) or isinstance(params['seed'],bool) or params['seed']<0: raise ValueError('patch_size and seed must be valid integers')
    if params['sdd_mm']<=params['sid_mm'] or params['sid_mm']<=0 or params['pixel_size_mm']<=0 or params['attenuation_per_mm']<=0 or params['integration_step_mm']<=0 or params['photons']<=0: raise ValueError('invalid geometry or physical parameters')
    if method=='A' and params['scatter_fraction']!=0: raise ValueError('A paper did not simulate scatter')
    if params['blur_sigma_px']<0 or params['min_background']<0 or params['scatter_fraction']<0 or params['false_positive_scale']<0 or any(not 0<=params[k]<=1 for k in ('false_negative_rate','false_positive_rate')): raise ValueError('invalid augmentation parameters')
    shape=params['detector_shape']
    if not isinstance(shape,list) or len(shape)!=2 or any(not isinstance(n,int) or n<1 for n in shape): raise ValueError('invalid detector_shape')
    manifest_path=Path(manifest_path).resolve(); doc=json.loads(manifest_path.read_text(encoding='utf-8')); cases=doc.get('cases') if isinstance(doc,dict) else None
    if not isinstance(cases,list) or not cases: raise ValueError('nonempty cases required')
    ids=set(); models={}; patients={}; inputs={}; records=[]
    for row in cases:
        if not isinstance(row,dict): raise ValueError('invalid case')
        ident=row.get('id'); split=row.get('split'); model_ids=row.get('model_ids'); patient=row.get('patient_id')
        if not isinstance(ident,str) or not ident or ident in ids: raise ValueError('unique case id required')
        ids.add(ident)
        if split not in ('train','validation','test'): raise ValueError('invalid case split')
        if not isinstance(model_ids,list) or not model_ids or any(not isinstance(m,str) or not m for m in model_ids): raise ValueError('model_ids required')
        if not isinstance(patient,str) or not patient: raise ValueError('patient_id required')
        role=row.get('projection_role')
        if role not in ('patient_without_tool','measured_with_tool'): raise ValueError('projection_role must explicitly describe whether the projection contains the tool')
        if role=='measured_with_tool' and row.get('registration_confirmed') is not True: raise ValueError('measured projections require confirmed registration with truth and calibrated geometry')
        for mid in model_ids:
            if mid in models and models[mid]!=split: raise ValueError('model across splits')
            models[mid]=split
        if patient in patients and patients[patient]!=split: raise ValueError('patient across splits')
        patients[patient]=split
        def resolve(field):
            value=row.get(field)
            if not isinstance(value,str) or not value: raise ValueError(f'{field} path required')
            p=(manifest_path.parent/value).resolve()
            if p in inputs and inputs[p]!=split: raise ValueError('data file across splits')
            inputs[p]=split
            return p
        truth_path,bg_path=resolve('truth'),resolve('background')
        truth,meta,grid,background,bg_meta=_case_arrays(row,truth_path,bg_path,params)
        angles=row.get('angles_deg')
        if not isinstance(angles,list) or not all(isinstance(a,(int,float)) and np.isfinite(a) for a in angles): raise ValueError('explicit angles_deg required')
        validate_profile_angles(profile,angles)
        background_angles=bg_meta.get('angles_deg')
        if bg_meta.get('units')!='line_integral' or not isinstance(background_angles,list) or len(background_angles)!=len(background) or any(not isinstance(a,(int,float)) or not np.isfinite(a) for a in background_angles): raise ValueError('background must declare line_integral units and an ordered angle per view')
        if background.shape[1:]!=tuple(shape): raise ValueError('background shape must match detector_shape')
        if row.get('ideal'):
            _load_native_ideal(row,'ideal',manifest_path.parent,angles,params)
            if background_angles!=angles and not row.get('dte_ideal'):raise ValueError('different DTE views require native dte_ideal targets')
            if row.get('dte_ideal'):_load_native_ideal(row,'dte_ideal',manifest_path.parent,background_angles,params)
        if (background<0).any(): raise ValueError('background line integrals must be nonnegative')
        if params['patch_size']>min(shape): raise ValueError('patch_size exceeds detector')
        crop_volume_grid(truth,grid,params.get('dtr_crop_shape',list(grid.shape)))
        records.append((row,truth_path,bg_path))
    if not {'train','validation'}.issubset({row[0]['split'] for row in records}): raise ValueError('train and validation cases are both required')
    output_dir=Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())): raise ValueError('output directory must be empty')
    output_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(params['seed']); dte=[]; dtr=[]; pools=[]
    for index,(row,truth_path,bg_path) in enumerate(records):
        truth,meta,grid,background,bg_meta=_case_arrays(row,truth_path,bg_path,params)
        name=f'case_{index:05d}'; angles=row['angles_deg']
        views=[circular_view(a,params['sdd_mm'],params['sid_mm'],shape,params['pixel_size_mm']) for a in angles]
        ideal=_load_native_ideal(row,'ideal',manifest_path.parent,angles,params) if row.get('ideal') else project_tool(truth,grid,views,params,row.get('material_density_g_cm3'))
        background_angles=bg_meta['angles_deg']
        if len(background_angles)==len(angles) and np.allclose(background_angles,angles):
            dte_ideal=ideal
        else:
            dte_views=[circular_view(a,params['sdd_mm'],params['sid_mm'],shape,params['pixel_size_mm']) for a in background_angles]
            dte_ideal=_load_native_ideal(row,'dte_ideal',manifest_path.parent,background_angles,params) if row.get('ideal') else project_tool(truth,grid,dte_views,params,row.get('material_density_g_cm3'))
        cropped_truth,dtr_grid=crop_volume_grid(truth,grid,params.get('dtr_crop_shape',list(grid.shape)))
        dtr_meta={**meta,'origin_mm':list(dtr_grid.origin_mm)}
        gt_name=f'{name}_truth.npz'; save_volume(output_dir/gt_name,cropped_truth,dtr_meta)
        common={'split':row['split'],'model_ids':row['model_ids'],'patient_id':row['patient_id']}
        if export_pools and row['projection_role']=='patient_without_tool':
            bg_file=f'{name}_background_pool.npz'; tool_file=f'{name}_tool_pool.npz'
            save_projections(output_dir/bg_file,background,{'units':'line_integral','angles_deg':background_angles})
            save_projections(output_dir/tool_file,dte_ideal,{'units':'line_integral','angles_deg':background_angles})
            pools.extend([{**common,'id':name+'_background','kind':'background','path':bg_file,'projection_role':'patient_without_tool'},
                          {**common,'id':name+'_tool','kind':'tool','path':tool_file}])
        for view_index in range(len(background_angles)):
            bg,tool=_crop(background[view_index],dte_ideal[view_index],params['patch_size'],params['min_background'],rng)
            sigma=float(rng.uniform(0,params['blur_sigma_px'])) if params['blur_sigma_px'] else 0
            if row['projection_role']=='patient_without_tool':
                x,y=synthesize_dte(bg,tool,rng,photons=params['photons'],blur_sigma=sigma,scatter_fraction=params['scatter_fraction'],flip=row['split']=='train')
            else:
                x=np.array(bg,dtype=np.float32,copy=True); y=np.array(tool,dtype=np.float32,copy=True)
            x_name=f'{name}_v{view_index}_dte_input.npz'; y_name=f'{name}_v{view_index}_dte_target.npz'
            save_projections(output_dir/x_name,x,{'units':'synthetic_line_integral' if row['projection_role']=='patient_without_tool' else 'line_integral','projection_role':row['projection_role']})
            save_projections(output_dir/y_name,y,{'units':'line_integral'})
            entry={**common,'id':f'{name}_view_{view_index}','input':x_name,'target':y_name}
            if row['projection_role']=='patient_without_tool':
                bg_name=f'{name}_v{view_index}_online_background.npz'; tool_name=f'{name}_v{view_index}_online_tool.npz'
                save_projections(output_dir/bg_name,bg,{'units':'line_integral'})
                save_projections(output_dir/tool_name,tool,{'units':'line_integral'})
                entry['online_dte']={'background':bg_name,'tool':tool_name,'photons':params['photons'],'blur_sigma_px':params['blur_sigma_px'],'scatter_fraction':params['scatter_fraction']}
            dte.append(entry)
        if method=='A':
            reconstructed=fdk(ideal,views,dtr_grid,angles,angular_mode='limited')
            input_name=f'{name}_ideal_fdk.npz'; save_volume(output_dir/input_name,reconstructed,{**dtr_meta,'units':'estimated_attenuation_per_mm','source':'ideal_tool_fdk'})
            dtr.append({**common,'id':name,'input':input_name,'target':gt_name,'source':'ideal_tool_fdk','angles_deg':angles})
        else:
            if params.get('error_model','independent_pixel')=='correlated_spatial':
                from stent_repro.projection_errors import apply_correlated_projection_errors
                corrupt=apply_correlated_projection_errors(ideal,seed=int(rng.integers(2**31)),missing_rate=params['false_negative_rate'],
                    false_positive_rate=params['false_positive_rate'],false_positive_amplitude=params['false_positive_scale'],correlation_sigma_px=params['correlation_sigma_px']).projection.astype(np.float32)
            else:
                corrupt=np.stack([perturb_projection(p,rng,false_negative_rate=params['false_negative_rate'],false_positive_rate=params['false_positive_rate'],false_positive_scale=params['false_positive_scale']) for p in ideal])
            channels=backproject_each(corrupt,views,dtr_grid)
            names=[]
            for j,ch in enumerate(channels):
                input_name=f'{name}_bp{j}.npz'; save_volume(output_dir/input_name,ch,{**dtr_meta,'units':'backprojected_line_integral','source':'error_augmented_per_view_backprojection','view_angle_deg':angles[j]}); names.append(input_name)
            dtr.append({**common,'id':name,'input':names,'target':gt_name,'source':'error_augmented_per_view_backprojection','angles_deg':angles})
    for stage,rows in [('dte',dte),('dtr',dtr)]:
        (output_dir/f'{stage}.json').write_text(json.dumps({'samples':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        read_training_manifest(output_dir/f'{stage}.json',method,stage,profile=profile if stage=='dtr' else None)
    provenance={'method':method,'profile':profile,'params':params,'input_manifest':str(manifest_path),'input_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'backend':'CPU scipy midpoint trilinear / circular flat-detector FDK','spectrum':'monoenergetic constant attenuation surrogate; no material spectrum','background_calibration':'scalar flat/dark and transmission clipping when detector_intensity is supplied','fdk_filter':'absolute-frequency ramp; limited arc trapezoid weights; no Parker weighting','dte':'background-only cases: line-integral Poisson + Gaussian and optional B scatter; registered measured cases: unchanged input with simulated target','b_dtr_errors':'independent pixel suppression/insertion surrogate','strict_paper_reproduction':False,'software_versions':{'numpy':np.__version__,'scipy':scipy.__version__},'generated_manifests_sha256':{name:hashlib.sha256((output_dir/name).read_bytes()).hexdigest() for name in ('dte.json','dtr.json')}}
    if params.get('spectral') is not None:provenance['spectrum']=params['spectral']
    provenance['native_ideal_cases']=[row['id'] for row,_,_ in records if row.get('ideal')]
    provenance['b_dtr_errors']=params.get('error_model','independent_pixel')+'; project implementation, not author error distribution'
    (output_dir/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result={'dte_manifest':str(output_dir/'dte.json'),'dtr_manifest':str(output_dir/'dtr.json'),'provenance':str(output_dir/'provenance.json')}
    if export_pools and pools:
        (output_dir/'dte_pools.json').write_text(json.dumps({'pools':pools},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        result['dte_pool_manifest']=str(output_dir/'dte_pools.json')
    return result
