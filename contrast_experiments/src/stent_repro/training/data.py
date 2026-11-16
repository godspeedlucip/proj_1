"""Prepared supervised samples. No projection/reconstruction is inferred here."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from stent_repro.io import load_projections, load_volume
from stent_repro.configuration import validate_profile_angles


def read_training_manifest(path, method, stage, profile=None):
    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding='utf-8'))
    rows = doc.get('samples') if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows: raise ValueError('manifest samples must be a nonempty list')
    if method not in ('A','B') or stage not in ('dte','dtr'): raise ValueError('invalid method or stage')
    ids, models, patients, files, result = set(), {}, {}, {}, []
    for raw in rows:
        if not isinstance(raw,dict): raise ValueError('invalid sample')
        row = dict(raw)
        if not isinstance(row.get('id'),str) or not row['id'] or row['id'] in ids: raise ValueError('missing or duplicate sample id')
        ids.add(row['id'])
        if row.get('split') not in ('train','validation','test'): raise ValueError('invalid split')
        if not isinstance(row.get('model_ids'),list) or not row['model_ids'] or any(not isinstance(m,str) or not m for m in row['model_ids']): raise ValueError('model_ids required for leakage checks')
        for model in row['model_ids']:
            if model in models and models[model] != row['split']: raise ValueError('model appears in multiple splits')
            models[model] = row['split']
        patient = row.get('patient_id')
        if patient is not None:
            if not isinstance(patient,str) or not patient: raise ValueError('invalid patient_id')
            if patient in patients and patients[patient] != row['split']: raise ValueError('patient appears in multiple splits')
            patients[patient] = row['split']
        if stage == 'dtr':
            source = 'ideal_tool_fdk' if method == 'A' else 'error_augmented_per_view_backprojection'
            if row.get('source') != source: raise ValueError('wrong DTR source')
            a = row.get('angles_deg')
            if not isinstance(a,list) or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in a): raise ValueError('DTR angles_deg required')
            if profile == 'A-original-4view':
                if len(a)!=4 or not np.allclose(np.mod(np.diff(a),360),45): raise ValueError('original A angles must be separated by 45 degrees')
            elif profile == 'B-original-3view' or profile == 'B-one-channel-ablation':
                if len(a)!=3 or not np.allclose(np.mod(np.diff(a),360),60): raise ValueError('original B angles must be separated by 60 degrees')
            elif profile in ('A-adapted-3view','B-adapted-3view','B-adapted-one-channel'):
                if len(a)!=3 or not np.allclose(a,[0,30,60]): raise ValueError('adapted angles must match the configured profile')
            elif profile and profile.startswith('A-reference-') and method=='A':validate_profile_angles(profile,a)
            else: raise ValueError('DTR profile required')
        paths = row.get('input')
        if method == 'B' and stage == 'dtr':
            if not isinstance(paths,list) or len(paths)!=3 or any(not isinstance(p,str) or not p for p in paths): raise ValueError('B DTR requires three ordered inputs')
            row['input'] = [str((path.parent/p).resolve()) for p in paths]
        else:
            if not isinstance(paths,str) or not paths: raise ValueError('input path required')
            row['input'] = str((path.parent/paths).resolve())
        if not isinstance(row.get('target'),str) or not row['target']: raise ValueError('target path required')
        row['target'] = str((path.parent/row['target']).resolve())
        input_paths = row['input'] if isinstance(row['input'],list) else [row['input']]
        if stage=='dte' and 'online_dte' in row:
            online=dict(row['online_dte'])
            for key in ('background','tool'):
                if not isinstance(online.get(key),str) or not online[key]: raise ValueError('online DTE source path required')
                online[key]=str((path.parent/online[key]).resolve()); input_paths.append(online[key])
            row['online_dte']=online
        for filename in input_paths + [row['target']]:
            if filename in files and files[filename] != row['split']: raise ValueError('file appears in multiple splits')
            files[filename] = row['split']
        result.append(row)
    return result


def _one_image(path):
    data, _ = load_projections(path)
    if len(data) != 1: raise ValueError('each DTE sample must contain one projection')
    return data[0]


def _volume(path):
    return load_volume(path)[0]

def _check_grids(metadata):
    for key in ('spacing_mm','origin_mm','direction'):
        present = [key in m for m in metadata]
        if any(present):
            if not all(present) or any(not np.allclose(m[key],metadata[0][key]) for m in metadata[1:]):
                raise ValueError(f'physical grid mismatch or missing {key}')


class PreparedDataset(Dataset):
    def __init__(self, rows, split, method, stage, slice_channels=5, input_channels=3, augment=False, online_dte=False, patch_shape=None, random_crop=False):
        if method not in ('A','B') or stage not in ('dte','dtr'): raise ValueError('invalid method/stage')
        if slice_channels not in (1,3,5) or input_channels not in (1,3): raise ValueError('unsupported channel count')
        if augment and (split != 'train' or method != 'B' or stage != 'dtr'):
            raise ValueError('augmentation is supported only for B DTR training')
        if online_dte and (split!='train' or stage!='dte'): raise ValueError('online DTE augmentation is restricted to the training split')
        self.augment = augment
        self.online_dte = online_dte
        self.patch_shape = patch_shape
        self.random_crop = random_crop
        if random_crop and split != 'train': raise ValueError('random crops require training split')
        self.rows = [r for r in rows if r['split'] == split]
        self.method, self.stage, self.slice_channels, self.input_channels = method, stage, slice_channels, input_channels
        self.indices = []
        for i,row in enumerate(self.rows):
            if method=='A' and stage=='dtr':
                depth = _volume(row['target']).shape[0]
                self.indices.extend((i,z) for z in range(depth))
            else: self.indices.append((i,None))
        if not self.indices: raise ValueError(f'{split} dataset is empty')
        if online_dte and not any('online_dte' in row for row in self.rows): raise ValueError('online DTE source data is unavailable')

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        i,z = self.indices[index]; row = self.rows[i]
        if self.stage == 'dte':
            if self.online_dte and 'online_dte' in row:
                from stent_repro.preprocessing import synthesize_dte
                spec=row['online_dte']; seed=int(torch.randint(0,2**32,(1,),dtype=torch.int64).item()); rng=np.random.default_rng(seed)
                sigma=float(rng.uniform(0,spec['blur_sigma_px'])) if spec['blur_sigma_px'] else 0
                x,y=synthesize_dte(_one_image(spec['background']),_one_image(spec['tool']),rng,photons=spec['photons'],blur_sigma=sigma,scatter_fraction=spec['scatter_fraction'],flip=True)
            else:
                x,y = _one_image(row['input']),_one_image(row['target'])
            if x.shape!=y.shape or np.any(y<0): raise ValueError('DTE input and continuous target must match; target must be nonnegative')
            x,y = x[None], y[None]
        else:
            y, target_meta = load_volume(row['target'])
            if not np.all((y==0)|(y==1)): raise ValueError('DTR target must be binary')
            if self.method=='A':
                x, input_meta = load_volume(row['input'])
                _check_grids([target_meta,input_meta])
                if x.shape!=y.shape: raise ValueError('A FDK and target shapes differ')
                x=x.astype(np.float32)
                x=(x-x.mean()) / max(float(x.std()),1e-6)
                ix = np.clip(np.arange(z-self.slice_channels//2,z+self.slice_channels//2+1),0,len(x)-1)
                x,y=x[ix],y[z][None]
            else:
                loaded = [load_volume(p) for p in row['input']]
                channels = [item[0] for item in loaded]
                declared = [item[1].get('view_angle_deg') for item in loaded]
                if any(a is not None for a in declared):
                    if any(a is None for a in declared) or 'angles_deg' not in row or len(row['angles_deg']) != 3 or not np.allclose(declared,row['angles_deg']):
                        raise ValueError('backprojection channel angle mismatch')
                _check_grids([target_meta] + [item[1] for item in loaded])
                if any(a.shape != y.shape for a in channels): raise ValueError('B channels and target shapes differ')
                x = np.stack(channels)
                if self.augment:
                    turns = int(torch.randint(4, ()).item())
                    flip = bool(torch.randint(2, ()).item())
                    x = np.rot90(x,turns,axes=(-2,-1)); y = np.rot90(y,turns,axes=(-2,-1))
                    if flip: x = np.flip(x,axis=1); y = np.flip(y,axis=0)
                if self.input_channels == 1: x = x.sum(axis=0,keepdims=True)
                y=y[None]
        if self.patch_shape is not None:
            if len(self.patch_shape)!=x.ndim-1 or any(n>m or n<1 for n,m in zip(self.patch_shape,x.shape[1:])):
                raise ValueError('patch_shape exceeds input or has wrong dimensionality')
            starts=[int(torch.randint(m-n+1,()).item()) if self.random_crop else (m-n)//2 for n,m in zip(self.patch_shape,x.shape[1:])]
            slices=(slice(None),*[slice(start,start+n) for start,n in zip(starts,self.patch_shape)])
            x,y=x[slices],y[slices]
        return torch.from_numpy(np.array(x,dtype=np.float32,copy=True)), torch.from_numpy(np.array(y,dtype=np.float32,copy=True))
