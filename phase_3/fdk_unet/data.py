"""One synchronized spatial stream for CNN, Swin, labels and validity."""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from .initialization import seed_for
from .manifest import json_digest
from .nifti_io import load_volume


@dataclass
class SampleBatch:
    image_unet: torch.Tensor
    image_swin: torch.Tensor | None
    target: torch.Tensor
    valid_mask: torch.Tensor
    case_ids: tuple
    metadata: list

    def to(self,device):
        return SampleBatch(self.image_unet.to(device),self.image_swin.to(device) if self.image_swin is not None else None,
                           self.target.to(device),self.valid_mask.to(device),self.case_ids,self.metadata)


def normalize_volume(volume,stats,*,swin_mode):
    x=volume.image_raw
    if not np.isfinite(x).all():raise ValueError('Nonfinite image')
    mean=x.mean(dtype=np.float64);std=x.std(dtype=np.float64,ddof=0)
    result={'image_unet':((x.astype(np.float64)-mean)/max(std,1e-6)).astype(np.float32)}
    if swin_mode=='train_percentile':
        if stats is None:raise ValueError('Saved training percentile statistics required')
        result['image_swin']=np.clip((x-stats.lo)/(stats.hi-stats.lo),0,1).astype(np.float32)
    elif swin_mode=='zscore':result['image_swin']=result['image_unet'].copy()
    elif swin_mode!='none':raise ValueError('Unknown Swin input mode')
    if volume.target is not None:result['target']=volume.target.astype(np.float32)
    result['valid_mask']=np.ones(x.shape,np.float32)
    return result


def padding_for(shape,patch):
    return tuple((max(p-s,0)//2,max(p-s,0)-max(p-s,0)//2) for s,p in zip(shape,patch))


class PatchStream:
    def __init__(self,manifest,split,config,stats):
        self.config=config
        self.groups=tuple(sorted(split.train))
        self.cases={g:sorted((c.case_id for c in manifest.cases if c.group_id==g)) for g in self.groups}
        if any(not ids for ids in self.cases.values()):raise ValueError('Empty training group')
        self.volumes={}
        for record in manifest.cases:
            if record.group_id in self.groups:
                volume=load_volume(record,require_label=True)
                mode=config.resolved['swin_input']['mode'] if config.resolved['prior'] is not None else 'none'
                self.volumes[record.case_id]=normalize_volume(volume,stats,swin_mode=mode)
        self.sampling=np.random.default_rng(seed_for(config.seed,'sampling'))
        self.augmentation=np.random.default_rng(seed_for(config.seed,'augmentation'))
        self.identity=json_digest({'groups':self.groups,'cases':self.cases,'patch':config.patch_size})

    def state_dict(self):
        return copy.deepcopy({'identity':self.identity,'sampling':self.sampling.bit_generator.state,
                              'augmentation':self.augmentation.bit_generator.state})

    def load_state_dict(self,state):
        if state['identity']!=self.identity:raise ValueError('Sampler data or patch mismatch')
        self.sampling.bit_generator.state=copy.deepcopy(state['sampling'])
        self.augmentation.bit_generator.state=copy.deepcopy(state['augmentation'])

    def next_batch(self):
        arrays=[];ids=[];metadata=[];patch=self.config.patch_size
        for _ in range(self.config.batch_size):
            group=self.groups[int(self.sampling.integers(len(self.groups)))]
            case_id=self.cases[group][int(self.sampling.integers(len(self.cases[group])))]
            full=self.volumes[case_id];padding=padding_for(full['image_unet'].shape,patch)
            padded={k:np.pad(v,padding) for k,v in full.items()}
            shape=padded['image_unet'].shape
            center=None
            if self.sampling.random()<self.config.resolved['foreground_probability'] and any(s>p for s,p in zip(shape,patch)):
                foreground=np.argwhere(padded['target']==1)
                if len(foreground):center=foreground[int(self.sampling.integers(len(foreground)))]
            start=tuple(int(np.clip(c-p//2,0,s-p)) if center is not None else int(self.sampling.integers(s-p+1))
                        for c,s,p in zip(center if center is not None else (0,0,0),shape,patch))
            flips=tuple(bool(v) for v in (self.augmentation.random(3)<.5)) if self.config.resolved['augment'] else (False,False,False)
            slices=tuple(slice(a,a+p) for a,p in zip(start,patch));sample={}
            for key,value in padded.items():
                crop=value[slices]
                for axis,flip in enumerate(flips):
                    if flip:crop=np.flip(crop,axis)
                sample[key]=np.ascontiguousarray(crop)
            arrays.append(sample);ids.append(case_id)
            metadata.append({'crop_start':start,'flips':flips,'padding':padding})
        def tensor(key):return torch.from_numpy(np.stack([a[key] for a in arrays])[:,None])
        return SampleBatch(tensor('image_unet'),tensor('image_swin') if 'image_swin' in arrays[0] else None,
                           tensor('target'),tensor('valid_mask'),tuple(ids),metadata)
