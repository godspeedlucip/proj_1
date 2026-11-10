"""Whole-volume normalization, synchronized windows, mean probabilities."""
from dataclasses import dataclass
from itertools import product
import time
import numpy as np
import torch
from .data import normalize_volume,padding_for
from .nifti_io import SpatialMeta


@dataclass
class Prediction:
    probability: np.ndarray
    mask: np.ndarray
    meta: SpatialMeta
    telemetry: dict


def validate_device(device):
    device=torch.device(device)
    if device.type not in ('cpu','cuda'):raise ValueError('Only CPU and CUDA devices are supported')
    if device.type=='cuda' and (not torch.cuda.is_available() or (device.index is not None and device.index>=torch.cuda.device_count())):
        raise ValueError('Requested CUDA device is unavailable')
    return device


def window_starts(length,patch,stride):
    if min(length,patch,stride)<1 or stride>patch:raise ValueError('Invalid window dimensions/stride')
    if length<=patch:return [0]
    return sorted(set([*range(0,length-patch+1,stride),length-patch]))


def predict_volume(model,volume,config,stats,*,device):
    device=validate_device(device);model=model.to(device).eval()
    if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter();patch=config.patch_size
    stride=tuple(p//2 for p in patch) if config.resolved['run_kind']=='smoke' else tuple(config.resolved['inference_stride'])
    mode=config.resolved['swin_input']['mode'] if config.resolved['prior'] is not None else 'none'
    full=normalize_volume(volume,stats,swin_mode=mode)
    padding=padding_for(volume.image_raw.shape,patch)
    arrays={k:np.pad(v,padding) for k,v in full.items() if k in ('image_unet','image_swin')}
    shape=arrays['image_unet'].shape;sums=np.zeros(shape,np.float32);weights=np.zeros(shape,np.float32)
    count=0
    with torch.inference_mode():
        for origin in product(*(window_starts(s,p,t) for s,p,t in zip(shape,patch,stride))):
            slices=tuple(slice(a,a+p) for a,p in zip(origin,patch))
            inputs=[torch.from_numpy(np.ascontiguousarray(arrays[k][slices]))[None,None].to(device)
                    for k in ('image_unet','image_swin') if k in arrays]
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(*inputs)
            if logits.shape!=inputs[0].shape or not torch.isfinite(logits).all():raise ValueError('Invalid prediction logits')
            probability=logits.float().sigmoid()[0,0].cpu().numpy()
            sums[slices]+=probability;weights[slices]+=1;count+=1
    if np.any(weights<=0):raise RuntimeError('Sliding windows left uncovered voxels')
    slices=tuple(slice(p[0],p[0]+s) for p,s in zip(padding,volume.image_raw.shape))
    probability=np.ascontiguousarray((sums/weights)[slices],dtype=np.float32)
    mask=(probability>=config.resolved['threshold']).astype(np.uint8)
    if device.type=='cuda':torch.cuda.synchronize(device)
    telemetry={'patch_size':list(patch),'stride':list(stride),'original_shape_dhw':list(volume.image_raw.shape),
               'windows':count,'minimum_weight':float(weights[slices].min()),'threshold':config.resolved['threshold'],
               'seconds':time.perf_counter()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
               'amp_dtype':'float16' if device.type=='cuda' else None}
    if stats is not None:
        telemetry.update(clipped_low_fraction=float((volume.image_raw<stats.lo).mean()),
                         clipped_high_fraction=float((volume.image_raw>stats.hi).mean()))
    return Prediction(probability,mask,volume.meta,telemetry)
