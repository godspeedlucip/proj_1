import dataclasses
import numpy as np
import pytest
import torch
from torch import nn
from phase_3.tests.fdk_fixtures import synthetic_dataset


class Constant(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return torch.zeros_like(x)+1.2


def test_mixed_shape_windows_cover_every_voxel_and_restore_grid(tmp_path):
    from phase_3.fdk_unet.config import get_experiment_config
    from phase_3.fdk_unet.manifest import load_manifest
    from phase_3.fdk_unet.nifti_io import load_volume
    from phase_3.fdk_unet.predict import predict_volume,window_starts
    mp,_=synthetic_dataset(tmp_path,shape=(173,145,97))
    v=load_volume(load_manifest(mp,require_labels=False).cases[0],require_label=False)
    model=Constant();prediction=predict_volume(model,v,get_experiment_config('A'),None,device=torch.device('cpu'))
    assert window_starts(173,128,64)==[0,45]
    assert prediction.probability.shape==(97,145,173) and model.calls==4
    assert prediction.probability.dtype==np.float32 and prediction.mask.dtype==np.uint8
    assert prediction.telemetry['minimum_weight']>0
    np.testing.assert_allclose(prediction.probability,torch.sigmoid(torch.tensor(1.2)).item(),atol=1e-6)
    with pytest.raises(ValueError):window_starts(200,32,40)


def test_overlap_averages_probabilities_not_logits(tmp_path):
    from phase_3.fdk_unet.config import get_experiment_config,resolve_config
    from phase_3.fdk_unet.manifest import load_manifest
    from phase_3.fdk_unet.nifti_io import load_volume
    from phase_3.fdk_unet.predict import predict_volume
    mp,_=synthetic_dataset(tmp_path,shape=(48,32,32));v=load_volume(load_manifest(mp,require_labels=False).cases[0],require_label=False)
    r=get_experiment_config('A').resolved;r.update(run_kind='smoke',experiment_id='A-smoke',patch_size=[32]*3,max_optimizer_steps=2)
    class Alternating(nn.Module):
        def __init__(self):super().__init__();self.i=0
        def forward(self,x):self.i+=1;return torch.zeros_like(x)+(0 if self.i==1 else 2)
    p=predict_volume(Alternating(),v,resolve_config(r),None,device=torch.device('cpu'))
    expected=(.5+torch.sigmoid(torch.tensor(2.)).item())/2
    np.testing.assert_allclose(p.probability[:,:,16:32],expected,atol=1e-6)


def test_predict_uses_saved_whole_volume_stats_without_label(tmp_path):
    from phase_3.fdk_unet.config import get_experiment_config,resolve_config
    from phase_3.fdk_unet.manifest import load_manifest,load_splits
    from phase_3.fdk_unet.nifti_io import load_volume
    from phase_3.fdk_unet.prior_stats import fit_prior_stats
    from phase_3.fdk_unet.predict import predict_volume
    mp,sp=synthetic_dataset(tmp_path,shape=(48,32,32));m=load_manifest(mp,require_labels=True)
    stats=fit_prior_stats(m,load_splits(sp)[0],seed=2026);v=load_volume(dataclasses.replace(m.cases[0],label=None),require_label=False)
    r=get_experiment_config('C').resolved;r.update(run_kind='smoke',experiment_id='C-smoke',patch_size=[32]*3,max_optimizer_steps=2)
    seen=[]
    class Dual(nn.Module):
        def forward(self,x,xs):seen.append((x.cpu().numpy(),xs.cpu().numpy()));return torch.zeros_like(x)
    predict_volume(Dual(),v,resolve_config(r),stats,device=torch.device('cpu'))
    full=(v.image_raw-v.image_raw.mean(dtype=np.float64))/v.image_raw.std(dtype=np.float64)
    np.testing.assert_allclose(seen[0][0][0,0],full[:,:,:32],atol=1e-6)
    np.testing.assert_array_equal(seen[0][1][0,0,:,:,16:32],seen[1][1][0,0,:,:,:16])
