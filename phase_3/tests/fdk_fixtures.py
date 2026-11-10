"""Synthetic fixtures only: these are never evidence of official SSL compatibility."""
import torch
from torch import nn
from torch.nn import functional as F
import json
from pathlib import Path
import numpy as np
import nibabel as nib


def synthetic_dataset(root, shape=(32,32,32), count=4):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    cases = []
    for i in range(count):
        rng = np.random.default_rng(100+i)
        label = np.zeros(shape,np.uint8); label[5:9,6:23,10:14] = 1
        image = rng.normal(0,.02,shape).astype(np.float32) + label * (1+i*.1)
        affine = np.diag([.2,.2,.2,1.])
        for name,array in ((f'case_{i}_0000',image),(f'case_{i}',label)):
            nii=nib.Nifti1Image(array,affine); nii.header.set_xyzt_units('mm')
            nib.save(nii,root/(name+'.nii.gz'))
        cases.append(dict(case_id=f'case_{i}',group_id=f'g{i}',source_model_ids=[f's{i}'],patient_id=None,
                          image=f'case_{i}_0000.nii.gz',label=f'case_{i}.nii.gz',input_kind='fdk_3view',
                          angles_deg=[0.,30.,60.],projection_source='ideal_mask',upstream_run_id=None))
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version='fdk_stent_manifest_v1',target_semantics='stent_struts',
                                        cases=cases,upstream_registry={},group_provenance={f'g{i}':'synthetic fixture' for i in range(count)})))
    splits = root/'splits.json'
    splits.write_text(json.dumps(dict(schema_version='fdk_stent_splits_v1',folds={
        '0':dict(train=['g0','g1'],validation=['g2'],test=['g3'])})))
    return manifest,splits


class FakePrior(nn.Module):
    def __init__(self, execute_through='layers3'):
        super().__init__()
        self.execute_through = execute_through
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, image):
        return {i: F.avg_pool3d(image, 2**(i+1)).repeat(1, c, 1, 1, 1) * self.scale
                for i, c in ((1,96),(2,192),(3,384))
                if i <= int(self.execute_through[-1])}


def make_test_hybrid(experiment_id, plain_state):
    from phase_3.fdk_unet.config import get_experiment_config
    from phase_3.fdk_unet.plain_unet import PlainUNet3D
    from phase_3.fdk_unet.hybrid_model import SwinPriorUNet3D
    cfg = get_experiment_config(experiment_id)
    unet = PlainUNet3D()
    unet.load_state_dict(plain_state)
    return SwinPriorUNet3D(unet, FakePrior(cfg.resolved['prior']['execute_through']),
                          fusion_config=cfg.resolved['fusion']).eval()
