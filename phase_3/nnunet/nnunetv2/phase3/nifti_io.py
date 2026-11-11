"""NIfTI storage ijk <-> tensor DHW, preserving the original physical grid."""
from dataclasses import dataclass
from pathlib import Path
import re
import nibabel as nib
import numpy as np


@dataclass
class SpatialMeta:
    original_shape: tuple
    affine: np.ndarray
    header: object
    unit: str
    affine_mm: np.ndarray


@dataclass
class VolumeCase:
    case_id: str
    group_id: str
    image_raw: np.ndarray
    target: np.ndarray | None
    meta: SpatialMeta


def read_nifti(path):
    nii = nib.load(str(path))
    if len(nii.shape) != 3 or any(s == 0 for s in nii.shape):
        raise ValueError(f'Expected a nonempty 3D NIfTI: {path}')
    unit = nii.header.get_xyzt_units()[0]
    factors = {'mm':1.,'meter':1000.,'micron':.001}
    if unit not in factors: raise ValueError(f'Unknown spatial units: {path}')
    affine = np.asarray(nii.affine,dtype=np.float64)
    affine_mm = affine.copy(); affine_mm[:3] *= factors[unit]
    if not np.isfinite(affine_mm).all() or abs(np.linalg.det(affine_mm[:3,:3])) < 1e-12:
        raise ValueError('Invalid affine')
    q,qcode=nii.get_qform(coded=True); s,scode=nii.get_sform(coded=True)
    if qcode and scode and not np.allclose(q[:3]*factors[unit],s[:3]*factors[unit],atol=1e-4,rtol=0):
        raise ValueError('Conflicting qform and sform')
    spacing = np.linalg.norm(affine_mm[:3,:3],axis=0)
    if not np.allclose(spacing,.2,atol=1e-4,rtol=0):
        raise ValueError(f'Expected 0.2 mm spacing, got {spacing}')
    array = np.asarray(nii.get_fdata(dtype=np.float32))
    if not np.isfinite(array).all(): raise ValueError('Nonfinite NIfTI data')
    return np.ascontiguousarray(array.transpose(2,1,0)), SpatialMeta(tuple(nii.shape),affine,nii.header.copy(),unit,affine_mm)


def same_grid(left, right):
    return left.original_shape == right.original_shape and np.allclose(left.affine_mm,right.affine_mm,atol=1e-4,rtol=0)


def load_volume(record, *, require_label):
    image, meta = read_nifti(record.image)
    target = None
    if require_label:
        if record.label is None: raise ValueError(f'Missing label: {record.case_id}')
        target, target_meta = read_nifti(record.label)
        if not same_grid(meta,target_meta): raise ValueError('Image and label physical grids differ')
        if not np.isin(target,[0,1]).all(): raise ValueError('Labels must be exactly 0 or 1')
    return VolumeCase(record.case_id,record.group_id,image,target,meta)


def save_prediction(probability, mask, meta, *, case_id, output):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',case_id): raise ValueError('Unsafe case_id')
    expected = tuple(reversed(meta.original_shape))
    if probability.shape != expected or mask.shape != expected:
        raise ValueError('Prediction shape differs from original grid')
    if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)) or not np.isin(mask,[0,1]).all():
        raise ValueError('Invalid probability or binary mask')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix,array,dtype in (('prob',probability,np.float32),('mask',mask,np.uint8)):
        header=meta.header.copy();header.set_data_dtype(dtype);header.set_slope_inter(1.,0.)
        nii=nib.Nifti1Image(np.asarray(array.transpose(2,1,0),dtype=dtype),meta.affine,header)
        nii.header.set_slope_inter(1.,0.)
        path=output/f'{case_id}_{suffix}.nii.gz';nib.save(nii,str(path));paths.append(path)
    return tuple(paths)
