import dataclasses
import numpy as np
import nibabel as nib
import pytest
from phase_3.tests.fdk_fixtures import synthetic_dataset


@pytest.mark.parametrize('unit,scale',[('mm',1.),('meter',.001)])
def test_nifti_asymmetric_roundtrip_and_units(tmp_path,unit,scale):
    from phase_3.fdk_unet.manifest import load_manifest
    from phase_3.fdk_unet.nifti_io import load_volume,save_prediction
    mp,_=synthetic_dataset(tmp_path,shape=(33,35,37))
    r=load_manifest(mp,require_labels=True).cases[0]
    affine=np.array([[0,-.2,0,10],[.2,0,0,-4],[0,0,.2,7],[0,0,0,1.]])
    stored=affine.copy(); stored[:3]*=scale
    data=np.zeros((33,35,37),np.float32);data[3,7,11]=19
    for path,array in ((r.image,data),(r.label,(data>0).astype(np.uint8))):
        nii=nib.Nifti1Image(array,stored);nii.header.set_xyzt_units(unit);nib.save(nii,path)
    v=load_volume(r,require_label=True)
    assert v.image_raw.shape==(37,35,33) and v.image_raw[11,7,3]==19
    np.testing.assert_allclose(v.meta.affine_mm,affine,atol=1e-6)
    p,m=save_prediction(v.target.astype(np.float32),v.target.astype(np.uint8),v.meta,case_id='out',output=tmp_path/'out')
    out=nib.load(m)
    np.testing.assert_array_equal(out.get_fdata(),data>0)
    np.testing.assert_allclose(out.affine,stored,atol=1e-6)
    assert out.header.get_xyzt_units()[0]==unit
    assert nib.load(p).get_data_dtype()==np.float32 and out.get_data_dtype()==np.uint8
    v=load_volume(dataclasses.replace(r,label=None),require_label=False)
    assert v.target is None


@pytest.mark.parametrize('damage',['label2','nan','affine','spacing','unit','qform'])
def test_invalid_nifti_rejected(tmp_path,damage):
    from phase_3.fdk_unet.manifest import load_manifest
    from phase_3.fdk_unet.nifti_io import load_volume
    mp,_=synthetic_dataset(tmp_path)
    r=load_manifest(mp,require_labels=True).cases[0]
    path=r.label if damage in ('label2','affine') else r.image
    nii=nib.load(path);a=nii.get_fdata();aff=nii.affine.copy();header=nii.header.copy()
    if damage=='label2':a[0,0,0]=2
    if damage=='nan':a[0,0,0]=np.nan
    if damage=='affine':aff[0,3]+=1
    if damage=='spacing':aff[0,0]=.3
    if damage=='unit':header.set_xyzt_units('unknown')
    out=nib.Nifti1Image(a,aff,header)
    if damage=='qform':
        other=aff.copy();other[0,3]+=2;out.set_qform(other,1);out.set_sform(aff,1)
    nib.save(out,path)
    with pytest.raises(ValueError):load_volume(r,require_label=True)
