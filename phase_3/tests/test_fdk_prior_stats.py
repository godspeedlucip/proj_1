from dataclasses import replace
import numpy as np
import pytest
from phase_3.tests.fdk_fixtures import synthetic_dataset


def test_percentiles_never_read_validation_test_or_labels(tmp_path,monkeypatch):
    from phase_3.fdk_unet.manifest import load_manifest,load_splits
    from phase_3.fdk_unet.prior_stats import fit_prior_stats
    import nibabel as nib
    mp,sp=synthetic_dataset(tmp_path);m=load_manifest(mp,require_labels=True);split=load_splits(sp)[0]
    allowed={str(c.image) for c in m.cases if c.group_id in split.train};old=nib.load
    def guarded(p,*a,**k):
        assert str(p) in allowed
        return old(p,*a,**k)
    monkeypatch.setattr(nib,'load',guarded)
    a=fit_prior_stats(m,split,seed=2026);b=fit_prior_stats(m,split,seed=2026)
    assert a==b and a.samples_per_group==65536 and a.sample_count==131072
    assert a.hi>a.lo and a.method=='linear'


def test_constant_and_nonfinite_normalization(tmp_path):
    from phase_3.fdk_unet.manifest import load_manifest
    from phase_3.fdk_unet.nifti_io import load_volume
    from phase_3.fdk_unet.data import normalize_volume
    mp,_=synthetic_dataset(tmp_path);v=load_volume(load_manifest(mp,require_labels=True).cases[0],require_label=True)
    v.image_raw[:]=4
    assert not normalize_volume(v,None,swin_mode='none')['image_unet'].any()
    v.image_raw[0,0,0]=np.nan
    with pytest.raises(ValueError):normalize_volume(v,None,swin_mode='none')
