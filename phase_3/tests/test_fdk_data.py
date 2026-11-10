import copy
import numpy as np
import torch
from phase_3.tests.fdk_fixtures import synthetic_dataset


def test_dual_branch_crop_flip_and_rng_are_shared(tmp_path):
    from phase_3.fdk_unet.manifest import load_manifest,load_splits
    from phase_3.fdk_unet.config import get_experiment_config,resolve_config
    from phase_3.fdk_unet.prior_stats import fit_prior_stats
    from phase_3.fdk_unet.data import PatchStream
    mp,sp=synthetic_dataset(tmp_path,shape=(37,45,27));m=load_manifest(mp,require_labels=True);split=load_splits(sp)[0]
    stats=fit_prior_stats(m,split,seed=2026)
    def config(name):
        r=get_experiment_config(name).resolved;r.update(run_kind='smoke',experiment_id=name+'-smoke',patch_size=[32]*3,max_optimizer_steps=4)
        return resolve_config(r)
    a=PatchStream(m,split,config('A'),None);b=PatchStream(m,split,config('C'),stats)
    for _ in range(4):
        ba,bb=a.next_batch(),b.next_batch()
        assert ba.case_ids==bb.case_ids and ba.metadata==bb.metadata
        for key in ('image_unet','target','valid_mask'):torch.testing.assert_close(getattr(ba,key),getattr(bb,key),atol=0,rtol=0)
        # Reconstruct saved crop + flip from full-volume normalized arrays independently.
        for i,(case_id,meta) in enumerate(zip(bb.case_ids,bb.metadata)):
            full=b.volumes[case_id]['image_swin']
            padded=np.pad(full,meta['padding'])
            crop=padded[tuple(slice(s,s+32) for s in meta['crop_start'])]
            for axis,flip in enumerate(meta['flips']):
                if flip:crop=np.flip(crop,axis)
            np.testing.assert_array_equal(bb.image_swin[i,0].numpy(),crop)
    state=copy.deepcopy(b.state_dict());expected=b.next_batch();b.load_state_dict(state);actual=b.next_batch()
    assert actual.metadata==expected.metadata and actual.case_ids==expected.case_ids
    torch.testing.assert_close(actual.image_swin,expected.image_swin,atol=0,rtol=0)
