"""Equal-group training-only raw-intensity percentile fitting."""
from dataclasses import dataclass, asdict
import numpy as np
import hashlib
import nibabel as nib

def seed_for(root_seed: int, scope: str) -> int:
    return int.from_bytes(hashlib.sha256(f"fdk-v3:{root_seed}:{scope}".encode()).digest()[:8], "big") % (2**32)
from .manifest import input_digest, json_digest
from .nifti_io import load_volume


@dataclass(frozen=True)
class PriorStats:
    lo: float
    hi: float
    seed: int
    method: str
    samples_per_group: int
    sample_count: int
    train_groups: tuple
    train_group_digest: str
    input_digest: str

    def __post_init__(self):
        if not np.isfinite([self.lo,self.hi]).all() or self.hi-self.lo<1e-6:
            raise ValueError('Degenerate prior intensity range')
        if self.method!='linear' or self.samples_per_group!=65536 or self.sample_count!=len(self.train_groups)*65536:
            raise ValueError('Invalid percentile sampling protocol')

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls,raw):
        return cls(**dict(raw,train_groups=tuple(raw['train_groups'])))


def fit_prior_stats(manifest,split,*,seed):
    rng=np.random.default_rng(seed_for(seed,'prior_stats'))
    samples=[];groups=tuple(sorted(split.train));n=65536
    for group in groups:
        cases=sorted((c for c in manifest.cases if c.group_id==group),key=lambda c:c.case_id)
        if not cases:raise ValueError('Empty training group')
        choices=rng.integers(len(cases),size=n);values=np.empty(n,np.float32)
        selected_by_image={}
        for i,c in enumerate(cases):
            selected=np.flatnonzero(choices==i)
            stat=c.image.stat();identity=(stat.st_dev,stat.st_ino)
            if identity not in selected_by_image:
                selected_by_image[identity]=[c,int(np.prod(nib.load(str(c.image)).shape)),[]]
            voxel_count=selected_by_image[identity][1]
            offsets=rng.integers(voxel_count,size=len(selected))
            selected_by_image[identity][2].append((selected,offsets))
        for c,_,selections in selected_by_image.values():
            raw=load_volume(c,require_label=False).image_raw.ravel()
            for selected,offsets in selections:
                values[selected]=raw[offsets]
        samples.append(values)
    lo,hi=np.percentile(np.concatenate(samples),[.5,99.5],method='linear')
    return PriorStats(float(lo),float(hi),seed,'linear',n,n*len(groups),groups,json_digest(list(groups)),
                      input_digest(manifest,groups,include_labels=False))


def validate_prior_stats(stats,manifest,split):
    groups=tuple(sorted(split.train))
    if (stats.train_groups!=groups or stats.train_group_digest!=json_digest(list(groups))
            or stats.input_digest!=input_digest(manifest,groups,include_labels=False)):
        raise ValueError('Prior statistics do not match this fold/training data')
