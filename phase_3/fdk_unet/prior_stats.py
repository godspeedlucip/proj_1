"""Equal-group training-only raw-intensity percentile fitting."""
from dataclasses import dataclass, asdict
import numpy as np
from .initialization import seed_for
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
        for i,c in enumerate(cases):
            raw=load_volume(c,require_label=False).image_raw.ravel()
            selected=np.flatnonzero(choices==i)
            values[selected]=raw[rng.integers(raw.size,size=len(selected))]
        samples.append(values)
    lo,hi=np.percentile(np.concatenate(samples),[.5,99.5],method='linear')
    return PriorStats(float(lo),float(hi),seed,'linear',n,n*len(groups),groups,json_digest(list(groups)),
                      input_digest(manifest,groups,include_labels=False))


def validate_prior_stats(stats,manifest,split):
    groups=tuple(sorted(split.train))
    if (stats.train_groups!=groups or stats.train_group_digest!=json_digest(list(groups))
            or stats.input_digest!=input_digest(manifest,groups,include_labels=False)):
        raise ValueError('Prior statistics do not match this fold/training data')
