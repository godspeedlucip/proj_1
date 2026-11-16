"""Replayable independent sample streams and training-only running moments."""
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class RunningStatistics:
    def __init__(self,state=None):
        self.count=0;self.mean=0.;self.m2=0.
        if state:
            self.count=int(state['count']);self.mean=float(state['mean'])
            self.m2=float(state.get('m2',state['std']**2*self.count))

    def update(self,x):
        x=x.detach().double();n=x.numel()
        if not n or not torch.isfinite(x).all():raise ValueError('finite nonempty normalization input required')
        mean=x.mean().item();delta=mean-self.mean;total=self.count+n
        self.m2+=((x-mean)**2).sum().item()+delta**2*self.count*n/total
        self.mean+=delta*n/total;self.count=total

    def state_dict(self):
        return {'count':self.count,'mean':self.mean,'m2':self.m2,'std':max((self.m2/max(self.count,1))**.5,1e-6)}

    def normalize(self,x):
        if not self.count:raise ValueError('running normalization has no training observations')
        return (x-self.mean)/self.state_dict()['std']


class EpochDataset(Dataset):
    def __init__(self,dataset,count,seed,epoch,stream):
        if type(count) is not int or count<1 or len(dataset)<1:raise ValueError('positive sample count and nonempty dataset required')
        self.dataset=dataset;self.count=count;self.seed=seed;self.epoch=epoch;self.stream=stream

    def __len__(self):return self.count

    def __getitem__(self,index):
        if index<0 or index>=self.count:raise IndexError(index)
        digest=hashlib.sha256(f'{self.seed}/{self.epoch}/{self.stream}/{index}'.encode()).digest()
        seed=int.from_bytes(digest[:4],'little');rng=np.random.default_rng(seed)
        item=int(rng.integers(len(self.dataset)))
        py_state=random.getstate();np_state=np.random.get_state();torch_state=torch.get_rng_state()
        try:
            random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
            return self.dataset[item]
        finally:
            random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(torch_state)


def numpy_rng_state():
    name,keys,pos,has_gauss,cached=np.random.get_state()
    return {'name':name,'keys':keys.tolist(),'pos':pos,'has_gauss':has_gauss,'cached':cached}


def restore_numpy_rng(state):
    np.random.set_state((state['name'],np.array(state['keys'],dtype=np.uint32),state['pos'],state['has_gauss'],state['cached']))
