"""Independent patient/background and tool pools for online DTE synthesis."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from stent_repro.io import load_projections
from stent_repro.preprocessing import synthesize_dte


class OnlineDTEDataset(Dataset):
    def __init__(self,rows,split,params,seed):
        owners={};ids=set()
        for r in rows:
            if r.get('kind') not in ('background','tool') or r.get('split') not in ('train','validation','test'):raise ValueError('invalid pool kind/split')
            if not r.get('id') or r['id'] in ids:raise ValueError('unique pool IDs required')
            ids.add(r['id']);keys=[('file',str(Path(r['path']).resolve()))]
            if r['kind']=='background':
                if r.get('projection_role')!='patient_without_tool' or not r.get('patient_id'):raise ValueError('background requires patient_id and patient_without_tool role')
                keys.append(('patient',r['patient_id']))
            else:
                if not isinstance(r.get('model_ids'),list) or not r['model_ids']:raise ValueError('tool model_ids required')
                keys.extend(('model',m) for m in r['model_ids'])
            for key in keys:
                if key in owners and owners[key]!=r['split']:raise ValueError('pool source appears in multiple splits')
                owners[key]=r['split']
        self.backgrounds=[r for r in rows if r['split']==split and r['kind']=='background']
        self.tools=[r for r in rows if r['split']==split and r['kind']=='tool']
        if not self.backgrounds or not self.tools:raise ValueError('each split needs background and tool pools')
        self.params=params;self.seed=seed

    def __len__(self):return max(len(self.backgrounds),len(self.tools))

    def __getitem__(self,index):
        rng=np.random.default_rng((int(torch.randint(0,2**32,(1,),dtype=torch.int64).item())+self.seed)%2**32)
        bg,_=load_projections(self.backgrounds[int(rng.integers(len(self.backgrounds)))]['path'])
        tool,_=load_projections(self.tools[int(rng.integers(len(self.tools)))]['path'])
        bg=bg[int(rng.integers(len(bg)))];tool=tool[int(rng.integers(len(tool)))]
        h,w=self.params['patch_shape']
        if bg.shape[0]<h or bg.shape[1]<w:raise ValueError('DTE patch exceeds background')
        for _ in range(100):
            y=int(rng.integers(bg.shape[0]-h+1));x=int(rng.integers(bg.shape[1]-w+1));background=bg[y:y+h,x:x+w]
            if background.mean()>=self.params['min_background']:break
        else:raise ValueError('no background patch above attenuation threshold')
        indices=np.argwhere(tool>0)
        if len(indices):
            lo=indices.min(0);hi=indices.max(0)+1;tool=tool[lo[0]:hi[0],lo[1]:hi[1]]
        else:tool=np.zeros((1,1),np.float32)
        if tool.shape[0]>h:
            start=int(rng.integers(tool.shape[0]-h+1));tool=tool[start:start+h]
        if tool.shape[1]>w:
            start=int(rng.integers(tool.shape[1]-w+1));tool=tool[:,start:start+w]
        if rng.integers(2):tool=np.flip(tool,0)
        if rng.integers(2):tool=np.flip(tool,1)
        canvas=np.zeros((h,w),np.float32)
        y=int(rng.integers(h-tool.shape[0]+1));x=int(rng.integers(w-tool.shape[1]+1))
        canvas[y:y+tool.shape[0],x:x+tool.shape[1]]=tool
        photons=float(rng.uniform(*self.params['photons_range']));scatter=float(rng.uniform(*self.params['scatter_range']))
        sigma=float(rng.uniform(0,self.params['blur_sigma_px']))
        image,target=synthesize_dte(background,canvas,rng,photons=photons,blur_sigma=sigma,scatter_fraction=scatter,flip=False)
        return torch.from_numpy(image[None]),torch.from_numpy(target[None])
