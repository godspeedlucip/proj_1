import json
import numpy as np
import pytest
import torch
from stent_repro.inference import reconstruct, run_inference, load_trained_model


class Capture(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__(); self.seen=[]; self.center=center
    def forward(self,x):
        self.seen.append(x.detach().cpu().clone())
        return torch.sigmoid(x[:,x.shape[1]//2:x.shape[1]//2+1] if self.center else x[:,:1])


def test_a_padding_and_center_slice_order(workflow_config,monkeypatch):
    c=workflow_config(); c['geometry']['detector_shape']=[15,13]
    c['geometry']['grid']['shape']=[5,9,11]
    v=np.broadcast_to(np.arange(5)[:,None,None],(5,9,11)).astype(np.float32)
    monkeypatch.setattr('stent_repro.inference.fdk',lambda *a,**k:v)
    dte=Capture();dtr=Capture(center=True)
    out=reconstruct(np.ones((3,15,13)),c,dte,dtr,normalization={'mean':0,'std':1})
    assert out['dte'].shape==(3,15,13)
    assert dte.seen[0].shape[-2:]==(16,16)
    assert out['probability'].shape==v.shape
    expected=torch.sigmoid(torch.tensor((np.arange(5)-2)/np.std(np.arange(5)))).numpy()
    np.testing.assert_allclose(out['probability'][:,4,4],expected,rtol=1e-6)
    assert dtr.seen[0][0,0,4,4]==dtr.seen[0][0,2,4,4]


def test_b_preserves_channels_or_explicit_sum(workflow_config,monkeypatch):
    c=workflow_config('B')
    channels=np.stack([np.full((16,16,16),i,np.float32) for i in (1,2,3)])
    monkeypatch.setattr('stent_repro.inference.backproject_each',lambda *a,**k:channels)
    dtr=Capture();out=reconstruct(np.ones((3,16,16)),c,Capture(),dtr)
    assert dtr.seen[0].shape==(1,3,16,16,16)
    assert out['geometric'].shape==(3,16,16,16)
    c['profile']='B-adapted-one-channel';c['network']['input_channels']=1
    dtr=Capture();reconstruct(np.ones((3,16,16)),c,Capture(),dtr)
    assert torch.all(dtr.seen[0]==6)


def test_inference_refuses_truth_and_wrong_checkpoint(tmp_path,workflow_config):
    c=workflow_config(); p=tmp_path/'m.json'
    p.write_text(json.dumps({'cases':[{'id':'a','projections':'x.npz','truth':'leak.npz'}]}),encoding='utf8')
    with pytest.raises(ValueError,match='truth'):run_inference(p,c,{},tmp_path/'out')
    torch.save({'contract':{'method':'B','stage':'dte'}},tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='method|stage'):load_trained_model(tmp_path/'bad.pt',c,'dtr')


def test_rejects_nonfinite_input(workflow_config):
    with pytest.raises(ValueError,match='finite'):
        reconstruct(np.full((3,16,16),np.nan),workflow_config('B'),Capture(),Capture())
