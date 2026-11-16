import numpy as np
import pytest
import torch
from stent_repro.training.sampling import RunningStatistics, EpochDataset
from stent_repro.training.online import OnlineDTEDataset
from stent_repro.training.engine import train_prepared
from stent_repro.training.data import PreparedDataset
from stent_repro.io import save_projections, save_volume


class RandomDataset:
    def __len__(self):return 3
    def __getitem__(self,i):
        x=torch.ones(1,4,4)*(i+float(torch.rand(()))+np.random.random())
        return x,x*.2


def test_epoch_count_remainder_and_replay():
    a=EpochDataset(RandomDataset(),7,3,0,'train')
    b=EpochDataset(RandomDataset(),7,3,0,'train')
    assert len(a)==7
    assert [len(x) for x,y in torch.utils.data.DataLoader(a,batch_size=3)]==[3,3,1]
    torch.testing.assert_close(a[3][0],b[3][0],rtol=0,atol=0)
    assert not torch.equal(a[3][0],EpochDataset(RandomDataset(),7,3,1,'train')[3][0])


def test_running_stats_and_resume_exclude_validation(tmp_path):
    def model():
        torch.manual_seed(99);return torch.nn.Conv2d(1,1,1)
    kw=dict(method='A',stage='dte',batch_size=3,learning_rate=.001,seed=3,device='cpu',
            train_pairs=7,validation_pairs=2,normalization_mode='running')
    train_prepared(RandomDataset(),RandomDataset(),model(),tmp_path/'full',epochs=2,**kw)
    train_prepared(RandomDataset(),RandomDataset(),model(),tmp_path/'res',epochs=1,**kw)
    train_prepared(RandomDataset(),RandomDataset(),model(),tmp_path/'res',epochs=2,resume=tmp_path/'res'/'last.pt',**kw)
    a=torch.load(tmp_path/'full'/'last.pt',weights_only=True);b=torch.load(tmp_path/'res'/'last.pt',weights_only=True)
    assert a['normalization']['count']==2*7*16
    assert a['history'][0]['train_samples']==7 and a['history'][0]['optimizer_steps']==3
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert a['normalization']==b['normalization']


def test_online_pool_recombines_and_checks_splits(tmp_path):
    rows=[]
    for i,split in enumerate(('train','validation')):
        save_projections(tmp_path/f'bg{i}.npz',np.full((2,24,24),.2+i*.1))
        tool=np.zeros((1,16,16),np.float32);tool[:,5:8,6:9]=.5
        save_projections(tmp_path/f't{i}.npz',tool)
        rows += [{'id':f'b{i}','kind':'background','split':split,'patient_id':f'p{i}','projection_role':'patient_without_tool','path':str(tmp_path/f'bg{i}.npz')},
                 {'id':f't{i}','kind':'tool','split':split,'model_ids':[f'm{i}'],'path':str(tmp_path/f't{i}.npz')}]
    params={'patch_shape':[16,16],'min_background':.01,'photons_range':[1e8,1e8],'blur_sigma_px':0,'scatter_range':[0,0]}
    ds=EpochDataset(OnlineDTEDataset(rows,'train',params,4),8,4,0,'train')
    assert any(not torch.equal(ds[0][1],ds[i][1]) for i in range(1,8))
    x,y=ds[0];np.testing.assert_allclose(x.numpy(),y.numpy()+.2,atol=.002)
    rows[2]['patient_id']='p0'
    with pytest.raises(ValueError,match='split'):OnlineDTEDataset(rows,'train',params,4)


def test_a_dtr_random_crop_matches_center_target(tmp_path):
    mask=np.zeros((7,20,22),np.uint8);mask[3,4:8,6:9]=1
    save_volume(tmp_path/'x.npz',mask.astype(np.float32));save_volume(tmp_path/'y.npz',mask)
    rows=[{'split':'train','input':str(tmp_path/'x.npz'),'target':str(tmp_path/'y.npz')}]
    ds=PreparedDataset(rows,'train','A','dtr',slice_channels=3,patch_shape=[12,12],random_crop=True)
    for i in range(7):
        x,y=ds[i]
        assert x.shape==(3,12,12) and y.shape==(1,12,12)
        assert torch.equal(x[1]>0,y[0].bool())
