import json
import numpy as np
import pytest
import torch
from stent_repro.training.data import PreparedDataset, read_training_manifest
from stent_repro.training.engine import train_prepared


def write_array(path, array):
    np.save(path, np.asarray(array, dtype=np.float32))


def test_a_dte_data_and_frozen_training_statistics(tmp_path):
    for name, value in [('x1',2),('y1',.2),('x2',10),('y2',.4)]:
        write_array(tmp_path / (name+'.npy'), np.full((16,16),value))
    manifest = {'samples':[
        {'id':'a','split':'train','model_ids':['m1'],'input':'x1.npy','target':'y1.npy'},
        {'id':'b','split':'validation','model_ids':['m2'],'input':'x2.npy','target':'y2.npy'}]}
    path = tmp_path/'dte.json'; path.write_text(json.dumps(manifest),encoding='utf8')
    rows = read_training_manifest(path,'A','dte')
    train = PreparedDataset(rows,'train','A','dte')
    val = PreparedDataset(rows,'validation','A','dte')
    assert train[0][0].shape == (1,16,16)
    assert torch.allclose(train[0][1],torch.full((1,16,16),.2))
    result = train_prepared(train,val,torch.nn.Conv2d(1,1,1), tmp_path/'run', method='A', stage='dte', epochs=2, batch_size=1, learning_rate=.01, seed=3, device='cpu')
    assert result['normalization']['mean'] == pytest.approx(2)
    assert result['normalization']['std'] > 0
    checkpoint = torch.load(tmp_path/'run'/'last.pt',weights_only=True)
    assert checkpoint['epoch'] == 2
    assert checkpoint['normalization'] == result['normalization']
    assert len(result['history']) == 2


def test_a_dtr_slices_and_source_restriction(tmp_path):
    x = np.broadcast_to(np.arange(5)[:,None,None],(5,16,16)).copy()
    y = np.zeros_like(x); y[2] = 1
    write_array(tmp_path/'x.npy',x); write_array(tmp_path/'y.npy',y)
    row = {'id':'a','split':'train','model_ids':['m'], 'input':'x.npy','target':'y.npy','source':'ideal_tool_fdk','angles_deg':[0,45,90,135]}
    path=tmp_path/'m.json'; path.write_text(json.dumps({'samples':[row]}),encoding='utf8')
    rows=read_training_manifest(path,'A','dtr',profile='A-original-4view')
    dataset=PreparedDataset(rows,'train','A','dtr',slice_channels=3)
    assert len(dataset)==5
    inputs,target=dataset[2]
    assert inputs.shape==(3,16,16) and torch.all(target==1)
    assert torch.all(inputs[0] < inputs[1]) and torch.all(inputs[1] < inputs[2])
    row['source']='dte_fdk'
    path.write_text(json.dumps({'samples':[row]}),encoding='utf8')
    with pytest.raises(ValueError,match='source'): read_training_manifest(path,'A','dtr',profile='A-original-4view')


def test_b_dtr_channels_ablation_and_source(tmp_path):
    for i in range(3): write_array(tmp_path/f'x{i}.npy', np.full((16,16,16),i+1))
    write_array(tmp_path/'y.npy',np.ones((16,16,16)))
    row={'id':'b','split':'train','model_ids':['s','w'], 'input':['x0.npy','x1.npy','x2.npy'], 'target':'y.npy','source':'error_augmented_per_view_backprojection','angles_deg':[0,30,60]}
    path=tmp_path/'m.json'; path.write_text(json.dumps({'samples':[row]}),encoding='utf8')
    rows=read_training_manifest(path,'B','dtr',profile='B-adapted-3view')
    x,y=PreparedDataset(rows,'train','B','dtr')[0]
    assert x.shape==(3,16,16,16) and x[:,0,0,0].tolist()==[1,2,3]
    summed,_=PreparedDataset(rows,'train','B','dtr',input_channels=1)[0]
    assert summed.shape==(1,16,16,16) and summed[0,0,0,0]==6
    assert y.shape==(1,16,16,16)


def test_manifest_rejects_leakage_and_wrong_angles(tmp_path):
    rows=[{'id':'a','split':'train','model_ids':['m'],'input':'a','target':'b'}, {'id':'b','split':'validation','model_ids':['m'],'input':'c','target':'d'}]
    p=tmp_path/'m.json'; p.write_text(json.dumps({'samples':rows}),encoding='utf8')
    with pytest.raises(ValueError, match='split'): read_training_manifest(p,'A','dte')


def test_resume_rejects_wrong_run_contract(tmp_path):
    class DS(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self,i): return torch.ones(1,8,8), torch.ones(1,8,8)
    ds=DS()
    train_prepared(ds,ds,torch.nn.Conv2d(1,1,1),tmp_path,method='B',stage='dte',epochs=1,batch_size=1,learning_rate=.01,seed=1,device='cpu')
    with pytest.raises(ValueError,match='contract'):
        train_prepared(ds,ds,torch.nn.Conv2d(1,1,1),tmp_path,method='A',stage='dte',epochs=2,batch_size=1,learning_rate=.01,seed=1,device='cpu',resume=tmp_path/'last.pt')

def test_resume_matches_uninterrupted_training(tmp_path):
    class DS(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self,i): return torch.full((1,4,4),float(i+1)), torch.full((1,4,4),float(i)*.1)
    def model():
        torch.manual_seed(90)
        return torch.nn.Sequential(torch.nn.Conv2d(1,2,1),torch.nn.Dropout(.2),torch.nn.Conv2d(2,1,1))
    kw = dict(method='B',stage='dte',batch_size=2,learning_rate=.01,seed=7,device='cpu')
    full=tmp_path/'full'; resumed=tmp_path/'resumed'
    train_prepared(DS(),DS(),model(),full,epochs=2,**kw)
    train_prepared(DS(),DS(),model(),resumed,epochs=1,**kw)
    train_prepared(DS(),DS(),model(),resumed,epochs=2,resume=resumed/'last.pt',**kw)
    a=torch.load(full/'last.pt',weights_only=True); b=torch.load(resumed/'last.pt',weights_only=True)
    for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert a['history'][1]['validation_loss']==pytest.approx(b['history'][1]['validation_loss'])


def test_cli_rejects_strict_before_allocating_model(tmp_path):
    from stent_repro.training.cli import main
    with pytest.raises(RuntimeError,match='strict'):
        main(['--method','B','--stage','dte','--config','configs/voth2022/original_3view_60step.yaml', '--manifest',str(tmp_path/'missing.json'),'--output',str(tmp_path/'out'),'--mode','strict'])

def test_cli_a_dte_smoke_on_tiff_pairs(tmp_path):
    import tifffile
    from stent_repro.training.cli import main
    rows=[]
    for i, split in enumerate(('train','validation')):
        tifffile.imwrite(tmp_path/f'x{i}.tif',np.full((16,16),10+i,dtype=np.float32))
        tifffile.imwrite(tmp_path/f'y{i}.tif',np.full((16,16),.25,dtype=np.float32))
        rows.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'input':f'x{i}.tif','target':f'y{i}.tif'})
    manifest=tmp_path/'m.json'; manifest.write_text(json.dumps({'samples':rows}),encoding='utf8')
    result=main(['--method','A','--stage','dte','--config','configs/eulig2021/original_4view.yaml','--manifest',str(manifest),'--output',str(tmp_path/'out'),'--mode','prepared-surrogate','--epochs','1','--batch-size','1','--device','cpu'])
    assert len(result['history'])==1
    assert (tmp_path/'out'/'last.pt').is_file() and (tmp_path/'out'/'best.pt').is_file()
    checkpoint = torch.load(tmp_path/'out'/'last.pt',weights_only=True)
    assert checkpoint['contract']['run_info']['mode']=='prepared-surrogate'


def test_cli_b_dtr_smoke_with_ordered_precomputed_channels(tmp_path):
    from stent_repro.training.cli import main
    rows=[]
    for i, split in enumerate(('train','train','validation')):
        for k in range(3): write_array(tmp_path/f'x{i}_{k}.npy', np.full((16,16,16), .1*(k+1)))
        write_array(tmp_path/f'y{i}.npy',np.zeros((16,16,16)))
        rows.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'input':[f'x{i}_{k}.npy' for k in range(3)],'target':f'y{i}.npy','source':'error_augmented_per_view_backprojection','angles_deg':[0,30,60]})
    path=tmp_path/'m.json'; path.write_text(json.dumps({'samples':rows}),encoding='utf8')
    result=main(['--method','B','--stage','dtr','--config','configs/voth2022/adapted_3view_0_30_60.yaml','--manifest',str(path),'--output',str(tmp_path/'out'),'--mode','prepared-surrogate','--epochs','1','--batch-size','2','--device','cpu'])
    assert np.isfinite(result['best_validation_loss'])

def test_reject_dtr_mismatched_physical_grids(tmp_path):
    from stent_repro.io import save_volume
    for k in range(3):
        save_volume(tmp_path/f'{k}.npz',np.zeros((2,2,2)),{'spacing_mm':[1,1,1],'origin_mm':[0,0,0]})
    save_volume(tmp_path/'target.npz',np.zeros((2,2,2)),{'spacing_mm':[2,1,1],'origin_mm':[0,0,0]})
    row={'id':'b','split':'train','model_ids':['m'],'input':[str(tmp_path/f'{k}.npz') for k in range(3)],'target':str(tmp_path/'target.npz')}
    with pytest.raises(ValueError,match='grid'): PreparedDataset([row],'train','B','dtr')[0]


def test_dte_rejects_patient_leakage_when_identified(tmp_path):
    rows=[{'id':'a','split':'train','model_ids':['m1'],'patient_id':'p','input':'a','target':'b'}, {'id':'b','split':'validation','model_ids':['m2'],'patient_id':'p','input':'c','target':'d'}]
    p=tmp_path/'m.json'; p.write_text(json.dumps({'samples':rows}),encoding='utf8')
    with pytest.raises(ValueError,match='patient.*split'): read_training_manifest(p,'A','dte')

def test_cli_a_dtr_smoke_with_ideal_fdk(tmp_path):
    from stent_repro.training.cli import main
    rows=[]
    for i, split in enumerate(('train','validation')):
        write_array(tmp_path/f'v{i}.npy', np.random.default_rng(i).normal(size=(2,16,16)))
        write_array(tmp_path/f'y{i}.npy', np.zeros((2,16,16)))
        rows.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'input':f'v{i}.npy','target':f'y{i}.npy','source':'ideal_tool_fdk','angles_deg':[5,50,95,140]})
    path=tmp_path/'m.json'; path.write_text(json.dumps({'samples':rows}),encoding='utf8')
    result=main(['--method','A','--stage','dtr','--config','configs/eulig2021/original_4view.yaml','--manifest',str(path),'--output',str(tmp_path/'out'),'--mode','prepared-surrogate','--epochs','1','--batch-size','1','--slice-channels','3','--device','cpu'])
    assert np.isfinite(result['best_validation_loss'])

def test_b_augmentation_transforms_channels_and_labels_together_only_in_training(tmp_path):
    x = np.zeros((4,4,4),dtype=np.float32); x[0,0,1]=1
    for i in range(3): write_array(tmp_path/f'p{i}.npy',x)
    write_array(tmp_path/'y.npy',x)
    common={'input':[str(tmp_path/f'p{i}.npy') for i in range(3)],'target':str(tmp_path/'y.npy'),'model_ids':['m']}
    rows=[{**common,'id':'a','split':'train'},{**common,'id':'b','split':'validation'}]
    train=PreparedDataset(rows,'train','B','dtr',augment=True)
    val=PreparedDataset(rows,'validation','B','dtr',augment=False)
    torch.manual_seed(1)
    px,py=train[0]
    assert torch.equal(px[0],py[0]) and torch.equal(px[1],py[0])
    assert not torch.equal(py[0],torch.from_numpy(x))
    vx,vy=val[0]
    assert torch.equal(vy[0],torch.from_numpy(x)) and torch.equal(vx[0],vy[0])

def test_manifest_rejects_reused_files_across_splits(tmp_path):
    rows=[{'id':'a','split':'train','model_ids':['m1'],'input':'same.tif','target':'a.tif'}, {'id':'b','split':'validation','model_ids':['m2'],'input':'same.tif','target':'b.tif'}]
    p=tmp_path/'m.json'; p.write_text(json.dumps({'samples':rows}),encoding='utf8')
    with pytest.raises(ValueError,match='file.*split'): read_training_manifest(p,'A','dte')
