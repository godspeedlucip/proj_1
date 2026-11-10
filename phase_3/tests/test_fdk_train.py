import copy
import importlib
import json
import numpy as np
import pytest
import torch
from phase_3.tests.fdk_fixtures import synthetic_dataset,make_test_hybrid


def inputs(tmp_path):
    from phase_3.fdk_unet.config import get_experiment_config,resolve_config
    from phase_3.fdk_unet.manifest import load_manifest,load_splits
    mp,sp=synthetic_dataset(tmp_path/'data');r=get_experiment_config('A').resolved
    r.update(run_kind='smoke',experiment_id='A-smoke',patch_size=[32]*3,max_optimizer_steps=4,validate_every=2)
    return resolve_config(r),load_manifest(mp,require_labels=True),load_splits(sp)


def assert_state_equal(left,right):
    if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,atol=0,rtol=0)
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left:assert_state_equal(left[k],right[k])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_state_equal(a,b)
    else:assert left==right


def test_continuous_and_resumed_training_match_next_update(tmp_path,monkeypatch):
    t=importlib.import_module('phase_3.fdk_unet.train')
    from phase_3.fdk_unet.checkpoint import load_run
    from phase_3.fdk_unet.data import PatchStream
    cfg,m,splits=inputs(tmp_path)
    common=dict(fold=0,device=torch.device('cpu'))
    full=t.train(cfg,m,splits,output=tmp_path/'full',**common)
    save=t.save_checkpoint
    def interrupt(path,payload):
        save(path,payload)
        if path.name=='last.pt' and payload['training_state']['global_step']==2:raise RuntimeError('simulated interruption')
    monkeypatch.setattr(t,'save_checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):t.train(cfg,m,splits,output=tmp_path/'resume',**common)
    monkeypatch.setattr(t,'save_checkpoint',save)
    resumed=t.train(cfg,m,splits,output=tmp_path/'resume',resume=tmp_path/'resume/last.pt',**common)
    assert full.completed_steps==resumed.completed_steps==4
    a=load_run(full.last_checkpoint,device=torch.device('cpu'),for_training=True)
    b=load_run(resumed.last_checkpoint,device=torch.device('cpu'),for_training=True)
    assert_state_equal(a.model.state_dict(),b.model.state_dict())
    for key in ('optimizer','rng','patch_stream','scaler'):assert_state_equal(a.training_state[key],b.training_state[key])
    sa,sb=PatchStream(m,splits[0],cfg,None),PatchStream(m,splits[0],cfg,None)
    sa.load_state_dict(a.training_state['patch_stream']);sb.load_state_dict(b.training_state['patch_stream'])
    ba,bb=sa.next_batch(),sb.next_batch();assert ba.case_ids==bb.case_ids;assert_state_equal(ba.image_unet,bb.image_unet)
    logs=[json.loads(line) for line in (tmp_path/'resume/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs if r['event']=='train']==[1,2,3,4]


def test_validation_uses_group_dice_and_keeps_earlier_tie(tmp_path,monkeypatch):
    t=importlib.import_module('phase_3.fdk_unet.train');cfg,m,splits=inputs(tmp_path)
    monkeypatch.setattr(t,'validate',lambda *a,**k:{'mean':{'dice':.3},'groups':{}})
    result=t.train(cfg,m,splits,fold=0,device=torch.device('cpu'),output=tmp_path/'run')
    p=torch.load(result.best_checkpoint,map_location='cpu',weights_only=False)
    assert p['training_state']['global_step']==2
    assert result.best_validation_dice==.3


def test_existing_output_and_changed_training_data_rejected(tmp_path):
    from phase_3.fdk_unet.train import train
    cfg,m,splits=inputs(tmp_path);out=tmp_path/'run'
    train(cfg,m,splits,fold=0,device=torch.device('cpu'),output=out)
    with pytest.raises(FileExistsError):train(cfg,m,splits,fold=0,device=torch.device('cpu'),output=out)
    import nibabel as nib
    nii=nib.load(m.cases[0].image);x=nii.get_fdata();x[0,0,0]+=1;nib.save(nib.Nifti1Image(x,nii.affine,nii.header),m.cases[0].image)
    with pytest.raises(ValueError,match='data'):train(cfg,m,splits,fold=0,device=torch.device('cpu'),output=out,resume=out/'last.pt')


def test_nonfinite_step_and_unavailable_cuda_fail(monkeypatch):
    from phase_3.fdk_unet.train import train_step
    from phase_3.fdk_unet.data import SampleBatch
    from phase_3.fdk_unet.predict import validate_device
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError):validate_device(torch.device('cuda'))
    model=torch.nn.Conv3d(1,1,1);optimizer=torch.optim.Adam(model.parameters())
    x=torch.full((1,1,2,2,2),float('nan'));batch=SampleBatch(x,None,torch.zeros_like(x),torch.ones_like(x),('x',),[])
    with pytest.raises(ValueError):train_step(model,batch,optimizer,None,amp_dtype=None)


def test_determinism_policy_keeps_cpu_strict_and_discloses_cuda_limit():
    from phase_3.fdk_unet.train import configure_determinism
    gpu=configure_determinism(torch.device('cuda'))
    assert gpu['warn_only'] is True and 'max_pool3d' in gpu['limitation']
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    cpu=configure_determinism(torch.device('cpu'))
    assert not cpu['warn_only'] and not torch.is_deterministic_algorithms_warn_only_enabled()


def test_s1_frozen_prior_and_trainable_fusion():
    from phase_3.fdk_unet.train import train_step
    from phase_3.fdk_unet.plain_unet import PlainUNet3D
    from phase_3.fdk_unet.data import SampleBatch
    torch.manual_seed(2026);model=make_test_hybrid('C',PlainUNet3D().state_dict()).train()
    before={k:v.clone() for k,v in model.prior.state_dict().items()}
    align=model.skips['2'].align.weight.detach().clone();gate=model.skips['2'].gate[2].weight.detach().clone()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    x=torch.randn(2,1,32,32,32);batch=SampleBatch(x,x.sigmoid(),(x>1).float(),torch.ones_like(x),('a','b'),[])
    for _ in range(3):train_step(model,batch,optimizer,None,amp_dtype=None)
    assert_state_equal(before,model.prior.state_dict())
    assert not torch.equal(align,model.skips['2'].align.weight) and not torch.equal(gate,model.skips['2'].gate[2].weight)
    for block in model.unet.encoder:assert block[0].weight.grad.abs().sum()>0


@pytest.mark.slow
def test_fixed_asymmetric_phantoms_overfit(tmp_path):
    from scipy.ndimage import gaussian_filter
    from phase_3.fdk_unet.train import train_step
    from phase_3.fdk_unet.config import get_experiment_config
    from phase_3.fdk_unet.model_factory import build_model
    from phase_3.fdk_unet.data import SampleBatch
    y=np.zeros((2,1,32,32,32),np.float32)
    y[0,0,5:25,8:12,10:14]=1;y[0,0,21:25,8:22,10:14]=1
    y[1,0,7:11,5:26,17:21]=1;y[1,0,7:24,22:26,17:21]=1
    rng=np.random.default_rng(2026)
    x=np.stack([gaussian_filter(v[0],.8) for v in y])[:,None]+rng.normal(0,.015,y.shape)
    x=((x-x.mean((2,3,4),keepdims=True))/x.std((2,3,4),keepdims=True)).astype(np.float32)
    batch=SampleBatch(torch.from_numpy(x),None,torch.from_numpy(y),torch.ones_like(torch.from_numpy(y)),('p0','p1'),[])
    model=build_model(get_experiment_config('A'));optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);curve=[]
    for step in range(1,501):
        model.train();metrics=train_step(model,batch,optimizer,None,amp_dtype=None)
        if step%25==0:
            model.eval()
            with torch.no_grad():p=(model(batch.image_unet).sigmoid()>=.5).float();dice=(2*(p*batch.target).sum((1,2,3,4))/(p.sum((1,2,3,4))+batch.target.sum((1,2,3,4)))).min().item()
            curve.append({'step':step,'minimum_training_dice':dice,**metrics})
            if dice>=.95:break
    (tmp_path/'phantom_curve.json').write_text(json.dumps(curve,indent=2))
    assert dice>=.95
