import json
import numpy as np
import pytest
from stent_repro.io import save_volume,save_projections
from stent_repro.preprocessing import prepare_samples, synthesize_dte, perturb_projection
from stent_repro.training.data import read_training_manifest,PreparedDataset

def test_dte_pair_is_continuous_reproducible_and_aligned():
    background=np.full((10,10),.4,dtype=np.float32)
    tool=np.zeros((10,10),dtype=np.float32); tool[2:5,3:7]=.25
    a,t=synthesize_dte(background,tool,np.random.default_rng(1),photons=1e8,blur_sigma=0,scatter_fraction=0,flip=False)
    b,u=synthesize_dte(background,tool,np.random.default_rng(1),photons=1e8,blur_sigma=0,scatter_fraction=0,flip=False)
    np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(t,u)
    np.testing.assert_allclose(t,tool)
    np.testing.assert_allclose(a,background+tool,atol=.002)
    assert .25 in np.unique(t)

def test_error_perturbation_has_false_negatives_and_positives():
    x=np.zeros((64,64),dtype=np.float32); x[16:48,16:48]=1
    y=perturb_projection(x,np.random.default_rng(3),false_negative_rate=1, false_positive_rate=.03, false_positive_scale=.5)
    assert np.count_nonzero(y[16:48,16:48])==0
    assert np.any(y[:10,:] > 0)

def test_prepare_from_projection_and_truth_feeds_both_training_datasets(tmp_path):
    cases=[]
    for i,split in enumerate(('train','validation')):
        truth=np.zeros((8,8,8),np.uint8); truth[2:6,3:5,3:5]=1
        save_volume(tmp_path/f't{i}.npz',truth,{'spacing_mm':[1,1,1],'origin_mm':[-3.5,-3.5,-3.5],'direction':[[1,0,0],[0,1,0],[0,0,1]]})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,16,16),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    manifest=tmp_path/'raw.json'; manifest.write_text(json.dumps({'cases':cases}),encoding='utf-8')
    params={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':1,'detector_shape':[16,16], 'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':0,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':16,'min_background':.01,'seed':8}
    out=tmp_path/'prep'
    prepare_samples(manifest,out,method='B',profile='B-adapted-3view',params=params)
    dte=read_training_manifest(out/'dte.json','B','dte')
    dtr=read_training_manifest(out/'dtr.json','B','dtr',profile='B-adapted-3view')
    assert len(dte)==6 and len(dtr)==2
    a,b=PreparedDataset(dte,'train','B','dte')[0]
    assert a.shape==b.shape==(1,16,16) and b.max()>0
    x,y=PreparedDataset(dtr,'train','B','dtr')[0]
    assert x.shape==(3,8,8,8) and y.shape==(1,8,8,8)
    assert (out/'provenance.json').is_file()

def test_requires_explicit_grid_and_log_background(tmp_path):
    save_volume(tmp_path/'truth.npy',np.ones((4,4,4)),{})
    save_projections(tmp_path/'bg.npy',np.ones((3,4,4)),{})
    cases=[{'id':'a','split':'train','model_ids':['m'],'patient_id':'p','projection_role':'patient_without_tool','truth':'truth.npy','background':'bg.npy','angles_deg':[0,30,60]}]
    m=tmp_path/'m.json'; m.write_text(json.dumps({'cases':cases}),encoding='utf8')
    params={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':1,'detector_shape':[4,4], 'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':0,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':4,'min_background':0,'seed':8}
    with pytest.raises(ValueError,match='spacing|grid'): prepare_samples(m,tmp_path/'out',method='A',profile='A-adapted-3view',params=params)

def test_prepare_a_trains_dte_and_ideal_dtr(tmp_path):
    from stent_repro.training.cli import main
    cases=[]
    for i,split in enumerate(('train','validation')):
        truth=np.zeros((8,16,16),np.uint8); truth[2:6,6:10,7:9]=1
        save_volume(tmp_path/f't{i}.npz',truth,{'spacing_mm':[1,1,1],'origin_mm':[-7.5,-7.5,-3.5],'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,16,16),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf-8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[16,16], 'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':.2,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':16,'min_background':.01,'seed':8}
    out=tmp_path/'prep'; prepare_samples(path,out,method='A',profile='A-adapted-3view',params=p)
    common=['--method','A','--config','configs/eulig2021/adapted_3view_0_30_60.yaml','--mode','prepared-surrogate','--epochs','1','--batch-size','2','--device','cpu']
    for stage in ('dte','dtr'):
        result=main(common+['--stage',stage,'--manifest',str(out/f'{stage}.json'),'--output',str(tmp_path/f'train_{stage}')]+(['--online-dte'] if stage=='dte' else []))
        assert np.isfinite(result['best_validation_loss'])
    import torch
    checkpoint=torch.load(tmp_path/'train_dtr'/'last.pt',weights_only=True)
    assert checkpoint['contract']['run_info']['preprocessing']['profile']=='A-adapted-3view'
    dte_checkpoint=torch.load(tmp_path/'train_dte'/'last.pt',weights_only=True)
    assert dte_checkpoint['contract']['run_info']['online_dte'] is True

def test_center_crop_updates_grid_origin(tmp_path):
    from stent_repro.preprocessing import crop_volume_grid
    from stent_repro.geometry import Grid
    v=np.zeros((8,10,12)); grid=Grid(v.shape,(.5,1,2),(-1,-2,-3))
    cropped,newgrid=crop_volume_grid(v,grid,(4,6,8))
    assert cropped.shape==(4,6,8)
    assert newgrid.origin_mm==pytest.approx((0,0,1))

def test_preprocess_cli_runs_from_parameter_file(tmp_path):
    from stent_repro.preprocess import main
    cases=[]
    for i,split in enumerate(('train','validation')):
        save_volume(tmp_path/f't{i}.npz',np.ones((4,4,4),np.uint8),{'spacing_mm':[1,1,1],'origin_mm':[-1.5]*3,'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,8,8),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf-8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[8,8], 'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':0,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':8,'min_background':.01,'seed':8}
    params=tmp_path/'params.json'; params.write_text(json.dumps(p),encoding='utf-8')
    main(['--method','A','--profile','A-adapted-3view','--manifest',str(path),'--params',str(params),'--output',str(tmp_path/'result')])
    assert (tmp_path/'result'/'dtr.json').exists()

def test_preprocess_b_then_train_three_channel_dtr(tmp_path):
    from stent_repro.training.cli import main
    cases=[]
    for i,split in enumerate(('train','train','validation')):
        truth=np.zeros((16,16,16),np.uint8); truth[5:11,6:10,7:9]=1
        save_volume(tmp_path/f't{i}.npz',truth,{'spacing_mm':[1,1,1],'origin_mm':[-7.5]*3,'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,16,16),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[16,16],'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':0,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':16,'min_background':.01,'seed':8}
    out=tmp_path/'prep'; prepare_samples(path,out,method='B',profile='B-adapted-3view',params=p)
    r=main(['--method','B','--stage','dtr','--config','configs/voth2022/adapted_3view_0_30_60.yaml','--manifest',str(out/'dtr.json'),'--output',str(tmp_path/'train'),'--mode','prepared-surrogate','--epochs','1','--batch-size','2','--device','cpu'])
    assert np.isfinite(r['best_validation_loss'])

def test_training_refuses_changed_generated_manifest(tmp_path):
    from stent_repro.training.cli import main
    folder=tmp_path/'prep'; folder.mkdir()
    (folder/'dtr.json').write_text('{}',encoding='utf8')
    (folder/'provenance.json').write_text(json.dumps({'generated_manifests_sha256':{'dtr.json':'incorrect'}}),encoding='utf8')
    with pytest.raises(ValueError,match='provenance'):
        main(['--method','A','--stage','dtr','--config','configs/eulig2021/adapted_3view_0_30_60.yaml','--manifest',str(folder/'dtr.json'),'--output',str(tmp_path/'out'),'--mode','prepared-surrogate','--epochs','1','--device','cpu'])

def test_b_rejects_backprojection_channel_angle_mismatch(tmp_path):
    from stent_repro.io import save_volume
    meta={'spacing_mm':[1,1,1],'origin_mm':[0,0,0],'direction':np.eye(3).tolist()}
    save_volume(tmp_path/'target.npz',np.zeros((2,2,2)),meta)
    for i,a in enumerate([30,0,60]): save_volume(tmp_path/f'{i}.npz',np.ones((2,2,2)),{**meta,'view_angle_deg':a})
    row={'id':'b','split':'train','model_ids':['m'],'angles_deg':[0,30,60],'input':[str(tmp_path/f'{i}.npz') for i in range(3)],'target':str(tmp_path/'target.npz')}
    with pytest.raises(ValueError,match='angle'):
        PreparedDataset([row],'train','B','dtr')[0]

def test_preprocessing_rejects_nonfinite_parameters_before_writing(tmp_path):
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':1,'detector_shape':[4,4], 'attenuation_per_mm':float('nan'),'photons':10000,'blur_sigma_px':0,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':4,'min_background':0,'seed':8}
    with pytest.raises(ValueError,match='finite'):
        prepare_samples(tmp_path/'missing.json',tmp_path/'out',method='A',profile='A-adapted-3view',params=p)
    assert not (tmp_path/'out').exists()

def test_dte_can_use_seventeen_views_independently_of_dtr_angles(tmp_path):
    cases=[]
    for i,split in enumerate(('train','validation')):
        save_volume(tmp_path/f't{i}.npz',np.ones((4,4,4),np.uint8),{'spacing_mm':[1,1,1],'origin_mm':[-1.5]*3,'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((17,8,8),.3,np.float32),{'angles_deg':list(range(0,170,10)),'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[8,8], 'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':0,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':8,'min_background':.01,'seed':8}
    out=tmp_path/'prep'; prepare_samples(path,out,method='A',profile='A-adapted-3view',params=p)
    assert len(json.loads((out/'dte.json').read_text(encoding='utf8'))['samples'])==34
    assert len(json.loads((out/'dtr.json').read_text(encoding='utf8'))['samples'])==2

def test_detector_intensity_explicit_calibration():
    from stent_repro.preprocessing import intensity_to_line_integrals
    x=np.array([[100.,55.,10.]])
    y=intensity_to_line_integrals(x,flat_field=100,dark_field=10,transmission_floor=.01)
    np.testing.assert_allclose(y,[[0,-np.log(.5),-np.log(.01)]])
    with pytest.raises(ValueError,match='flat'):
        intensity_to_line_integrals(x,flat_field=10,dark_field=10,transmission_floor=.01)

def test_raw_tiff_inputs_can_declare_calibration_and_grid_in_manifest(tmp_path):
    import tifffile
    cases=[]
    for i,split in enumerate(('train','validation')):
        tifffile.imwrite(tmp_path/f't{i}.tif',np.ones((4,4,4),np.uint8),photometric='minisblack')
        tifffile.imwrite(tmp_path/f'b{i}.tif',np.full((3,8,8),60,np.uint16),photometric='minisblack')
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.tif','truth_grid':{'spacing_mm':[1,1,1],'origin_mm':[-1.5]*3,'direction':np.eye(3).tolist()},'background':f'b{i}.tif','background_units':'detector_intensity','background_angles_deg':[0,30,60],'angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[8,8], 'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':0,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':8,'min_background':.01,'seed':8,'flat_field':100,'dark_field':0,'transmission_floor':.001}
    out=tmp_path/'prep'; prepare_samples(path,out,method='B',profile='B-adapted-3view',params=p)
    assert len(json.loads((out/'dte.json').read_text(encoding='utf8'))['samples'])==6

def test_measured_projection_is_not_recombined_or_corrupted(tmp_path):
    cases=[]
    for i,split in enumerate(('train','validation')):
        truth=np.ones((4,4,4),np.uint8)
        save_volume(tmp_path/f't{i}.npz',truth,{'spacing_mm':[1,1,1],'origin_mm':[-1.5]*3,'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,8,8),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','projection_role':'measured_with_tool','registration_confirmed':True,'angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[8,8], 'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':1,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':8,'min_background':.01,'seed':8}
    out=tmp_path/'prep'; prepare_samples(path,out,method='A',profile='A-adapted-3view',params=p)
    rows=read_training_manifest(out/'dte.json','A','dte')
    x,y=PreparedDataset(rows,'train','A','dte')[0]
    np.testing.assert_allclose(x.numpy(),.3)
    assert y.max() > 0

def test_online_dte_reaugments_only_training_from_saved_sources(tmp_path):
    cases=[]
    for i,split in enumerate(('train','validation')):
        save_volume(tmp_path/f't{i}.npz',np.ones((4,4,4),np.uint8),{'spacing_mm':[1,1,1],'origin_mm':[-1.5]*3,'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,8,8),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[8,8],'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':.5,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':8,'min_background':.01,'seed':8}
    out=tmp_path/'prep'; prepare_samples(path,out,method='A',profile='A-adapted-3view',params=p)
    rows=read_training_manifest(out/'dte.json','A','dte')
    train=PreparedDataset(rows,'train','A','dte',online_dte=True)
    val=PreparedDataset(rows,'validation','A','dte',online_dte=False)
    a,b=train[0]; c,d=train[0]
    assert not np.array_equal(a.numpy(),c.numpy())
    assert set(np.unique(b.numpy())) == set(np.unique(d.numpy()))
    a,_=val[0]; b,_=val[0]
    np.testing.assert_array_equal(a.numpy(),b.numpy())

def test_online_dte_resume_reproduces_uninterrupted_training(tmp_path):
    import torch
    from stent_repro.training.engine import train_prepared
    cases=[]
    for i,split in enumerate(('train','validation')):
        save_volume(tmp_path/f't{i}.npz',np.ones((4,4,4),np.uint8),{'spacing_mm':[1,1,1],'origin_mm':[-1.5]*3,'direction':np.eye(3).tolist()})
        save_projections(tmp_path/f'b{i}.npz',np.full((3,8,8),.3,np.float32),{'angles_deg':[0,30,60],'units':'line_integral'})
        cases.append({'id':str(i),'split':split,'model_ids':[f'm{i}'],'patient_id':f'p{i}','projection_role':'patient_without_tool','truth':f't{i}.npz','background':f'b{i}.npz','angles_deg':[0,30,60]})
    path=tmp_path/'raw.json'; path.write_text(json.dumps({'cases':cases}),encoding='utf8')
    p={'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[8,8],'attenuation_per_mm':.1,'photons':10000,'blur_sigma_px':.5,'scatter_fraction':0,'false_negative_rate':.1,'false_positive_rate':.01,'false_positive_scale':.2,'integration_step_mm':1,'patch_size':8,'min_background':.01,'seed':8}
    out=tmp_path/'prep'; prepare_samples(path,out,method='A',profile='A-adapted-3view',params=p)
    rows=read_training_manifest(out/'dte.json','A','dte')
    train=PreparedDataset(rows,'train','A','dte',online_dte=True); val=PreparedDataset(rows,'validation','A','dte')
    def model():
        torch.manual_seed(99)
        return torch.nn.Conv2d(1,1,1)
    kw=dict(method='A',stage='dte',batch_size=2,learning_rate=.01,seed=3,device='cpu')
    train_prepared(train,val,model(),tmp_path/'full',epochs=2,**kw)
    train_prepared(train,val,model(),tmp_path/'resume',epochs=1,**kw)
    train_prepared(train,val,model(),tmp_path/'resume',epochs=2,resume=tmp_path/'resume'/'last.pt',**kw)
    a=torch.load(tmp_path/'full'/'last.pt',weights_only=True); b=torch.load(tmp_path/'resume'/'last.pt',weights_only=True)
    for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
