import json
import nibabel as nib
import numpy as np
import pytest
import torch
from phase_3.tests.fdk_fixtures import synthetic_dataset


def write_smoke(path,name='A'):
    from phase_3.fdk_unet.config import get_experiment_config
    r=get_experiment_config(name).resolved
    r.update(run_kind='smoke',experiment_id=name+'-smoke',patch_size=[32]*3,max_optimizer_steps=2,validate_every=1)
    path.write_text(json.dumps(r));return path


def test_cli_synthetic_train_predict_evaluate_resume(tmp_path):
    from phase_3.fdk_unet.cli import main
    mp,sp=synthetic_dataset(tmp_path/'data',shape=(33,35,37));cfg=write_smoke(tmp_path/'config.json');run=tmp_path/'run'
    common=['--manifest',str(mp),'--splits',str(sp),'--fold','0']
    assert main(['validate-data',*common,'--smoke'])==0
    assert main(['train','--config',str(cfg),*common,'--seed','2027','--device','cpu','--output',str(run)])==0
    assert main(['train','--resume',str(run/'last.pt'),'--device','cpu'])==0
    assert main(['predict','--checkpoint',str(run/'best.pt'),*common,'--split','test','--device','cpu','--output',str(run/'predictions')])==0
    assert main(['evaluate',*common,'--split','test','--predictions',str(run/'predictions'),'--output',str(run/'evaluation')])==0
    assert (run/'config.resolved.json').is_file() and (run/'evaluation/summary.json').is_file()
    assert json.loads((run/'config.resolved.json').read_text())['seed']==2027
    actual=nib.load(run/'predictions/case_3_mask.nii.gz');original=nib.load(tmp_path/'data/case_3_0000.nii.gz')
    assert actual.shape==original.shape;np.testing.assert_array_equal(actual.affine,original.affine)
    assert main(['train','--config',str(cfg),*common,'--device','cpu','--output',str(run)])==2


def test_predict_unlabeled_manifest(tmp_path):
    from phase_3.fdk_unet.cli import main
    mp,sp=synthetic_dataset(tmp_path/'data');cfg=write_smoke(tmp_path/'config.json');run=tmp_path/'run'
    assert main(['train','--config',str(cfg),'--manifest',str(mp),'--splits',str(sp),'--fold','0','--device','cpu','--output',str(run)])==0
    raw=json.loads(mp.read_text());raw['cases']=raw['cases'][:1];raw['cases'][0].pop('label');mp.write_text(json.dumps(raw))
    assert main(['predict','--checkpoint',str(run/'last.pt'),'--manifest',str(mp),'--device','cpu','--output',str(run/'unlabeled')])==0


def test_experimental_prediction_rejects_checkpoint_from_another_fold(tmp_path):
    from phase_3.fdk_unet.cli import main
    mp,sp=synthetic_dataset(tmp_path/'data');cfg=write_smoke(tmp_path/'config.json');run=tmp_path/'run'
    assert main(['train','--config',str(cfg),'--manifest',str(mp),'--splits',str(sp),'--fold','0','--device','cpu','--output',str(run)])==0
    raw=json.loads(sp.read_text());raw['folds']['1']={'train':['g2','g3'],'validation':['g1'],'test':['g0']};sp.write_text(json.dumps(raw))
    out=tmp_path/'wrong_fold'
    assert main(['predict','--checkpoint',str(run/'best.pt'),'--manifest',str(mp),'--splits',str(sp),'--fold','1','--split','test','--device','cpu','--output',str(out)])==2
    assert not out.exists()


def test_evaluation_checks_saved_prediction_fold_provenance(tmp_path):
    from phase_3.fdk_unet.cli import main
    mp,sp=synthetic_dataset(tmp_path/'data');cfg=write_smoke(tmp_path/'config.json');run=tmp_path/'run'
    common=['--manifest',str(mp),'--splits',str(sp),'--fold','0']
    assert main(['train','--config',str(cfg),*common,'--device','cpu','--output',str(run)])==0
    assert main(['predict','--checkpoint',str(run/'best.pt'),*common,'--split','test','--device','cpu','--output',str(run/'predictions')])==0
    provenance=run/'predictions/prediction_run.json'
    raw=json.loads(provenance.read_text()) if provenance.exists() else {}
    raw['split_digest']='different-fold';provenance.write_text(json.dumps(raw))
    assert main(['evaluate',*common,'--split','test','--predictions',str(run/'predictions'),'--output',str(run/'evaluation')])==2


@pytest.mark.monai
def test_six_core_structures_cli_pipeline(tmp_path):
    pytest.importorskip('monai')
    from phase_3.fdk_unet.cli import main
    from phase_3.fdk_unet.swin_prior import SwinPriorEncoder
    mp,sp=synthetic_dataset(tmp_path/'data');common=['--manifest',str(mp),'--splits',str(sp),'--fold','0']
    weights=tmp_path/'synthetic_ssl.pt';torch.save(SwinPriorEncoder(execute_through='layers3').swinViT.state_dict(),weights)
    for init in ('ssl','random'):
        args=['prepare-prior',*common,'--smoke','--init',init,'--output',str(tmp_path/init)]
        if init=='ssl':args+=['--weights',str(weights)]
        assert main(args)==0
    for name in ('A','B','H00','H10','H01','C'):
        cfg=write_smoke(tmp_path/(name+'.json'),name);run=tmp_path/name
        args=['train','--config',str(cfg),*common,'--device','cpu','--output',str(run),'--initialization-artifact',str(tmp_path/'shared.pt')]
        if name!='A':args+=['--prior-artifact',str(tmp_path/('random' if name=='B' else 'ssl')/'prior.json')]
        assert main(args)==0
        assert main(['predict','--checkpoint',str(run/'best.pt'),*common,'--split','test','--device','cpu','--output',str(run/'predictions')])==0
