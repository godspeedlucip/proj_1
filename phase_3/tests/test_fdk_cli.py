import json
import pytest
from phase_3.tests.fdk_fixtures import synthetic_dataset


def test_prepare_prior_requires_explicit_download(tmp_path,monkeypatch):
    from phase_3.fdk_unet.cli import main
    import urllib.request
    def forbidden(*a,**k):raise AssertionError('implicit network')
    monkeypatch.setattr(urllib.request,'urlretrieve',forbidden)
    mp,sp=synthetic_dataset(tmp_path/'data')
    common=['--manifest',str(mp),'--splits',str(sp),'--fold','0','--smoke']
    assert main(['prepare-prior',*common,'--output',str(tmp_path/'bad')])==2
    assert main(['prepare-prior',*common,'--init','random','--output',str(tmp_path/'random')])==0
    r=json.loads((tmp_path/'random/prior.json').read_text())
    assert r['weights'] is None and r['sha256'] is None and r['init']=='random'
    assert r['stats']['sample_count']==131072


def test_resume_rejects_conflicting_cli_overrides():
    from phase_3.fdk_unet.cli import main
    assert main(['train','--resume','missing.pt','--seed','2027','--device','cpu'])==2
    assert main(['train','--resume','missing.pt','--config','new.json','--device','cpu'])==2
