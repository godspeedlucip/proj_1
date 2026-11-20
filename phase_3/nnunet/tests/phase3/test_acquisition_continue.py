import json

from acquisition_continue import assess


def test_formal_starts_only_after_all_five_smoke_runs_succeed(tmp_path):
    names=['a60_n3','a45_n3','a75_n3','a60_n2','a60_n5']
    for name in names:
        path=tmp_path/name/'smoke';path.mkdir(parents=True)
        (path/'passed.json').write_text('{"passed": true}')
    status=dict(stop_after='smoke',workers={name:dict(returncode=0) for name in names})
    (tmp_path/'supervisor_status.json').write_text(json.dumps(status))
    assert assess(tmp_path)=='ready'
    (tmp_path/names[0]/'smoke/passed.json').unlink()
    assert assess(tmp_path)=='waiting'
    status['workers'][names[0]]['returncode']=1
    (tmp_path/'supervisor_status.json').write_text(json.dumps(status))
    assert assess(tmp_path)=='failed'
