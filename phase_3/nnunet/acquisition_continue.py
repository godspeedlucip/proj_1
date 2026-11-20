"""Start formal Phase 3 training only after every arm passes the CUDA smoke gate."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ARMS=('a60_n3','a45_n3','a75_n3','a60_n2','a60_n5')


def assess(root):
    root=Path(root)
    status_file=root/'supervisor_status.json'
    if not status_file.exists():return 'waiting'
    status=json.loads(status_file.read_text())
    if status.get('stop_after')!='smoke':
        raise ValueError('Expected a smoke-only supervisor')
    workers=status.get('workers',{})
    if set(workers)!=set(ARMS):raise ValueError('Smoke supervisor did not cover all five arms')
    if any(row.get('returncode') not in (None,0) for row in workers.values()):return 'failed'
    if any(row.get('returncode') is None for row in workers.values()):return 'waiting'
    for arm in ARMS:
        passed=root/arm/'smoke/passed.json'
        if not passed.is_file() or json.loads(passed.read_text()).get('passed') is not True:
            return 'waiting'
    return 'ready'


def record(path,value):
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(dict(value,updated_unix=time.time()),indent=2))
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--check-once',action='store_true')
    args=parser.parse_args()
    root=args.run_dir.resolve()
    if args.check_once:
        print(assess(root));return
    lock=root/'formal_continuation.lock'
    with lock.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Formal continuation already active') from exc
        (root/'formal_continuation.pid').write_text(str(os.getpid())+'\n')
        status_path=root/'formal_continuation_status.json'
        while True:
            decision=assess(root)
            record(status_path,dict(state=decision))
            if decision=='failed':raise RuntimeError('Smoke arm failed; formal training withheld')
            if decision=='ready':break
            time.sleep(30)
        info=json.loads((root/'run.json').read_text())
        entry=Path(info['code_root'])/'acquisition_launch.py'
        cmd=[info['phase3_python'],'-u',str(entry),'run','--run-dir',str(root),
             '--gpus','0','1','2','4','5']
        prior=root/'formal_supervisor.pid'
        if prior.exists():
            pid=int(prior.read_text());proc=Path(f'/proc/{pid}/cmdline')
            if proc.exists() and 'acquisition_launch.py' in proc.read_text():
                raise RuntimeError('Formal supervisor is already active')
        with (root/'formal_supervisor.log').open('a') as log:
            child=subprocess.Popen(cmd,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        prior.write_text(str(child.pid)+'\n')
        record(status_path,dict(state='formal_running',pid=child.pid,command=cmd,
                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        code=child.wait()
        record(status_path,dict(state='complete' if code==0 else 'failed',pid=child.pid,returncode=code))
        if code:raise RuntimeError('Formal experiment failed; inspect arm status and logs')


if __name__=='__main__':
    main()
