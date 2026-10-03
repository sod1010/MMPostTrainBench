#!/usr/bin/env python3
"""CPU-only real guard/exec/su probe. Does not certify browser or model acceptance."""
import argparse,json,secrets,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mmswe_service import runtime_guard as guard
from mmswe_service.container_policy import inspect_security
from mmswe_service.policy import pinned_image,private_file


def run(a):
    pin=guard.load_guard({'path':a.guard,'sha256':a.guard_sha256});private_file(a.docker)
    pinned_image(a.image)
    out=Path(a.out).resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
    home=out/'docker-home';home.mkdir(mode=0o700)
    nonce=secrets.token_hex(32);name='mmptb-guard-probe-'+nonce[:24]
    def cli(args):
        return subprocess.run([a.docker,*args],env=guard.docker_env(home),capture_output=True,text=True,timeout=30)
    result={'profile':guard.PROFILE,'kind':a.kind,'image':a.image,'guard':pin,
            'official_acceptance':False,'gpu_count':0,'status':'pending'}
    attempted=False
    try:
        args=['create','--name',name,'--pull','never','--interactive','--init','--user','0:0',
              '--network','none','--read-only','--pids-limit','128','--memory','1g','--cpus','1',
              '--log-driver','local','--entrypoint',guard.GUARD_PATH,a.image,*guard.command(a.kind,'probe',nonce)]
        attempted=True;p=cli(args)
        if p.returncode:raise RuntimeError('create rejected: '+p.stderr[:600])
        info=json.loads(cli(['inspect',name]).stdout)[0]
        result['security']=inspect_security(info,a.kind,guard.PROFILE)
        if not result['security']['compliant']:raise ValueError('static policy mismatch')
        guard.validate_launch(info,a.kind,'probe',nonce,a.image)
        guard.verify_container_guard(a.docker,home,name,pin)
        result['runtime']=guard.guarded_start(a.docker,home,name,a.kind,'probe',nonce,out/'runtime.log',time.monotonic()+60)
        wait=cli(['wait',name]);result['exit_code']=wait.stdout.strip()
        body=(out/'runtime.log').read_text(errors='replace')
        if wait.returncode or result['exit_code']!='0' or result['runtime']['attach_returncode'] or 'MMPTB_GUARD_PROBE_FINISHED' not in body:
            raise RuntimeError('guard runtime probe failed')
        result['status']='guard_probe_passed_not_official_acceptance'
    except Exception as error:
        result.update(status='blocked',error=str(error)[:1200])
    finally:
        if attempted:
            try:
                p=cli(['rm','-f',name]);result['cleanup_ok']=p.returncode==0 or 'No such container' in p.stderr
            except Exception:result['cleanup_ok']=False
            if not result['cleanup_ok']:result['status']='cleanup_failed'
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return 0 if result['status']=='guard_probe_passed_not_official_acceptance' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['docker','image','guard','guard-sha256','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--kind',choices=['workload','instance'],required=True)
    raise SystemExit(run(p.parse_args()))
