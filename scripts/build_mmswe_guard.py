#!/usr/bin/env python3
"""Build the fixed guard with an existing Linux toolchain; never downloads tools."""
import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mmswe_service.runtime_guard import load_guard
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',required=True,help='new operator-owned output directory')
p.add_argument('--cc',default='gcc');a=p.parse_args()
os.umask(0o077);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
source=ROOT/'src/mmswe_service/runtime_guard.c';binary=out/'guard'
cmd=[a.cc,'-static','-O2','-std=c11','-Wall','-Wextra','-Werror','-fstack-protector-strong','-D_FORTIFY_SOURCE=2','-Wl,-z,relro,-z,now','-o',str(binary),str(source)]
subprocess.run(cmd,check=True,timeout=60);binary.chmod(0o555)
pin={'path':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
load_guard(pin)
receipt={'pin':pin,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'compiler':subprocess.check_output([a.cc,'--version'],text=True).splitlines()[0],'command':cmd}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
