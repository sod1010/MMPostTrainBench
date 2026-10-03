"""Protocol/launch checks and real Linux bootstrap tests (no candidate models)."""
import copy
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from mmswe_service import runtime_guard as guard
from mmswe_service.container_policy import inspect_security

NONCE='c'*64
IMAGE='sha256:'+'a'*64

def attestation(kind='workload'):
    return {'protocol':1,'kind':kind,'purpose':'probe','nonce':NONCE,
            'uid':65534 if kind=='workload' else 0,'gid':65534 if kind=='workload' else 0,
            'groups':0,'nnp':1,'caps':{k:'0'*16 for k in ['eff','prm','inh','bnd','amb']}}

def launch():
    return {'Image':IMAGE,'Config':{'Image':IMAGE,'User':'0:0','Entrypoint':[guard.GUARD_PATH],
        'Cmd':['workload','probe',NONCE],'OpenStdin':True,'Tty':False},
        'HostConfig':{'Privileged':False,'CapAdd':None,'SecurityOpt':None,'CapDrop':None,
                     'NetworkMode':'none','IpcMode':'private','PidMode':'','UTSMode':'','ReadonlyRootfs':True},'Mounts':[]}

class GuardPolicyTests(unittest.TestCase):
    def test_explicit_guard_profile_requires_fixed_entry(self):
        good=launch()
        self.assertTrue(inspect_security(good,'workload',guard.PROFILE)['compliant'])
        self.assertFalse(inspect_security(good,'workload','platform-managed-v1')['compliant'])
        guard.validate_launch(good,'workload','probe',NONCE,IMAGE)
        for key,val in [('Entrypoint',['/bin/sh']),('Cmd',['workload','probe','d'*64]),('OpenStdin',False),('Tty',True)]:
            info=copy.deepcopy(good);info['Config'][key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):guard.validate_launch(info,'workload','probe',NONCE,IMAGE)
        info=copy.deepcopy(good);info['Mounts']=[{'Destination':'/opt'}]
        with self.assertRaises(ValueError):guard.validate_launch(info,'workload','probe',NONCE,IMAGE)
        with self.assertRaises(ValueError):guard.validate_launch(good,'workload','generate',NONCE,IMAGE)
        for key,val in [('Privileged',True),('CapAdd',['SYS_ADMIN']),('NetworkMode','bridge'),('IpcMode','host')]:
            info=copy.deepcopy(good);info['HostConfig'][key]=val
            self.assertFalse(inspect_security(info,'workload',guard.PROFILE)['compliant'])
    def test_attestation_rejects_weak_identity_caps_or_replay(self):
        good=attestation();guard.validate_attestation(good,'workload','probe',NONCE)
        for key,value in [('uid',0),('gid',0),('groups',1),('nnp',0),('nonce','d'*64),('protocol',True)]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):guard.validate_attestation(bad,'workload','probe',NONCE)
        for kind,cap in [('workload',1),('instance',1<<13),('instance',1<<27),('instance',1<<21)]:
            bad=attestation(kind);bad['caps']['bnd']=f'{cap:016x}'
            with self.assertRaises(ValueError):guard.validate_attestation(bad,kind,'probe',NONCE)
    def test_missing_pin_and_unsupported_purpose_fail_closed(self):
        with self.assertRaises(ValueError):guard.guard_from_environment({},guard.PROFILE)
        with self.assertRaises(ValueError):guard.guard_from_environment({guard.PIN_ENV:'/tmp/x'},'strict-v1')
        with self.assertRaises(ValueError):guard.command('instance','train',NONCE)
        with self.assertRaises(ValueError):guard.command('workload','generate',None)

@unittest.skipUnless(sys.platform=='linux' and shutil.which('gcc'),'Linux static compiler needed')
class NativeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.binary=cls.root/'guard'
        subprocess.run(['gcc','-static','-O2','-Wall','-Wextra','-Werror','-o',str(cls.binary),str(ROOT/'src/mmswe_service/runtime_guard.c')],check=True,capture_output=True)
        cls.pin={'path':str(cls.binary),'sha256':hashlib.sha256(cls.binary.read_bytes()).hexdigest()}
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_pin_static_elf_and_no_symlink_staging(self):
        with patch('mmswe_service.policy.private_file',side_effect=Path):
            guard.load_guard(self.pin)
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);guard.stage_guard(root,self.pin)
                self.assertEqual(hashlib.sha256((root/guard.GUARD_PATH.lstrip('/')).read_bytes()).hexdigest(),self.pin['sha256'])
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);(root/'opt').symlink_to(self.root,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'unsafe'):guard.stage_guard(root,self.pin)
            with self.assertRaisesRegex(ValueError,'bytes changed'):guard.load_guard({**self.pin,'sha256':'0'*64})
    @unittest.skipUnless(os.geteuid()==0,'root child process needed; parent privileges are unchanged')
    def test_no_probe_exec_without_host_ack_and_wrong_ack_is_rejected(self):
        p=subprocess.Popen([str(self.binary),'workload','probe',NONCE],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            self.assertTrue(select.select([p.stdout],[],[],5)[0])
            ready=json.loads(p.stdout.readline());guard.validate_attestation(ready,'workload','probe',NONCE)
            self.assertEqual(select.select([p.stdout],[],[],.1)[0],[])
            stdout,stderr=p.communicate(b'WRONG\n',timeout=5)
            self.assertEqual(p.returncode,78);self.assertNotIn(b'PROBE_FINISHED',stdout)
            self.assertIn(b'approval-mismatch',stderr)
        finally:
            if p.poll() is None:p.kill();p.wait()
            for s in (p.stdin,p.stdout,p.stderr):s.close()
    @unittest.skipUnless(os.geteuid()==0,'root child process needed; parent privileges are unchanged')
    def test_restrictions_survive_exec_and_su(self):
        for kind in ['workload','instance']:
            p=subprocess.Popen([str(self.binary),kind,'probe',NONCE],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                self.assertTrue(select.select([p.stdout],[],[],5)[0])
                ready=json.loads(p.stdout.readline());guard.validate_attestation(ready,kind,'probe',NONCE)
                stdout,stderr=p.communicate(('GO '+NONCE+'\n').encode(),timeout=10)
                self.assertEqual(p.returncode,0,stderr);self.assertIn(b'MMPTB_GUARD_PROBE_FINISHED',stdout)
                self.assertIn(b'uid=65534',stdout);self.assertIn(b'NoNewPrivs:\t1',stdout)
            finally:
                if p.poll() is None:p.kill();p.wait()
                for s in (p.stdin,p.stdout,p.stderr):s.close()

if __name__=='__main__':unittest.main()
