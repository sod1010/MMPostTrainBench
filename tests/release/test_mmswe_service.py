"""Adversarial CPU checks. Fake backend tests NEVER certify container isolation."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from mmswe_service.artifacts import Rejected, freeze, validate_model
from mmswe_service.policy import validate_request, pinned_image
from mmswe_service.backend import DockerBackend, BackendFailure, BackendUnhealthy
from mmswe_service.service import Broker, Server
from mmswe_service.container_policy import inspect_security, security_args

spec = importlib.util.spec_from_file_location('docker_grade_cpu', ROOT/'src/mmswe_service/docker_grade.py')
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)


def model(path):
    path.mkdir(parents=True)
    (path/'config.json').write_text(json.dumps({'model_type':'qwen3_omni_moe'}))
    (path/'tokenizer_config.json').write_text('{}')
    h=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    (path/'model.safetensors').write_bytes(struct.pack('<Q',len(h))+h+b'1234')


def policy(root):
    image='example.invalid/mmptb@sha256:'+'a'*64
    profile={'timeout':30,'cpus':2,'memory_gib':8,'pids':64,'gpus':[], 'artifact_bytes':1024*1024}
    return {'state':str(root/'state'),'socket':str(root/'sockets/broker.sock'),
            'docker':'/usr/bin/docker','train_image':image,'generate_image':image,
            'grader_python':sys.executable,'cache':str(root/'cache'),'baseline':str(root/'base'),
            'runs':{'a':{'uid':1001,'workspace':str(root/'agent_a'),'budget_seconds':90},
                    'b':{'uid':1002,'workspace':str(root/'agent_b'),'budget_seconds':90}},
            'profiles':{'train':dict(profile),'score':dict(profile)},
            'dataset':'SWE-bench/SWE-bench_Multimodal','dev_limit':1,'test_limit':1,
            'image_manifest_sha256':'a'*64,'asset_manifest_sha256':'b'*64}


def inspected(kind='workload'):
    return {'Config': {'User': '65534:65534' if kind == 'workload' else '0:0'},
            'HostConfig': {'Privileged': False, 'CapAdd': [],
                           'CapDrop': ['ALL'] if kind == 'workload' else ['NET_RAW', 'MKNOD'],
                           'SecurityOpt': ['no-new-privileges=true'], 'NetworkMode': 'none',
                           'IpcMode': 'private', 'PidMode': '', 'UTSMode': '',
                           'ReadonlyRootfs': kind == 'workload'}, 'Mounts': []}


class FakeBackend:
    def __init__(self):self.sealed=[];self.fail=False;self.cleaned=False
    def cleanup(self):self.cleaned=True
    def score(self,job,directory,model,sealed,deadline):
        self.sealed.append(sealed)
        if self.fail:raise BackendFailure('secret test detail must stay private')
        return {'accuracy':0.0,'correct':0,'n':1,'split':'test' if sealed else 'dev',
                'predictions_sha256':'a'*64}
    def train(self,job,directory,code,data,deadline):
        model(directory/'output/model')
        return directory/'output/model'


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.config=policy(self.root)
        model(self.root/'agent_a/model');model(self.root/'agent_b/model')
        self.fake=FakeBackend();self.broker=Broker(self.config,self.fake)
    def submit(self,op,uid=1001,run='a',request_id='r1',**kw):
        req={'op':op,'run':run,'request_id':request_id,**kw}
        return self.broker.submit(req,uid)
    def imported(self):
        result=self.submit('import_model',model='model');self.broker.execute(result['id'])
        job=json.loads(self.broker.job_path(result['id']).read_text())
        self.assertEqual(job['status'],'succeeded')
        return job['artifact']
    def test_unknown_fields_and_role_spoofing_are_rejected(self):
        base={'op':'import_model','run':'a','request_id':'r','model':'model'}
        for name,value in [('command','touch /tmp/x'),('env',{'MMPTB_ROLE':'verifier'}),('role','verifier'),('uid',1001),('cwd','/')]:
            with self.subTest(name=name), self.assertRaises(Rejected):
                validate_request(dict(base,**{name:value}),1001,self.config)
        with self.assertRaises(Rejected):validate_request(base,1002,self.config)
        with self.assertRaises(Rejected):validate_request({'op':'seal','run':'a','request_id':'r','artifact':'x'},1001,self.config)
        with self.assertRaises(Rejected):validate_request(base,os.geteuid(),self.config,True)
    def test_snapshot_is_a_copy_and_binds_configuration(self):
        first=freeze(self.root/'agent_a','model',self.root/'copy1',1024*1024)
        src=self.root/'agent_a/model/model.safetensors';dest=self.root/'copy1/model.safetensors'
        self.assertNotEqual(src.stat().st_ino,dest.stat().st_ino)
        before=dest.read_bytes();src.write_bytes(b'changed')
        self.assertEqual(dest.read_bytes(),before)
        (self.root/'agent_a/model/tokenizer_config.json').write_text('{"x":1}')
        other=freeze(self.root/'agent_a','model',self.root/'copy2',1024*1024)
        self.assertNotEqual(first['sha256'],other['sha256'])
    def test_path_traversal_symlink_and_special_file_refused(self):
        for rel in ['../agent_b','/etc','model/../model','model//x']:
            with self.subTest(rel=rel),self.assertRaises((Rejected,OSError)):
                freeze(self.root/'agent_a',rel,self.root/'bad',1024)
        (self.root/'agent_a/link').symlink_to(self.root/'agent_b/model',target_is_directory=True)
        with self.assertRaises(OSError):freeze(self.root/'agent_a','link',self.root/'bad2',1024)
        (self.root/'agent_a/model/link').symlink_to('/etc/passwd')
        with self.assertRaises(Rejected):freeze(self.root/'agent_a','model',self.root/'bad3',1024*1024)
        (self.root/'agent_a/model/link').unlink()
        os.mkfifo(self.root/'agent_a/model/pipe')
        with self.assertRaises(Rejected):freeze(self.root/'agent_a','model',self.root/'bad4',1024*1024)
    def test_model_schema_rejects_code_pickle_and_bad_weights(self):
        p=self.root/'agent_a/model';validate_model(p)
        (p/'model.py').write_text('raise Exception')
        with self.assertRaises(Rejected):validate_model(p)
        (p/'model.py').unlink();(p/'config.json').write_text('{"model_type":"qwen3_omni_moe","auto_map":{"AutoModel":"custom.X"}}')
        with self.assertRaises(Rejected):validate_model(p)
        (p/'config.json').write_text('{"model_type":"qwen3_omni_moe"}')
        (p/'model.safetensors').write_bytes(b'broken')
        with self.assertRaises(Rejected):validate_model(p)
    def test_cross_run_artifact_and_job_reads_refused(self):
        artifact=self.imported()
        with self.assertRaises(Rejected):self.submit('score',uid=1002,run='b',artifact=artifact)
        jid=self.broker.jobs()[0]['id']
        with self.assertRaises(Rejected):self.broker.submit({'op':'status','run':'b','job':jid},1002)
    def test_idempotent_submission_does_not_charge_twice(self):
        a=self.submit('import_model',model='model');b=self.submit('import_model',model='model')
        self.assertEqual(a['id'],b['id']);self.assertEqual(len(self.broker.jobs()),1)
        with self.assertRaises(Rejected):self.submit('import_model',model='different')
    def test_reserved_budget_prevents_overbooking(self):
        self.config['runs']['a']['budget_seconds']=30
        self.submit('import_model',model='model')
        with self.assertRaises(Rejected):self.submit('import_model',request_id='r2',model='model')
    def test_zero_is_valid_failure_has_no_score_and_is_charged(self):
        artifact=self.imported();job=self.submit('score',request_id='score1',artifact=artifact)
        self.broker.execute(job['id']);j=json.loads(self.broker.job_path(job['id']).read_text())
        self.assertEqual(j['score']['accuracy'],0.0);self.assertGreater(j['charged_seconds'],0)
        self.fake.fail=True;job=self.submit('score',request_id='score2',artifact=artifact)
        self.broker.execute(job['id']);j=json.loads(self.broker.job_path(job['id']).read_text())
        self.assertEqual(j['status'],'failed');self.assertNotIn('score',j)
        self.assertNotIn('secret',json.dumps(self.broker.public(j,False)))
    def test_operator_sealing_closes_research_and_hides_results(self):
        artifact=self.imported()
        req={'op':'seal','run':'a','request_id':'sealed1','artifact':artifact}
        result=self.broker.submit(req,os.geteuid(),True);self.broker.execute(result['id'])
        self.assertEqual(self.fake.sealed,[True])
        with self.assertRaises(Rejected):self.submit('score',request_id='late',artifact=artifact)
        with self.assertRaises(Rejected):self.broker.submit({'op':'status','run':'a','job':result['id']},1001)
        r=self.broker.submit({'op':'status','run':'a','job':result['id']},os.geteuid(),True)
        self.assertEqual(r['score']['accuracy'],0.0)
    def test_cleanup_failure_blocks_further_admission(self):
        artifact=self.imported();result=self.submit('score',request_id='failcleanup',artifact=artifact)
        with patch.object(self.fake,'score',side_effect=BackendUnhealthy('cleanup failed')):
            self.broker.execute(result['id'])
        self.assertTrue((self.broker.state/'blocked.json').is_file())
        with self.assertRaises(Rejected):self.submit('score',request_id='after',artifact=artifact)
        self.broker.recover()
        self.assertFalse((self.broker.state/'blocked.json').exists())
    def test_snapshot_size_and_deadline_enforced(self):
        with self.assertRaises(Rejected):freeze(self.root/'agent_a','model',self.root/'small',1)
        with self.assertRaises(TimeoutError):freeze(self.root/'agent_a','model',self.root/'late',1024*1024,deadline=0)
        self.assertFalse((self.root/'small').exists())
        self.assertFalse((self.root/'late').exists())
    def test_directory_entries_count_towards_limit(self):
        p=self.root/'agent_a/directories';p.mkdir()
        for i in range(4):(p/str(i)).mkdir()
        with self.assertRaises(Rejected):freeze(self.root/'agent_a','directories',self.root/'bounded',1000,max_files=3)
        self.assertFalse((self.root/'bounded').exists())
    def test_invalid_model_leaves_no_registered_or_partial_artifact(self):
        (self.root/'agent_a/model/forbidden.py').write_text('pass')
        job=self.submit('import_model',model='model');self.broker.execute(job['id'])
        self.assertEqual(self.broker.jobs()[0]['status'],'failed')
        self.assertEqual(list((self.broker.state/'artifacts').iterdir()),[])
    def test_import_does_not_claim_gpu_consumption(self):
        self.config['profiles']['train']['gpus']=[0,1]
        self.imported();j=self.broker.jobs()[0]
        self.assertGreater(j['charged_seconds'],0)
        self.assertEqual(j['profile_gpu_seconds_upper_estimate'],0)
        self.assertNotIn('reserved_gpu_seconds',j)
    def test_restart_does_not_replay_jobs_or_lose_reserved_cost(self):
        self.submit('import_model',model='model');self.broker.recover()
        job=self.broker.jobs()[0];self.assertEqual(job['status'],'failed')
        self.assertEqual(job['charged_seconds'],30);self.assertTrue(self.fake.cleaned)
    def test_restart_refuses_changed_arm_identity_before_backend_cleanup(self):
        self.broker.recover()
        changed=json.loads(json.dumps(self.config));changed['runs']['a']['uid']=2001
        backend=FakeBackend();restarted=Broker(changed,backend)
        with self.assertRaisesRegex(Rejected,'policy changed'):restarted.recover()
        self.assertFalse(backend.cleaned)
    def test_restart_reuses_identical_policy_and_checks_legacy_jobs(self):
        self.imported();self.broker.recover()
        restarted=Broker(self.config,FakeBackend());restarted.recover()
        self.assertEqual(restarted.service_id,self.broker.service_id)
        # Removing the pin cannot relabel existing jobs with a different budget.
        (self.broker.state/'policy.json').unlink()
        changed=json.loads(json.dumps(self.config));changed['runs']['a']['budget_seconds']=999
        backend=FakeBackend()
        with self.assertRaisesRegex(Rejected,'historical jobs'):Broker(changed,backend).recover()
        self.assertFalse(backend.cleaned)
    def test_template_image_cannot_be_used_as_a_deployment_pin(self):
        for value in ['registry.example/train@sha256:'+'0'*64,'image:latest',None]:
            with self.subTest(value=value),self.assertRaises(Rejected):pinned_image(value)
        self.assertEqual(pinned_image(self.config['train_image']),self.config['train_image'])
        self.assertEqual(pinned_image('sha256:'+'a'*64),'sha256:'+'a'*64)
    def test_docker_workload_has_no_privilege_or_user_controlled_role(self):
        b=DockerBackend(self.config,'a'*32)
        args=b.create_args('b'*32,'train',[(self.root/'fixed','/input/code')],self.root/'out')
        for bad in ['--privileged','--cap-add','--pid=host','--network=host','/var/run/docker.sock']:
            self.assertNotIn(bad,args)
        self.assertIn('--read-only',args);self.assertIn('ALL',args)
        self.assertIn('65534:65534',args);self.assertIn('none',args)
        self.assertEqual(args[-2:],['/opt/mmptb-service/entry.sh','train'])
    def test_instance_uses_default_container_isolation_without_nested_privileges(self):
        a=grade.instance_args('name','sha256:'+'a'*64,'svc','job')
        for bad in ['--privileged','--cap-add','--mount','--volume','--pid']:
            self.assertNotIn(bad,a)
        self.assertEqual(a[-1],'/run_in_chroot.sh');self.assertIn('none',a)
    def test_wait_timeout_removes_container_and_preserves_failure(self):
        b=DockerBackend(self.config,'a'*32);calls=[]
        def cli(args,**kwargs):
            calls.append(args)
            if args[0]=='wait':raise subprocess.TimeoutExpired('docker wait',1)
            if args[0]=='inspect':return subprocess.CompletedProcess(args,0,json.dumps([inspected()]),'')
            return subprocess.CompletedProcess(args,0,'b'*64 if args[0]=='create' else '', '')
        with patch.object(b,'cli',side_effect=cli),self.assertRaises(subprocess.TimeoutExpired):
            b.container('b'*32,'train',[],self.root/'output',time.monotonic()+10)
        self.assertEqual(calls[-1],['rm','-f','b'*64])
    def test_create_timeout_cleanup_failure_is_unhealthy(self):
        b=DockerBackend(self.config,'a'*32);calls=[]
        def cli(args,**kwargs):
            calls.append(args)
            if args[0]=='create':raise subprocess.TimeoutExpired('docker create',1)
            return subprocess.CompletedProcess(args,1,'','authorization denied')
        with patch.object(b,'cli',side_effect=cli),self.assertRaises(BackendUnhealthy):
            b.container('b'*32,'train',[],self.root/'output',time.monotonic()+10)
        self.assertEqual(calls[-1],['rm','-f','mmptb-'+'a'*12+'-'+'b'*32])
    def test_managed_profile_requires_equivalent_policy_before_workload_start(self):
        self.config['container_profile']='platform-managed-v1'
        b=DockerBackend(self.config,'a'*32);calls=[]
        info=inspected();info['HostConfig'].update(SecurityOpt=None,CapDrop=None)
        def cli(args,**kwargs):
            calls.append(args)
            value=json.dumps([info]) if args[0]=='inspect' else 'b'*64
            return subprocess.CompletedProcess(args,0,value,'')
        with patch.object(b,'cli',side_effect=cli),self.assertRaisesRegex(BackendFailure,'not started'):
            b.container('b'*32,'train',[],self.root/'managed_output',time.monotonic()+10)
        self.assertNotIn('start',[c[0] for c in calls])
        self.assertEqual([c[0] for c in calls].count('create'),1)
        self.assertNotIn('--cap-drop',calls[0])
        self.assertEqual(calls[-1][:2],['rm','-f'])
        receipt=json.loads((self.root/('b'*32+'-container-security.json')).read_text())
        self.assertFalse(receipt['compliant'])
    def test_instance_security_mismatch_cleans_up_before_official_tests(self):
        rootfs=self.root/'staged/rootfs';rootfs.mkdir(parents=True)
        (rootfs/'fixture').write_text('fixture');calls=[]
        info=inspected('instance');info['HostConfig']['NetworkMode']='host'
        def cli(args,**kwargs):
            calls.append(args)
            result={'import':b'sha256:'+b'a'*64,'create':'b'*64,'inspect':json.dumps([info])}.get(args[0],'')
            return subprocess.CompletedProcess(args,0,result,'')
        with patch.object(grade,'cli',side_effect=cli),self.assertRaisesRegex(RuntimeError,'not started'):
            grade._run_in_docker(rootfs,io.StringIO(),1)
        self.assertNotIn('start',[c[0] for c in calls])
        self.assertEqual([c[0] for c in calls][-2:],['rm','image'])
    def test_legacy_environment_switch_cannot_silently_weaken_grader(self):
        with patch.dict(os.environ,{'MMSWE_DOCKER_HARDENING':'0'}),patch.object(grade,'cli') as cli:
            with self.assertRaisesRegex(RuntimeError,'unsupported'):
                grade._run_in_docker(self.root,io.StringIO(),1)
            cli.assert_not_called()
    def test_unsupported_network_fails_before_native_preparation(self):
        for network in ['bridge','host','container:other','', 'named-net']:
            with self.subTest(network=network),patch.dict(os.environ,{'MMSWE_DOCKER_NETWORK':network},clear=True),patch.object(grade.native,'main') as native:
                with self.assertRaisesRegex(RuntimeError,'MMSWE_DOCKER_NETWORK'):grade.main()
                native.assert_not_called()
        with patch.dict(os.environ,{'MMSWE_DOCKER_NETWORK':'none'},clear=True),patch.object(grade.native,'main',return_value=0) as native:
            self.assertEqual(grade.main(),0);native.assert_called_once()
    def test_legacy_hardening_fails_before_native_preparation(self):
        with patch.dict(os.environ,{'MMSWE_DOCKER_HARDENING':'platform'},clear=True),patch.object(grade.native,'main') as native:
            with self.assertRaisesRegex(RuntimeError,'unsupported'):grade.main()
            native.assert_not_called()
    def test_grader_cleanup_attempts_both_resources_and_blocks_next_instance(self):
        rootfs=self.root/'scratch/rootfs';rootfs.mkdir(parents=True)
        (rootfs/'fixture').write_text('test')
        grade.BLOCKED.clear();self.addCleanup(grade.BLOCKED.clear)
        calls=[]
        def cli(args,**kwargs):
            calls.append(args)
            if args[0]=='import':raise subprocess.TimeoutExpired('docker import',1)
            if args[0]=='rm':raise subprocess.TimeoutExpired('docker rm',1)
            return subprocess.CompletedProcess(args,0,'','')
        with patch.object(grade,'cli',side_effect=cli):
            with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
                grade.run_in_docker(rootfs,io.StringIO(),1)
            count=len(calls)
            with self.assertRaisesRegex(RuntimeError,'blocked'):
                grade.run_in_docker(rootfs,io.StringIO(),1)
            self.assertEqual(len(calls),count)
        self.assertEqual(calls[-1][:3],['image','rm','-f'])
        self.assertFalse((rootfs.parent/'docker-rootfs.tar').exists())
    def test_host_log_copy_rejects_symlink_and_multiple_members(self):
        def archive(members):
            out=io.BytesIO()
            with tarfile.open(fileobj=out,mode='w') as t:
                for member in members:t.addfile(member,io.BytesIO(b'x') if member.isfile() else None)
            return out.getvalue()
        f=tarfile.TarInfo('test_output.txt');f.size=1
        self.assertEqual(grade.extract_log(archive([f])),b'x')
        link=tarfile.TarInfo('escape');link.type=tarfile.SYMTYPE;link.linkname='/etc/passwd'
        with self.assertRaises(RuntimeError):grade.extract_log(archive([link]))
        with self.assertRaises(RuntimeError):grade.extract_log(archive([f,f]))
    @unittest.skipUnless(hasattr(socket,'SO_PEERCRED'),'Linux actual peer credential test')
    def test_actual_unix_socket_does_not_trust_claimed_uid(self):
        # Connection is made by this process, not policy UID 1001. Merely writing
        # run=a must not grant its authority, even on a mode-0666 socket.
        sock=self.root/'broker.sock';server=Server(str(sock),self.broker,False)
        self.addCleanup(server.server_close)
        thread=threading.Thread(target=server.handle_request);thread.start()
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.connect(str(sock));client.sendall(b'{"op":"import_model","run":"a","request_id":"x","model":"model"}\n')
            data=json.loads(client.makefile('rb').readline())
        thread.join(5);self.assertFalse(data['ok']);self.assertEqual(self.broker.jobs(),[])


class ContainerPolicyTests(unittest.TestCase):
    def test_profiles_accept_only_equivalent_daemon_policy(self):
        for kind in ('instance','workload'):
            for profile in ('strict-v1','platform-managed-v1'):
                self.assertTrue(inspect_security(inspected(kind),kind,profile)['compliant'])
                for key,value in [('Privileged',True),('CapAdd',['SYS_ADMIN']),('SecurityOpt',[]),
                                  ('CapDrop',[]),('NetworkMode','host'),('IpcMode','host')]:
                    info=inspected(kind);info['HostConfig'][key]=value
                    with self.subTest(kind=kind,profile=profile,key=key):
                        self.assertFalse(inspect_security(info,kind,profile)['compliant'])
        with self.assertRaises(ValueError):security_args('instance','off')
        self.assertEqual(security_args('workload','platform-managed-v1'),[])
    def test_unconfined_and_host_mounts_are_not_accepted(self):
        info=inspected('instance');info['HostConfig']['SecurityOpt'].append('seccomp=unconfined')
        self.assertFalse(inspect_security(info,'instance')['compliant'])
        info=inspected('instance');info['Mounts']=[{'Source':'/var/run/docker.sock'}]
        self.assertFalse(inspect_security(info,'instance')['compliant'])


if __name__=='__main__':unittest.main()
