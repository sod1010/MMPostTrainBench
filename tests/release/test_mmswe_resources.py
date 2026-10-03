"""CPU checks for resource propagation; no claim of runtime acceptance."""
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from mmswe_service.resources import (instance_resources, resources_from_environment,
    resource_environment, inspect_resources, RESOURCE_ENV)
from mmswe_service.backend import DockerBackend
from mmswe_service import policy

spec = importlib.util.spec_from_file_location('resource_grade', ROOT/'src/mmswe_service/docker_grade.py')
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)


class ResourceTests(unittest.TestCase):
    def test_legacy_resource_aliases_are_bounded_and_json_cannot_conflict(self):
        r = resources_from_environment({'MMSWE_DOCKER_PIDS':'8192','MMSWE_DOCKER_MEM':'32g'})
        self.assertEqual(r, {'cpus':8,'pids':8192,'memory_gib':32})
        self.assertEqual(resources_from_environment(resource_environment(r)), r)
        for key, values in [('MMSWE_DOCKER_PIDS',['0','-1','65537','1.5','true']),
                            ('MMSWE_DOCKER_MEM',['0','0g','-1g','32g --privileged','1025g','32'])]:
            for value in values:
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    resources_from_environment({key:value})
        for value in [[], {}, {'cpus':True,'memory_gib':32,'pids':8192},
                      {'cpus':8,'memory_gib':32,'pids':8192,'network':'bridge'}]:
            with self.subTest(value=value), self.assertRaises(ValueError):instance_resources(value)
        with self.assertRaisesRegex(ValueError,'conflicting'):
            resources_from_environment({**resource_environment(r),'MMSWE_DOCKER_MEM':'32g'})

    def test_invalid_limits_stop_before_expensive_native_preparation(self):
        with patch.dict(os.environ, {'MMSWE_DOCKER_PIDS':'-1'}, clear=True), patch.object(grade.native,'main') as main:
            with self.assertRaises(ValueError):grade.main()
            main.assert_not_called()

    def test_expected_docker_limits_and_receipt(self):
        r={'cpus':8,'memory_gib':32,'pids':8192}
        args=grade.instance_args('name','sha256:'+'a'*64,'svc','job',resources=r)
        for flag,value in [('--pids-limit','8192'),('--memory','32g'),('--memory-swap','32g'),('--cpus','8')]:
            self.assertEqual(args[args.index(flag)+1],value)
        info={'HostConfig':{'NanoCpus':8*10**9,'Memory':32*1024**3,
                            'MemorySwap':32*1024**3,'PidsLimit':8192}}
        self.assertTrue(inspect_resources(info,r)['compliant'])
        for key in info['HostConfig']:
            bad={'HostConfig':dict(info['HostConfig'],**{key:0})}
            self.assertFalse(inspect_resources(bad,r)['compliant'])

    def test_daemon_ignoring_pid_limit_is_rejected_before_start_and_cleaned(self):
        with tempfile.TemporaryDirectory() as td:
            rootfs=Path(td)/'rootfs';rootfs.mkdir();(rootfs/'fixture').write_text('x')
            calls=[]
            def cli(args,**kw):
                calls.append(args)
                value={'import':b'sha256:'+b'a'*64,'create':'b'*64,
                       'inspect':json.dumps([{'HostConfig':{'PidsLimit':0}}])}.get(args[0],'')
                return subprocess.CompletedProcess(args,0,value,'')
            log=io.StringIO()
            with patch.dict(os.environ,{},clear=True),patch.object(grade,'cli',side_effect=cli),patch.object(grade,'inspect_security',return_value={'compliant':True}):
                with self.assertRaisesRegex(RuntimeError,'resource policy mismatch'):
                    grade._run_in_docker(rootfs,log,1)
            self.assertNotIn('start',[c[0] for c in calls])
            self.assertEqual([c[0] for c in calls][-2:],['rm','image'])
            self.assertIn('container_resources=',log.getvalue())

    def test_broker_uses_frozen_policy_and_not_parent_resource_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();cache=root/'cache';job=root/'job';job.mkdir()
            digest=hashlib.sha256(b'{}').hexdigest()
            for name in ['images','assets']:
                (cache/name).mkdir(parents=True);(cache/name/'manifest.json').write_bytes(b'{}')
            config={'state':str(root),'cache':str(cache),'docker':'/fake/docker',
                    'grader_python':sys.executable,'dataset':'SWE-bench/SWE-bench_Multimodal',
                    'dev_limit':1,'test_limit':1,'asset_manifest_sha256':digest,
                    'image_manifest_sha256':digest,
                    'instance_resources':{'cpus':8,'memory_gib':32,'pids':8192}}
            backend=DockerBackend(config,'a'*32)
            def generation(jid,mode,mounts,out,deadline):
                out.mkdir();(out/'predictions.jsonl').write_text('{"instance_id":"dev-1"}\n')
                (out/'meta.json').write_text(json.dumps({'split':'dev','eval_split':'val','n_generation_failed':0,'instance_ids':['dev-1'],'evaluation_contract_sha256':'c'*64}))
            with patch.dict(os.environ,{'MMSWE_DOCKER_MEM':'999g'}),patch.object(backend,'container',side_effect=generation),patch('mmswe_service.backend.score_contract',return_value={'contract_sha256':'c'*64,'instance_ids':['dev-1'],'host_dataset_file':'/fake/dev.arrow','container_dataset_file':'/input/hf/dev.arrow','dataset_file_sha256':'d'*64}),patch.object(backend,'cli',return_value=subprocess.CompletedProcess([],0,'','')),patch('mmswe_service.backend.subprocess.Popen',side_effect=RuntimeError('captured grading launch')) as launch:
                with self.assertRaisesRegex(RuntimeError,'captured grading launch'):
                    backend.score('job',job,root/'model',False,time.monotonic()+10)
            env=launch.call_args.kwargs['env']
            self.assertEqual(json.loads(env[RESOURCE_ENV]),config['instance_resources'])
            self.assertNotIn('MMSWE_DOCKER_MEM',env)

            # Same denominator is insufficient: altered predictions must never
            # reach the official grader, even when metadata names the right IDs.
            def wrong_generation(jid,mode,mounts,out,deadline):
                generation(jid,mode,mounts,out,deadline)
                (out/'predictions.jsonl').write_text('{"instance_id":"different-1"}\n')
            other=root/'wrong-job';other.mkdir()
            with patch.object(backend,'container',side_effect=wrong_generation),patch('mmswe_service.backend.score_contract',return_value={'contract_sha256':'c'*64,'instance_ids':['dev-1'],'host_dataset_file':'/fake/dev.arrow','container_dataset_file':'/input/hf/dev.arrow','dataset_file_sha256':'d'*64}),patch('mmswe_service.backend.subprocess.Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'frozen evaluation identity'):
                    backend.score('wrong-job',other,root/'model',False,time.monotonic()+10)
                launch.assert_not_called()

    def test_operator_policy_load_normalizes_and_validates_resources(self):
        data=json.loads((ROOT/'src/mmswe_service/policy.example.json').read_text())
        data['train_image']=data['generate_image']='repo/image@sha256:'+'a'*64
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'policy.json'
            data['instance_resources']={'cpus':8,'memory_gib':32,'pids':8192}
            path.write_text(json.dumps(data))
            with patch.object(policy,'private_file',side_effect=lambda p:Path(p)),patch.object(policy,'load_contract'),patch.object(policy,'load_baseline'):
                self.assertEqual(policy.load(path)['instance_resources'],data['instance_resources'])
                data['instance_resources']['pids']=-1;path.write_text(json.dumps(data))
                with self.assertRaisesRegex(policy.Rejected,'pids'):policy.load(path)

    def test_native_resource_evidence_does_not_change_test_exit_marker(self):
        script=grade.native.build_chroot_script({'Env':[]})
        checked=subprocess.run(['/bin/bash','-n'],input=script,text=True,capture_output=True)
        self.assertEqual(checked.returncode,0,checked.stderr)
        self.assertIn('pids.events',script)
        self.assertIn('mmptb_resource_diag before >> /tmp/preflight.log',script)
        self.assertIn('mmptb_resource_diag after >> /tmp/preflight.log',script)
        self.assertIn('/bin/bash /eval.sh >> /tmp/test_output.txt 2>&1\n  echo "EVAL_RC=$?"',script)


if __name__=='__main__':unittest.main()
