"""Exercise preflight stop/cleanup behavior without claiming real acceptance."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('preflight', ROOT/'scripts/mmswe_runtime_preflight.py')
preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def exercise(self, inspect_response=None, cleanup_ok=True):
        with tempfile.TemporaryDirectory() as td:
            args=types.SimpleNamespace(out=str(Path(td)/'out'), docker='/fake/docker',
                image='sha256:'+'a'*64,profile='platform-managed-v1',kind='instance',
                probes='identity',test_user='nobody',chrome=None,firefox=None,timeout=10)
            calls=[]
            def invoke(cmd,**kw):
                calls.append(cmd)
                self.assertNotIn('DOCKER_HOST',kw['env'])
                op=cmd[1]
                if op=='create':
                    return subprocess.CompletedProcess(cmd,0 if inspect_response else 1,'c'*64,'denied' if not inspect_response else '')
                if op=='inspect':return subprocess.CompletedProcess(cmd,0,json.dumps([inspect_response]),'')
                if op=='rm':return subprocess.CompletedProcess(cmd,0 if cleanup_ok else 1,'','' if cleanup_ok else 'denied')
                raise AssertionError('probe must not start after configuration failure')
            with patch.object(preflight.subprocess,'run',side_effect=invoke),contextlib.redirect_stdout(io.StringIO()):
                code=preflight.run(args)
            return code,json.loads((Path(args.out)/'result.json').read_text()),calls

    def test_create_denial_is_not_retried_with_another_profile(self):
        code,result,calls=self.exercise()
        self.assertEqual(code,2);self.assertEqual(result['status'],'create_rejected_or_failed')
        self.assertEqual([c[1] for c in calls],['create','rm'])
        self.assertTrue(result['cleanup_ok']);self.assertFalse(result['official_acceptance'])

    def test_missing_policy_stops_before_start_and_cleanup_failure_is_not_hidden(self):
        info={'Config':{'User':'0:0'},'HostConfig':{'Privileged':False}}
        code,result,calls=self.exercise(info)
        self.assertEqual(code,2);self.assertEqual(result['status'],'configuration_rejected_before_start')
        self.assertNotIn('start',[c[1] for c in calls])
        code,result,_=self.exercise(info,cleanup_ok=False)
        self.assertEqual(code,4);self.assertEqual(result['status'],'cleanup_failed')

    def test_shell_probes_parse_and_never_disable_browser_sandbox(self):
        for kind,probes,user in [('instance','identity','nobody'),('workload','identity','nobody'),
                                 ('instance','browsers','chromeuser')]:
            script=preflight.probe_script(kind,probes,user)
            check=subprocess.run(['/bin/sh','-n'],input=script,text=True,capture_output=True)
            self.assertEqual(check.returncode,0,check.stderr)
            self.assertNotIn('--no-sandbox',script)
            self.assertNotIn('--disable-setuid-sandbox',script)


if __name__=='__main__':unittest.main()
