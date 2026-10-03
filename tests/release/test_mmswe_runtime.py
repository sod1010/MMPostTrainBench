"""CPU regressions for score completion and frozen grading assets."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / 'src/dlc/score_job.py'
spec = importlib.util.spec_from_file_location('grade_asset_test', ROOT / 'src/eval/tasks/mmswe/dlc_native_grade.py')
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'out'
        self.out.mkdir()
        self.adapter = self.root / 'adapter.py'
        self.adapter.write_text('''import os,sys,json,time,subprocess
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--json-output-file')+1])
mode=os.environ.get('CPU_SCORE_CASE','ok')
assert os.environ['MMPTB_ROLE']=='agent' and os.environ['EVAL_SPLIT']=='val'
if mode=='hang':
    subprocess.Popen([sys.executable,'-c',"import time;from pathlib import Path;time.sleep(1);Path("+repr(str(out.parent/'orphan'))+").touch()"])
    time.sleep(30)
if mode=='fail':raise SystemExit(8)
if mode!='missing':out.write_text(json.dumps({'accuracy':0 if mode!='nan' else float('nan'),'n':1,'correct':0,'task':'swe_bench_multimodal@val'}))
''')

    def execute(self, mode):
        env = dict(os.environ, CPU_SCORE_CASE=mode, EVAL_SPLIT='eval', MMPTB_ROLE='verifier')
        return subprocess.run([sys.executable, str(JOB), 'run', '--out', str(self.out),
                               '--model', 'fixture', '--adapter', str(self.adapter),
                               '--timeout', '.2' if mode == 'hang' else '5'],
                              env=env, capture_output=True, text=True, timeout=10)

    def waiter(self, *extra):
        return subprocess.run([sys.executable, str(JOB), 'wait', '--out', str(self.out),
                               '--timeout', '.15', '--poll', '.01', *extra],
                              capture_output=True, text=True, timeout=5)

    def test_success_zero_and_failures_publish_terminal_status(self):
        for mode, expected in [('ok', 0), ('fail', 8), ('missing', 70), ('nan', 70)]:
            with self.subTest(mode=mode):
                (self.out/'reward.txt').write_text('0.99')
                result = self.execute(mode)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertTrue((self.out/'DONE').exists())
                self.assertEqual(json.loads((self.out/'status.json').read_text())['returncode'], expected)
                self.assertEqual(self.waiter().returncode, expected)
                self.assertEqual((self.out/'reward.txt').exists(), expected == 0)
                self.assertEqual((self.out/'metrics.json').exists(), expected == 0)

    def test_timeout_cleans_descendants_and_old_score(self):
        self.assertEqual(self.execute('hang').returncode, 124)
        time.sleep(1.1)
        self.assertFalse((self.out/'orphan').exists())
        self.assertFalse((self.out/'metrics.json').exists())
        self.assertEqual(self.waiter().returncode, 124)

    def test_wait_has_deadline_and_detects_worker_exit_and_loss(self):
        (self.out/'DONE').write_text('DONE')  # old marker cannot certify success
        self.assertEqual(self.waiter().returncode, 124)
        queue = self.root/'queue'
        (queue/'tasks').mkdir(parents=True)
        args = ['--queue', str(queue), '--job-id', 'qworker-cmd.1.2']
        self.assertEqual(self.waiter(*args).returncode, 69)
        (queue/'.heartbeat').touch()
        (queue/'tasks/cmd.1.2.rc').write_text('0')
        self.assertEqual(self.waiter(*args).returncode, 70)


class AssetTests(unittest.TestCase):
    def test_frozen_screenshots_reject_unknown_and_corrupt_inputs(self):
        spec = importlib.util.spec_from_file_location('frozen_image_test', ROOT/'eval_omni/runners/run_mmswe_official.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = b'image fixture'
            digest = hashlib.sha256(data).hexdigest()
            (root/'blobs').mkdir()
            blob = root/'blobs'/(digest+'.png')
            blob.write_bytes(data)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'version':1,'images':{
                'image':{'status':'available','sha256':digest,'size':len(data)},
                'known_missing':{'status':'unavailable'}}}))
            with patch.dict(os.environ, {'MMSWE_IMAGE_MANIFEST':str(manifest),
                                        'MMSWE_IMAGE_MANIFEST_SHA256':hashlib.sha256(manifest.read_bytes()).hexdigest()}, clear=True):
                self.assertEqual(runner._fetch_images(['image','known_missing'],4),[str(blob.resolve())])
                with self.assertRaises(KeyError):runner._fetch_images(['unknown'],4)
                blob.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):runner._fetch_images(['image'],4)
            with patch.dict(os.environ, {'MMSWE_REQUIRE_FROZEN_IMAGES':'1'},clear=True):
                with self.assertRaises(ValueError):runner._fetch_images(['image'],4)

    def test_frozen_manifest_never_uses_network_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = b'fixture binary asset'
            digest = hashlib.sha256(data).hexdigest()
            (root/'blobs').mkdir()
            (root/'blobs'/digest).write_bytes(data)
            url = 'https://raw.githubusercontent.com/a/b/commit/asset.png'
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'version':1,'assets':{url:{'sha256':digest,'size':len(data)}}}))
            pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with patch.dict(os.environ, {'MMSWE_ASSET_MANIFEST':str(manifest), 'MMSWE_ASSET_MANIFEST_SHA256':pin}), \
                 patch.object(grade.urllib.request, 'urlopen', side_effect=AssertionError('network must not run')):
                self.assertEqual(grade.load_asset_bytes(url), data)
                spec = types.SimpleNamespace(image_assets={'test_patch':[{'path':'test/a.png','url':url}]})
                commands, failed = grade._stage_assets_into_rootfs(spec, root, io.StringIO())
                self.assertEqual(failed, [])
                self.assertEqual(len(commands), 1)
                self.assertIn('exit 87', commands[0])
                (root/'blobs'/digest).write_bytes(b'corrupt')
                with self.assertRaises(ValueError):grade.load_asset_bytes(url)
                _, failed = grade._stage_assets_into_rootfs(spec, root, io.StringIO())
                self.assertEqual(failed, ['test/a.png'])
                (root/'blobs'/digest).write_bytes(data)
                manifest.write_text(manifest.read_text()+' ')
                with self.assertRaises(ValueError):grade.load_asset_bytes(url)

    def test_cache_hit_is_hash_verified_before_unpacking_without_online_repair(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);body=b'cached layer';digest='sha256:'+hashlib.sha256(body).hexdigest()
            file=root/digest.replace(':','_');file.write_bytes(body)
            with patch.object(grade,'BLOB_CACHE',root),patch.object(grade,'fetch_blob_verified') as fetch:
                self.assertEqual(grade.ensure_blob('repo',digest,root,io.StringIO()),(file,False))
                file.write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError,'cached blob digest mismatch'):
                    grade.ensure_blob('repo',digest,root,io.StringIO())
                fetch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
