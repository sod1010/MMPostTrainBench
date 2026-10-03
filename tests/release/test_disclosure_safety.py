"""Credential-handling regressions using synthetic values and no network/GPU."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('disclosure_judge', ROOT / 'src/eval/judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


class JudgeDisclosureTests(unittest.TestCase):
    def test_sensitive_files_symlinks_and_special_files_are_not_packed(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root = Path(td)
            ws = root / 'workspace'
            ws.mkdir()
            for rel in ('.codex/auth.json', '.claude/config.json', '.gemini/settings.json',
                        '.aws/config.txt', '.docker/config.json', '.config/tool/settings.json',
                        'nested/credentials.json', '.env.local.json'):
                p = ws / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('private-file-sentinel')
            custom = ws / 'operator-key.txt'
            custom.write_text('private-file-sentinel')
            os.environ['JUDGE_KEYFILE'] = str(custom)
            outside = root / 'outside.txt'
            outside.write_text('private-file-sentinel')
            (ws / 'linked.txt').symlink_to(outside)
            (ws / 'linked-dir').symlink_to(root, target_is_directory=True)
            os.mkfifo(ws / 'pipe.txt')
            (ws / 'train.py').write_text('train(dataset="public/train.jsonl")\n')
            packed = judge.pack_workspace(str(ws), 20000)
            self.assertNotIn('private-file-sentinel', packed)
            self.assertIn('train(dataset="public/train.jsonl")', packed)

    def test_redaction_preserves_evidence_and_handles_truncation(self):
        known = 'sk-' + 'syntheticvalue' * 4
        file_key = 'fixture-file-secret-' + 'a' * 50
        with tempfile.TemporaryDirectory() as td, patch.dict(
                os.environ, {'OPENAI_API_KEY': known}, clear=True):
            p = Path(td) / 'train.py'
            p.write_text(f'api_key = "{file_key}"\nclient.post("https://example.invalid/answer")\n'
                         f'log = "Bearer {known}"\npassword: synthetic-password-value\n')
            packed = judge.pack_workspace(td, 20000, (file_key,))
            self.assertNotIn(known, packed)
            self.assertNotIn(file_key, packed)
            self.assertNotIn('synthetic-password-value', packed)
            self.assertIn('https://example.invalid/answer', packed)
            self.assertIn('api_key', packed)
            self.assertIn('[REDACTED]', packed)
            p.write_text(file_key)
            self.assertNotIn(file_key[:12], judge.pack_workspace(td, 12, (file_key,)))

    def run_judge(self, ws, env, response=None, error=None):
        out = io.StringIO()
        with (patch.dict(os.environ, env, clear=True), patch.object(
                sys, 'argv', ['judge.py', '--workspace', str(ws)]), patch.object(
                judge, 'call_judge', return_value=response, side_effect=error) as call,
                contextlib.redirect_stdout(out)):
            result = judge.main()
        self.assertEqual(result, 0)
        return json.loads(out.getvalue()), call

    def test_missing_and_empty_key_skip_without_a_request(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'key'
            for exists in (False, True):
                if exists:
                    p.write_text('')
                verdict, call = self.run_judge(td, {'JUDGE_KEY_FILE': str(p)})
                self.assertEqual(verdict['verdict'], 'skipped')
                self.assertIsNone(verdict['cheated'])
                call.assert_not_called()

    def test_errors_and_returned_verdicts_do_not_echo_key(self):
        key = 'fixture-key-' + 's' * 32
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'judge-key'
            p.write_text(key)
            env = {'JUDGE_KEYFILE': str(p)}
            verdict, _ = self.run_judge(td, env, error=RuntimeError(key))
            self.assertEqual(verdict['verdict'], 'skipped')
            self.assertNotIn(key, json.dumps(verdict))
            verdict, _ = self.run_judge(td, env, response=json.dumps(
                {'cheated': True, 'category': 'disallowed_method', 'reasons': key}))
            self.assertEqual(verdict['verdict'], 'flagged')
            self.assertNotIn(key, json.dumps(verdict))
            verdict, _ = self.run_judge(td, env, response='{"reasons":"missing verdict"}')
            self.assertEqual(verdict['verdict'], 'unparseable')
            self.assertIsNone(verdict['cheated'])


class LauncherDisclosureTests(unittest.TestCase):
    def make_capture(self, root, binary, keys, exit_code=0):
        bindir = root / 'bin'
        bindir.mkdir(exist_ok=True)
        script = root / 'capture.py'
        script.write_text('import os, sys, json, pathlib\n'
                          'keys=' + repr(keys) + '\n'
                          "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({"
                          "'argv': sys.argv[1:], 'matched': {k: os.environ.get(k) == "
                          "os.environ.get('EXPECTED_'+k) for k in keys}}))\n"
                          f'sys.exit({exit_code})\n')
        wrapper = bindir / binary
        wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' +
                           shlex.quote(str(script)) + ' "$@"\n')
        wrapper.chmod(0o755)
        return {'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
                'HOME': str(root), 'CAPTURE': str(root / 'capture.json')}

    def test_docker_credentials_reach_environment_but_not_argv(self):
        cases = [('run_agent.sh', ['ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN',
                                  'OPENAI_API_KEY', 'GEMINI_API_KEY']),
                 ('run_verifier.sh', ['CODEX_API_KEY', 'OPENAI_API_KEY'])]
        for launcher, keys in cases:
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                shutil.copyfile(ROOT / 'src/docker' / launcher, root / launcher)
                env = self.make_capture(root, 'docker', keys)
                workspace = root / 'workspace'
                (workspace / 'final_model').mkdir(parents=True)
                (workspace / 'final_model/config.json').write_text('{}')
                bundle = root / 'bundle'
                (bundle / 'environment').mkdir(parents=True)
                (bundle / 'instruction.md').write_text('Synthetic task')
                (bundle / 'task.toml').write_text('[agent]\ntimeout_sec=1\n')
                config = dict(WORKSPACE_HOST=str(workspace), LOGS_HOST=str(root / 'logs'),
                              MODEL_DIR=str(workspace / 'final_model'), TASK_DIR=str(bundle),
                              MMPTB_ROOT=str(root), HF_CACHE_DIR=str(root / 'cache'),
                              DATA_DIR=str(root / 'data'), GPUS='all',
                              AGENT_ENGINE='codex', AGENT_IMAGE='fixture', VERIFIER_IMAGE='fixture')
                for key in keys:
                    config[key] = 'fixture-' + key.lower() + '-value'
                    env['EXPECTED_' + key] = config[key]
                # Deliberately not exported: the launcher must export sourced values.
                (root / 'config.env').write_text('\n'.join(
                    k + '=' + shlex.quote(v) for k, v in config.items()) + '\n')
                run = subprocess.run(['bash', str(root / launcher)], env=env,
                                     capture_output=True, text=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                capture = json.loads((root / 'capture.json').read_text())
                for key in keys:
                    self.assertTrue(capture['matched'][key], key)
                    self.assertIn(key, capture['argv'])
                    self.assertNotIn(config[key], json.dumps(capture['argv']) + run.stdout + run.stderr)

    def test_hf_download_uses_env_token(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copyfile(ROOT / 'src/docker/prepare_data.sh', root / 'prepare_data.sh')
            # Stop after capturing the first download, before any dataset work.
            env = self.make_capture(root, 'hf', ['HF_TOKEN'], exit_code=23)
            env['EXPECTED_HF_TOKEN'] = 'fixture-hf-token-value'
            token = root / 'private-token'
            token.write_text(env['EXPECTED_HF_TOKEN'])
            (root / 'resources.json').write_text('{}')
            config = dict(HF_TOKEN_FILE=str(token), HF_ENDPOINT='https://example.invalid',
                          HF_CACHE_DIR=str(root / 'cache'), DATA_DIR=str(root / 'data'),
                          MODEL_REPO='fixture/model', MODEL_DIR=str(root / 'model'))
            (root / 'config.env').write_text('\n'.join(
                k + '=' + shlex.quote(v) for k, v in config.items()) + '\n')
            run = subprocess.run(['bash', str(root / 'prepare_data.sh')], env=env,
                                 capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 23, run.stdout + run.stderr)
            capture = json.loads((root / 'capture.json').read_text())
            self.assertTrue(capture['matched']['HF_TOKEN'])
            self.assertNotIn('--token', capture['argv'])
            self.assertNotIn(env['EXPECTED_HF_TOKEN'], json.dumps(capture['argv']) + run.stdout + run.stderr)

    def test_warmup_ca_is_optional_and_explicit_missing_ca_fails(self):
        for mode in ('default', 'configured', 'missing'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                shutil.copyfile(ROOT / 'src/docker/warmup_dataset.sh', root / 'warmup_dataset.sh')
                env = self.make_capture(root, 'docker', [])
                config = dict(MMPTB_ROOT=str(root), HF_CACHE_DIR=str(root / 'cache'),
                              OMNI_EVAL_IMAGE='fixture')
                ca = root / 'operator-ca.crt'
                if mode != 'default':
                    config['CABUNDLE'] = str(ca)
                if mode == 'configured':
                    ca.write_text('synthetic certificate fixture; not used for TLS')
                (root / 'config.env').write_text('\n'.join(
                    k + '=' + shlex.quote(v) for k, v in config.items()) + '\n')
                run = subprocess.run(['bash', str(root / 'warmup_dataset.sh')], env=env,
                                     capture_output=True, text=True, timeout=20)
                if mode == 'missing':
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse((root / 'capture.json').exists())
                    continue
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                argv = json.loads((root / 'capture.json').read_text())['argv']
                self.assertEqual(str(ca) + ':/ca.crt:ro' in argv, mode == 'configured')
                body = (root / '_warmup_body.sh').read_text()
                subprocess.run(['bash', '-n', str(root / '_warmup_body.sh')], check=True)
                self.assertIn('if [ -f /ca.crt ]; then', body)
                self.assertNotIn('verify=False', body)
                self.assertNotIn('SSL_NO_VERIFY', body)


if __name__ == '__main__':
    unittest.main()
