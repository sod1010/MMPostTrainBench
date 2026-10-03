"""Publication regressions. All fixtures run on CPU without models or datasets."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from docker_env_fixture import image_policy_env

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "src/eval"
FIX = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(EVAL))
from split_util import keep, resolve_split, mmswe_config
from test_val_isolation import inspect_output

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

grader = load("native_grader", EVAL / "tasks/mmswe/dlc_native_grade.py")
lmms = load("lmms_adapter", EVAL / "lmms_common/evaluate.py")

class SplitTests(unittest.TestCase):
    def test_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            for request in (None, "", "all", "full", "none", "val", "eval", " eval "):
                self.assertEqual(resolve_split(request), "val")
            with self.assertRaises(ValueError):
                resolve_split("typo")
            os.environ["MMPTB_ROLE"] = "verifier"
            self.assertEqual(resolve_split("eval"), "eval")
            self.assertEqual(resolve_split("all"), "")
            self.assertEqual(mmswe_config("eval")[1], "test")
            os.environ["SWE_SPLIT"] = "dev"
            with self.assertRaises(ValueError):
                mmswe_config("eval")
            os.environ["MMSWE_DATASET"] = "wrong"
            os.environ["SWE_DATASET"] = "other"
            with self.assertRaises(ValueError):
                mmswe_config("val")

    def test_membership_unchanged(self):
        import hashlib
        for i in range(1000):
            expected = int(hashlib.md5(f"1234:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < .3
            self.assertEqual(keep(i, "val"), expected)
            self.assertEqual(keep(i, "eval"), not expected)

class AdapterTests(unittest.TestCase):
    def run_adapter(self, bench, mode="ok", split="val", role="agent", target=None, extra_env=None):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        out = Path(td.name)
        (out / "metrics.json").write_text('{"accuracy": 0.99}')
        (out / "diag.jsonl").write_text("stale")
        data = out / "data.json"
        data.write_text("[]")
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(("LMMS_", "SWE_", "MMSWE_", "MMPTB_")):
                del env[key]
        env.update(PYTHONPATH=os.pathsep.join((str(FIX), str(EVAL))),
                   EVAL_SPLIT=split, MMPTB_ROLE=role, FIXTURE_MODE=mode,
                   FIXTURE_KIND=bench, OVB_DATA=str(data), JOINTAV_DATA=str(data),
                   SWE_VENV_PY=sys.executable, MMSWE_GRADER=str(FIX / "runner.py"))
        for key in ("OVB_RUNNER", "MMAU_RUNNER", "MMAR_RUNNER", "JOINTAV_RUNNER", "MMSWE_RUNNER"):
            env[key] = str(FIX / "runner.py")
        env.update(extra_env or {})
        cmd = [sys.executable, str(target or EVAL / f"tasks/{bench}/evaluate.py"),
               "--model-path", "fixture-model", "--json-output-file", str(out / "metrics.json")]
        result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / "run.log").write_text(result.stdout)
        return result, out

    def test_mmswe_image_defaults_follow_effective_split(self):
        for dockerfile in (ROOT / "src/harbor_adapter/template/tests/Dockerfile.mmswe",
                           ROOT / "harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/tests/Dockerfile"):
            baked = image_policy_env(ROOT / "eval_omni/Dockerfile", dockerfile)
            for requested, role, effective, dataset_split in (
                ("eval", "verifier", "eval", "test"),
                ("val", "verifier", "val", "dev"),
                ("eval", "agent", "val", "dev"),
            ):
                with self.subTest(dockerfile=dockerfile, requested=requested, role=role):
                    # Model Docker ENV + explicit -e precedence, without
                    # importing host split defaults into the container.
                    env = dict(baked, EVAL_SPLIT=requested, MMPTB_ROLE=role)
                    result, out = self.run_adapter("mmswe", extra_env=env)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn("generation split=" + dataset_split, result.stdout)
                    self.assertIn("grading split=" + dataset_split, result.stdout)
                    metrics = json.loads((out / "metrics.json").read_text())
                    self.assertEqual(metrics["task"], "swe_bench_multimodal@" + effective)
                    self.assertEqual((out / "diag.jsonl").exists(), effective == "val")

    def test_mmswe_explicit_conflict_still_fails_before_generation(self):
        baked = image_policy_env(ROOT / "eval_omni/Dockerfile",
                                 ROOT / "harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/tests/Dockerfile")
        for requested, declaration in (
            ("eval", {"SWE_SPLIT": "dev"}),
            ("eval", {"MMSWE_SPLIT": "dev"}),
            ("eval", {"SWE_SPLIT": "dev", "MMSWE_SPLIT": "dev"}),
            ("val", {"SWE_SPLIT": "test", "MMSWE_SPLIT": "test"}),
        ):
            with self.subTest(requested=requested, declaration=declaration):
                env = dict(baked, EVAL_SPLIT=requested, MMPTB_ROLE="verifier", **declaration)
                result, out = self.run_adapter("mmswe", extra_env=env)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("generation split=", result.stdout)
                self.assertFalse((out / "metrics.json").exists())
                self.assertFalse((out / "diag.jsonl").exists())

    def test_all_benches_default_downgrade_zero_and_sealed(self):
        for bench in ("mmmu_pro", "video_mmmu", "videomme_v2", "omnivideobench", "mmau", "mmar", "jointavbench", "mmswe"):
            for split, role in (("val", "agent"), ("", "agent"), ("eval", "agent"), ("eval", "verifier")):
                with self.subTest(bench=bench, split=split, role=role):
                    result, out = self.run_adapter(bench, split=split, role=role)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    metrics = json.loads((out / "metrics.json").read_text())
                    self.assertEqual(metrics["accuracy"], 0)
                    self.assertGreater(metrics["n"], 0)
                    self.assertNotIn("SEALED_SENTINEL", result.stdout)
                    self.assertNotIn("Correct predictions:", result.stdout)
                    if role == "agent":
                        self.assertEqual(inspect_output(out), [])
                        diag = (out / "diag.jsonl").read_text()
                        self.assertNotIn("SEALED_SENTINEL", diag)
                        self.assertNotIn("stale", diag)
                    else:
                        self.assertFalse((out / "diag.jsonl").exists())

    def test_failures_never_reuse_old_metrics(self):
        matrix = {
            "mmmu_pro": ("runner_fail", "no_output", "no_samples", "wrong_metric", "wrong_task", "missing_metric", "duplicate", "nan", "bad_json"),
            "omnivideobench": ("runner_fail", "no_output", "bad_json", "empty", "malformed"),
            "mmau": ("runner_fail", "no_output", "empty"),
            "mmswe": ("runner_fail", "grader_fail", "partial", "wrong_split", "infra", "gen_failed"),
        }
        for bench, modes in matrix.items():
            for mode in modes:
                with self.subTest(bench=bench, mode=mode):
                    result, out = self.run_adapter(bench, mode)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((out / "metrics.json").exists(), result.stdout)
                    self.assertFalse((out / "diag.jsonl").exists())
                    self.assertNotIn("SEALED_SENTINEL", result.stdout)

    def test_invalid_split_fails_before_launch(self):
        for bench in ("mmmu_pro", "omnivideobench", "mmau", "mmar", "jointavbench", "mmswe"):
            result, out = self.run_adapter(bench, split="typo")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((out / "metrics.json").exists())

    def test_missing_split_helper(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "metrics.json"
            output.write_text('{"accuracy":1}')
            args = type("Args", (), {"json_output_file": str(output)})()
            with patch.object(lmms, "parse_args", return_value=args), patch.object(lmms, "_SPLIT_UTIL_OK", False), patch.object(lmms.subprocess, "run") as run:
                self.assertNotEqual(lmms.main(), 0)
                run.assert_not_called()
                self.assertFalse(output.exists())

    def test_flat_bundles_execute(self):
        for bench in ("mmmu_pro", "video_mmmu", "videomme_v2", "omnivideobench", "mmau", "mmar", "jointavbench", "mmswe"):
            result, _ = self.run_adapter(bench, target=ROOT / f"harbor_tasks/mmposttrainbench-{bench}-qwen3-omni-30b/tests/evaluate.py")
            self.assertEqual(result.returncode, 0, result.stdout)

class GraderTests(unittest.TestCase):
    def test_non_utf8_execution_log_reaches_strict_parser_and_retains_raw_bytes(self):
        from types import SimpleNamespace
        import hashlib
        for outcome, permission_error in ((True, False), (False, False), (True, True)):
            with self.subTest(outcome=outcome, permission_error=permission_error), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                out, scratch = root / "out", root / "scratch"
                out.mkdir()
                raw = (grader.APPLY_PATCH_PASS + "\n" + grader.START_TEST_OUTPUT + "\n").encode()
                raw += b"progress: \xe2\n"
                raw += b"VALID_PASS\n" if outcome else b"VALID_FAIL\n"
                if permission_error:
                    raw += b"su: cannot set groups: Operation not permitted\n"
                raw += b">>>>> End Test Output\n"

                def execute(rootfs, logf, timeout):
                    (rootfs / "tmp/test_output.txt").write_bytes(raw)
                    (rootfs / "tmp/apply.log").write_text("APPLIED=1\nEVAL_RC=0\n")
                    return 0

                parser_paths = []
                def strict_parser(test_spec, prediction, test_log_path, include_tests_status):
                    parser_paths.append(Path(test_log_path))
                    # Reproduce upstream get_logs_eval's strict UTF-8 reader.
                    text = Path(test_log_path).read_text(encoding="utf-8")
                    return {"case": {"resolved": "VALID_PASS\n" in text,
                                     "patch_successfully_applied": True}}

                args = SimpleNamespace(pull_budget=1, timeout=1, keep_rootfs=True)
                spec = SimpleNamespace(image="fixture", eval_script="fixture", image_assets={})
                with patch.object(grader, "pull_and_unpack", return_value={}), \
                     patch.object(grader, "_stage_assets_into_rootfs", return_value=([], [])), \
                     patch.object(grader, "run_in_namespace", side_effect=execute):
                    _, report, resolved = grader.grade_one(
                        "case", {"case": {"model_patch": "fixture"}}, {"case": {}},
                        out, scratch, args, lambda _: spec, strict_parser)
                row = report["case"]
                self.assertEqual(row["infra_failure"], permission_error)
                self.assertEqual(resolved, outcome and not permission_error)
                original = out / "case/test_output.txt"
                self.assertEqual(original.read_bytes(), raw)
                self.assertEqual(parser_paths, [out / "case/test_output.utf8.txt"])
                decoding = json.loads((out / "case/test_output.decoding.json").read_text())
                self.assertEqual(row["log_decoding"], decoding)
                self.assertEqual(decoding["raw_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(decoding["parser_sha256"], hashlib.sha256(parser_paths[0].read_bytes()).hexdigest())

    def test_valid_utf8_log_keeps_original_parser_input(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_output.txt"
            raw = "测试名称 — café\n".encode("utf-8")
            path.write_bytes(raw)
            parser_path, text, decoding = grader.prepare_grading_log(path)
            self.assertEqual(parser_path, path)
            self.assertEqual(text.encode("utf-8"), raw)
            self.assertIsNone(decoding)
            self.assertEqual(list(Path(td).iterdir()), [path])

    def test_browser_sandbox_fatal_is_infra_even_with_complete_markers(self):
        # Synthetic startup failures: no browser or container is launched.
        fatal_lines = (
            '[38:38:0913/010000.000000:FATAL:zygote_host_impl_linux.cc(127)] No usable sandbox!',
            '[38:38:0913/010000.000000:FATAL:content/browser/zygote_host/zygote_host_impl_linux.cc:128] No usable sandbox!',
        )
        for source in (EVAL / 'tasks/mmswe/dlc_native_grade.py',
                       ROOT / 'harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/tests/dlc_native_grade.py',
                       ROOT / 'harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/environment/dlc_native_grade.py'):
            module = load('browser_failure_fixture', source)
            for fatal in fatal_lines:
                for resolved in (False, True):
                    with self.subTest(source=source, fatal=fatal, resolved=resolved):
                        report = {'a': {'resolved': resolved, 'patch_successfully_applied': True}}
                        log = '\n'.join((module.APPLY_PATCH_PASS, module.START_TEST_OUTPUT,
                                         fatal, '>>>>> End Test Output'))
                        r = module.validate_grade_report('a', report, 'APPLIED=1\n', log, 0, [])['a']
                        self.assertTrue(r['infra_failure'])
                        self.assertFalse(r['resolved'])
                        self.assertIn('browser sandbox startup failure', r['infra_reasons'])

    def test_sandbox_mention_and_ordinary_test_failure_remain_valid(self):
        report = {'a': {'resolved': False, 'patch_successfully_applied': True}}
        for text in ('Expected error string: No usable sandbox!', 'FAIL ordinary application assertion'):
            log = '\n'.join((grader.APPLY_PATCH_PASS, grader.START_TEST_OUTPUT,
                             text, '>>>>> End Test Output'))
            r = grader.validate_grade_report('a', report, 'APPLIED=1\n', log, 0, [])['a']
            self.assertFalse(r['infra_failure'])
            self.assertFalse(r['resolved'])

    def test_browser_launcher_exhaustion_is_infra_but_transient_failure_is_not(self):
        for source in (EVAL / 'tasks/mmswe/dlc_native_grade.py',
                       ROOT / 'harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/tests/dlc_native_grade.py',
                       ROOT / 'harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/environment/dlc_native_grade.py'):
            module = load('launcher_failure_fixture', source)
            terminal = '11 09 2026 10:57:31.569:ERROR [launcher]: Firefox failed 2 times (cannot start). Giving up.'
            cases = [(terminal, True), ('\x1b[91m'+terminal+'\x1b[39m', True),
                     (terminal.replace('Firefox', 'Chrome'), True), ('+'+terminal, False),
                     ('Expected log: '+terminal, False),
                     ('11 09 2026 10:57:31.569:ERROR [launcher]: Cannot start Firefox\n'
                      '11 09 2026 10:57:32.000:INFO [Firefox]: Connected', False)]
            for text, failed in cases:
                for resolved in (True, False):
                    with self.subTest(source=source, text=text, resolved=resolved):
                        report = {'a': {'resolved': resolved, 'patch_successfully_applied': True}}
                        log = '\n'.join((module.APPLY_PATCH_PASS, module.START_TEST_OUTPUT,
                                         text, '>>>>> End Test Output'))
                        result = module.validate_grade_report('a', report, 'APPLIED=1\n', log, 0, [])['a']
                        self.assertEqual(result['infra_failure'], failed)
                        self.assertEqual(result['resolved'], resolved and not failed)

    def test_runtime_evidence(self):
        iid = "a"
        good = {iid: {"resolved": True, "patch_successfully_applied": True}}
        log = "\n".join((grader.APPLY_PATCH_PASS, grader.START_TEST_OUTPUT, ">>>>> End Test Output"))
        r = grader.validate_grade_report(iid, good, "APPLIED=1\n", log, 0, [])[iid]
        self.assertTrue(r["resolved"])
        self.assertFalse(r["infra_failure"])
        for altered, rc, assets in ((log + "\nsu: cannot set groups: Operation not permitted", 0, []), ("", 0, []), (log, 1, []), (log, 0, ["missing.png"])):
            r = grader.validate_grade_report(iid, good, "APPLIED=1\n", altered, rc, assets)[iid]
            self.assertFalse(r["resolved"])
            self.assertTrue(r["infra_failure"])
        r = grader.validate_grade_report(iid, good, "APPLIED=0\n", grader.APPLY_PATCH_FAIL, 0, [])[iid]
        self.assertFalse(r["resolved"])
        self.assertFalse(r["infra_failure"])

    def test_real_patch_apply_gate_without_container(self):
        for valid in (False, True):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                repo = root / "testbed"
                repo.mkdir()
                (root / "tmp").mkdir()
                env = dict(os.environ, GIT_CONFIG_GLOBAL=str(root / "gitconfig"), GIT_CONFIG_NOSYSTEM="1")
                def git(*args):
                    return subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
                git("init")
                (repo / "a.txt").write_text("before\n")
                git("add", "a.txt")
                git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "base")
                (repo / "a.txt").write_text("after\n")
                diff = git("diff").stdout.decode()
                git("checkout", "--", "a.txt")
                (root / "tmp/patch.diff").write_text(diff if valid else "not a patch\n")
                marker = root / "test-ran"
                (root / "eval.sh").write_text(f"touch {shlex.quote(str(marker))}\n")
                # Replace original /tmp references first: on Linux, the fixture
                # repo itself lives under /tmp and must not be rewritten twice.
                script = grader.build_chroot_script({}).replace("/tmp/", str(root / "tmp") + "/").replace("/testbed", str(repo)).replace("/eval.sh", str(root / "eval.sh"))
                result = subprocess.run(["bash"], input=script, text=True, env=env, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(marker.exists(), valid)
                self.assertIn("APPLIED=" + str(int(valid)), (root / "tmp/apply.log").read_text())

    def test_asset_restore_order_and_paths(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            spec = SimpleNamespace(image_assets={"patch": [{"path": "test/a b.png", "url": "https://fixture.invalid/image"}]})
            with patch.object(grader.urllib.request, "urlopen", return_value=contextlib.closing(io.BytesIO(b"png"))):
                commands, missing = grader._stage_assets_into_rootfs(spec, Path(td), io.StringIO())
            self.assertEqual(missing, [])
            script = grader._inject_asset_restore("apply tests\necho '>>>>> Start Test Output'\nrun tests", commands)
            self.assertLess(script.index(commands[0]), script.index(">>>>> Start"))
            self.assertIn("'test/a b.png'", commands[0])
            with self.assertRaises(ValueError):
                grader._inject_asset_restore("run tests", commands)

class AcceptanceTests(unittest.TestCase):
    def test_empty_bad_and_known_leaks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertTrue(inspect_output(root))
            (root / "run.log").write_text("completed val\n")
            for text in ("{}", "[]", "broken", '{"accuracy":NaN,"n":1}', '{"accuracy":0,"n":0}'):
                (root / "metrics.json").write_text(text)
                self.assertTrue(inspect_output(root))
            (root / "metrics.json").write_text('{"accuracy":0,"correct":0,"n":1}')
            self.assertEqual(inspect_output(root), [])
            self.assertTrue(inspect_output(root, require_manifest=True))
            for leak in ("Correct predictions: 3", "Accuracy: 30%", "SELF-CHECK recompute-over-all=0.3", "n_total: 20"):
                (root / "run.log").write_text(leak)
                self.assertTrue(inspect_output(root))

class VerifierShellTests(unittest.TestCase):
    def test_retry_checks_python_exit_not_tee_exit(self):
        source = (ROOT / "src/harbor_adapter/template/tests/test.sh").read_text()
        functions = source[source.index("run_evaluation() {"):source.index("# Determine token limit")]
        for succeed in (False, True):
            with self.subTest(succeed=succeed), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "evaluate.py").write_text('''import json, pathlib, sys
p = pathlib.Path(sys.argv[sys.argv.index("--json-output-file") + 1])
p.write_text(json.dumps({"accuracy": 0, "n": 1}))
raise SystemExit(''' + ("0" if succeed else "7") + ")\n")
                script = f'''set -e
TESTS={shlex.quote(td)}
WORKSPACE={shlex.quote(td)}
LOGS_DIR={shlex.quote(td)}
EVAL_COUNTER=0
sleep() {{ :; }}
kill_gpu_processes() {{ :; }}
''' + functions + '\nif run_evaluation_with_retry 2 ""; then exit 0; else exit 1; fi\n'
                result = subprocess.run(["bash"], input=script, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0 if succeed else 1, result.stdout + result.stderr)
                self.assertEqual((root / "metrics.json").exists(), succeed)

    def test_reward_extraction_rejects_malformed_results(self):
        source = (ROOT / "src/harbor_adapter/template/tests/test.sh").read_text()
        script = source[source.index('if [ -f "$LOGS_DIR/metrics.json" ]; then', source.index("# Extract accuracy")):]
        for content, valid in (({"accuracy": 0}, True), ({"accuracy": float("nan")}, False), ({"unrelated": 1}, False), ({"accuracy": .5, "error": "infra"}, False)):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "metrics.json").write_text(json.dumps(content))
                (root / "reward.txt").write_text("0.99")
                result = subprocess.run(["bash"], input=f'LOGS_DIR={shlex.quote(td)}\n' + script, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0 if valid else 1, result.stdout + result.stderr)
                self.assertEqual((root / "reward.txt").exists(), valid)
                if valid:
                    self.assertEqual((root / "reward.txt").read_text().strip(), "0")

    def test_raw_entrypoint_rejects_val_before_gpu_work(self):
        result = subprocess.run(["bash", str(ROOT / "eval_omni/entrypoint.sh"), "mmmu_pro", "8"],
                                env={"PATH": os.environ["PATH"], "EVAL_SPLIT": "val"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("run_verifier.sh", result.stderr)

if __name__ == "__main__":
    unittest.main()
