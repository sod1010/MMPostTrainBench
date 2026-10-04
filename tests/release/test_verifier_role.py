"""Exercise verifier shell boundaries without Docker, models, or GPU work.

The Docker command is captured, not run. The inner test.sh is executed with
temporary paths and a CPU evaluator that uses the real split policy.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from docker_env_fixture import image_policy_env

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "src/harbor_adapter/template/tests/test.sh"
PROBE = """import json, os, sys
from split_util import keep, resolve_split, mmswe_config
split = resolve_split(os.environ.get('EVAL_SPLIT'))
result = dict(role=os.environ.get('MMPTB_ROLE'), split=split,
              dataset_split=mmswe_config(split)[1],
              indices=[i for i in range(20) if keep(i, split)])
"""


def clean_env(bin_dir):
    # Do not inherit the operator's role, split, credentials, or shell hooks.
    return {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}


def python_shim(path, script):
    path.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " "
                    + shlex.quote(str(script)) + ' "$@"\n')
    path.chmod(0o755)


class VerifierRoleTests(unittest.TestCase):
    def test_docker_launcher_passes_role_and_split_explicitly(self):
        # The last case exercises a mounted pilot test.sh: the outer role must
        # be present even when that override does not set a role itself.
        cases = (({}, "eval", False),
                 ({"MMPTB_ROLE": "agent", "EVAL_SPLIT": "val"}, "eval", False),
                 ({"VERIFIER_SPLIT": "eval"}, "eval", False),
                 ({"VERIFIER_SPLIT": "val"}, "val", False),
                 ({"MMPTB_ROLE": "agent"}, "eval", True))
        for overrides, expected, pilot in cases:
            with self.subTest(overrides=overrides, pilot=pilot), tempfile.TemporaryDirectory() as td:
                root = Path(td).resolve()
                bin_dir = root / "bin"
                bin_dir.mkdir()
                launcher = root / "run_verifier.sh"
                shutil.copyfile(ROOT / "src/docker/run_verifier.sh", launcher)
                shutil.copyfile(ROOT / "src/docker/runtime_paths.py", root / "runtime_paths.py")
                model = root / "workspaces/run/final_model"
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}")
                config = dict(REPO_ROOT=str(ROOT), MODEL_DIR=str(root / "models/base"), MMPTB_ROOT=str(root),
                              WORKSPACE_HOST=str(model.parent), LOGS_HOST=str(root / "logs/run"),
                              HF_CACHE_DIR=str(root / "cache"), DATA_DIR=str(root / "data"),
                              GPUS="all", VERIFIER_IMAGE="fixture-verifier", CODEX_API_KEY="")
                (root / "config.env").write_text("\n".join(
                    "export " + key + "=" + shlex.quote(value) for key, value in config.items()) + "\n")
                capture = root / "capture.py"
                capture.write_text("import json, os, pathlib, sys\n"
                                   "pathlib.Path(os.environ['CAPTURE_PATH']).write_text(json.dumps(sys.argv[1:]))\n")
                python_shim(bin_dir / "docker", capture)
                env = clean_env(bin_dir)
                env.update(overrides, CAPTURE_PATH=str(root / "argv.json"))
                if pilot:
                    override = root / "pilot test.sh"
                    override.write_text("#!/bin/bash\n# No role set here.\n")
                    env.update(BENCH="mmswe", VERIFIER_TEST_SH=str(override))
                run = subprocess.run(["bash", str(launcher)], env=env,
                                     capture_output=True, text=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                argv = json.loads((root / "argv.json").read_text())
                passed_env = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-e"]
                self.assertEqual([v for v in passed_env if v.startswith("MMPTB_ROLE=")],
                                 ["MMPTB_ROLE=verifier"])
                self.assertEqual([v for v in passed_env if v.startswith("EVAL_SPLIT=")],
                                 ["EVAL_SPLIT=" + expected])
                self.assertEqual(argv[-2:], ["fixture-verifier", "/tests/test.sh"])
                self.assertEqual(argv[argv.index("--entrypoint") + 1], "/bin/bash")
                if pilot:
                    self.assertIn(str(override) + ":/tests/test.sh:ro", argv)
                    self.assertIn(str(ROOT / "harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/tests")
                                  + ":/tests:ro", argv)
                # Docker ENV remains unless explicitly overridden by -e.
                # The MMSWE pilot therefore also tests baked dataset defaults.
                dockerfile = (ROOT / "harbor_tasks/mmposttrainbench-mmswe-qwen3-omni-30b/tests/Dockerfile"
                              if pilot else ROOT / "src/harbor_adapter/template/tests/Dockerfile")
                container_env = image_policy_env(ROOT / "eval_omni/Dockerfile", dockerfile)
                container_env["PYTHONPATH"] = str(ROOT / "eval_omni/runners")
                # Policy variables are explicit assignments; name-only credential
                # forwarding is tested separately in test_disclosure_safety.py.
                container_env.update(v.split("=", 1) for v in passed_env if "=" in v)
                probe = subprocess.run([sys.executable, "-c", PROBE + "print(json.dumps(result))"],
                                       env=container_env, capture_output=True, text=True, check=True)
                resolved = json.loads(probe.stdout)
                self.assertEqual(resolved["split"], expected)
                self.assertEqual(resolved["dataset_split"], "test" if expected == "eval" else "dev")
                self.assertNotIn("forcing val", probe.stderr)

    def test_full_verifier_shell_sets_role_for_all_bundles(self):
        bundles = sorted((ROOT / "harbor_tasks").glob("*/tests/test.sh"))
        self.assertEqual(len(bundles), 8)
        for source in [TEMPLATE, *bundles]:
            for overrides, expected in (({}, "eval"),
                                        ({"MMPTB_ROLE": "agent", "EVAL_SPLIT": "eval"}, "eval"),
                                        ({"MMPTB_ROLE": "agent", "EVAL_SPLIT": "val"}, "val")):
                with self.subTest(source=source.relative_to(ROOT), overrides=overrides), tempfile.TemporaryDirectory() as td:
                    root = Path(td).resolve()
                    bin_dir = root / "bin"
                    bin_dir.mkdir()
                    for command in ("sleep", "nvidia-smi"):
                        stub = bin_dir / command
                        stub.write_text("#!/bin/sh\nexit 0\n")
                        stub.chmod(0o755)
                    python_shim(bin_dir / "python3", root / "python_forward.py")
                    (root / "python_forward.py").write_text(
                        "import os, sys\nos.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n")
                    tests, workspace, logs = root / "tests", root / "workspace", root / "logs"
                    tests.mkdir()
                    (workspace / "final_model").mkdir(parents=True)
                    (workspace / "final_model/config.json").write_text("{}")
                    (tests / "evaluate.py").write_text(PROBE + """from pathlib import Path
result.update(accuracy=0.25, n=len(result['indices']))
Path(sys.argv[sys.argv.index('--json-output-file') + 1]).write_text(json.dumps(result))
""")
                    script = source.read_text()
                    for name, old, new in (("TESTS", "/tests", tests),
                                           ("WORKSPACE", "/home/agent/workspace", workspace),
                                           ("LOGS_DIR", "/logs/verifier", logs)):
                        assignment = name + '="' + old + '"'
                        self.assertEqual(script.count(assignment), 1)
                        script = script.replace(assignment, name + "=" + shlex.quote(str(new)))
                    helper = ROOT / "src/eval" if source == TEMPLATE else source.parent
                    env = clean_env(bin_dir)
                    env.update(image_policy_env(ROOT / "eval_omni/Dockerfile", source.parent / "Dockerfile"))
                    env.update(overrides, PYTHONPATH=str(helper))
                    run = subprocess.run(["bash"], input=script, env=env,
                                         capture_output=True, text=True, timeout=20)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    resolved = json.loads((logs / "metrics.json").read_text())
                    self.assertEqual(resolved["role"], "verifier")
                    self.assertEqual(resolved["split"], expected)
                    self.assertEqual(resolved["dataset_split"], "test" if expected == "eval" else "dev")
                    self.assertGreater(resolved["n"], 0)
                    self.assertEqual((logs / "reward.txt").read_text().strip(), "0.25")
                    self.assertNotIn("forcing val", run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
