import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_harnesses", ROOT / "eval_omni/prepare_harnesses.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class HarnessDeliveryTests(unittest.TestCase):
    def test_pinned_checkout_and_patch_reconstruct_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            upstream = root / "upstream"
            upstream.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(upstream), *args], text=True).strip()
            git("init", "--quiet")
            (upstream / "model.py").write_text("value = 1\n")
            git("add", "model.py")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base")
            pinned = git("rev-parse", "HEAD")
            (upstream / "model.py").write_text("value = 2\n")
            git("add", "model.py")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "later")
            patch = root / "fix.patch"
            patch.write_text("--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-value = 1\n+value = 3\n")
            expected = root / "expected"
            expected.mkdir()
            (expected / "model.py").write_text("value = 3\n")
            config = {"name": "fixture", "url": str(upstream), "commit": pinned, "paths": ["."],
                      "patches": [{"path": "fix.patch", "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}],
                      "source_tree": prep.source_tree(expected)}
            dest = root / "delivered"
            prep.prepare(config, dest, root)
            self.assertEqual((dest / "fixture/model.py").read_text(), "value = 3\n")
            # An idempotent call validates existing source rather than refetching.
            prep.prepare(config, dest, root)
            (dest / "fixture/model.py").write_text("local change\n")
            with self.assertRaises(ValueError):
                prep.prepare(config, dest, root)
            self.assertEqual((dest / "fixture/model.py").read_text(), "local change\n")
            patch.write_text("corrupt patch\n")
            with self.assertRaisesRegex(ValueError, "patch checksum"):
                prep.prepare(config, root / "new-dest", root)

    def test_checks_detect_missing_or_extra_source_but_ignore_bytecode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "model.py").write_text("value = 1\n")
            expected = {"name": "fixture", "source_tree": prep.source_tree(root)}
            (root / "__pycache__").mkdir()
            (root / "__pycache__/model.pyc").write_bytes(b"bytecode")
            prep.verify(root, expected)
            (root / "extra.py").write_text("unexpected\n")
            with self.assertRaises(ValueError):
                prep.verify(root, expected)
            (root / "extra.py").unlink()
            (root / "model.py").unlink()
            with self.assertRaises(ValueError):
                prep.verify(root, expected)


if __name__ == "__main__":
    unittest.main()
